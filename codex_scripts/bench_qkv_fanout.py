#!/usr/bin/env python3
"""One-rack QKV producer/consumer benchmark; CPU build and Slurm launch included.

submit --root TASK --source REPO --image IMAGE --python PYTHON --fabric-lib SO
       --nodes 'nvl576d002-R01-T[01-08]'
Fixed-resource protocol comparisons with exact GPU payload checks.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

CUDA = r'''
#include "afd_fmha_transport.cu"
#include <cstring>

__device__ __forceinline__ int64_t stamp() {
  int64_t value;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));return value;
}
struct PackedEdges { int64_t values[32][5]; };
template<int Rows>
__global__ void publish_qkv_packed_kernel(PublishQKVParams p,PackedEdges edges,int64_t* ready) {
  const int edge=blockIdx.x/4, pack=(blockIdx.x%4)*256+threadIdx.x;
  const auto* d=edges.values[edge];
  auto* qout=reinterpret_cast<__nv_bfloat16*>(d[0])+int64_t(p.slot)*d[4]*8192;
  auto* cache=reinterpret_cast<__nv_bfloat16*>(d[1]);
#pragma unroll
  for(int r=0;r<Rows;++r) {
    const int row=edge*Rows+r;
    reinterpret_cast<uint4*>(qout+r*8192)[pack]=reinterpret_cast<const uint4*>(p.q+row*p.q_stride)[pack];
    if(pack<64) {
      const int64_t loc=p.loc_i64 ? static_cast<const int64_t*>(p.out_loc)[row]
                                : static_cast<const int*>(p.out_loc)[row];
      const int64_t dst=(int64_t(p.layer)*d[2]+loc)*512+pack*8;
      *reinterpret_cast<uint4*>(cache+dst)=reinterpret_cast<const uint4*>(p.k+row*p.k_stride)[pack];
      *reinterpret_cast<uint4*>(cache+dst+d[3]*d[2]*512)=reinterpret_cast<const uint4*>(p.v+row*p.v_stride)[pack];
    }
  }
  __syncthreads();
  if(threadIdx.x==0) {
    int arrived;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;" : "=r"(arrived) : "l"(p.completion_counter) : "memory");
    if(arrived==int(gridDim.x)-1) {
      atomicExch(p.completion_counter,0);store_release_system(ready,p.ready_value);
    }
  }
}

__device__ __forceinline__ __nv_bfloat16 pattern(int64_t i,int epoch,int kind) {
  return __float2bfloat16(float((i+epoch*17+kind*29)%127)-63.f);
}
__global__ void fill_input(__nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,int rows,int epoch) {
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*8192;i+=gridDim.x*blockDim.x)
    q[i]=pattern(i,epoch,0);
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*512;i+=gridDim.x*blockDim.x) {
    k[i]=pattern(i,epoch,1);v[i]=pattern(i,epoch,2);
  }
}
__global__ void check_ack(const __nv_bfloat16* q,const __nv_bfloat16* kv,
    int64_t* ack,int* errors,int rows,int q_slot_stride,int source,int cache_rows,int layers,int epoch) {
  const int slot=epoch%2,layer=epoch%layers;
  int bad=0;
  for(int i=threadIdx.x;i<rows*8192;i+=blockDim.x)
    bad+=__bfloat16_as_ushort(q[slot*q_slot_stride+i])!=__bfloat16_as_ushort(pattern(source*rows*8192+i,epoch,0));
  for(int i=threadIdx.x;i<rows*512;i+=blockDim.x) {
    const int row=i/512,loc=(row*7919+source*17+5)%cache_rows;
    const int64_t offset=(int64_t(layer)*cache_rows+loc)*512+i%512;
    bad+=__bfloat16_as_ushort(kv[offset])!=__bfloat16_as_ushort(pattern(source*rows*512+i,epoch,1));
    bad+=__bfloat16_as_ushort(kv[offset+int64_t(layers)*cache_rows*512])!=__bfloat16_as_ushort(pattern(source*rows*512+i,epoch,2));
  }
  if(bad)atomicAdd(errors,bad);
  __syncthreads();
  if(threadIdx.x==0)store_release_system(ack+source,epoch);
}


// LL64 is scalar-64 atomic data+tag. LL128 relies on NVLink's ordered
// 128-byte line stores (120 data + 8 tag), as in NCCL's LL128 protocol.
// Never substitute a vector-atomicity assumption for the LL128 hardware contract.
struct Mailboxes { uint64_t* ptr[32]; };
__device__ __forceinline__ ulong2 load_pair(const uint64_t* p) {
  ulong2 v;asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
      : "=l"(v.x),"=l"(v.y) : "l"(p) : "memory");return v;
}
__device__ __forceinline__ void store_pair(uint64_t* p,ulong2 v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};"
      :: "l"(p),"l"(v.x),"l"(v.y) : "memory");
}
__device__ __forceinline__ uint64_t source_word(const uint64_t* q,const uint64_t* k,
    const uint64_t* v,int row,int word) {
  return word<2048 ? q[row*2048+word] : word<2176 ? k[row*128+word-2048]
       : word<2304 ? v[row*128+word-2176] : 0;
}
__device__ __forceinline__ void destination_word(uint64_t* q,uint64_t* kv,int row,
    int word,uint64_t value,int source,int cache_rows,int layers,int epoch) {
  if(word<2048) q[(epoch%2)*8192+row*2048+word]=value;
  else if(word<2304) {
    const int loc=(row*7919+source*17+5)%cache_rows;
    const int64_t dst=(int64_t(epoch%layers)*cache_rows+loc)*128;
    kv[dst+(word-2048)%128+(word>=2176 ? int64_t(layers)*cache_rows*128 : 0)]=value;
  }
}
template<bool LL128>
__global__ void publish_qkv_lamport_kernel(const uint64_t* q,const uint64_t* k,
    const uint64_t* v,Mailboxes boxes,int rows,int blocks,int epoch) {
  constexpr int packs=LL128 ? 1232 : 2304;
  const unsigned edge=blockIdx.x/blocks,lane=(blockIdx.x%blocks)*blockDim.x+threadIdx.x;
  auto* out=boxes.ptr[edge]+(epoch%2)*18432;
  for(unsigned i=lane;i<rows*packs;i+=blocks*blockDim.x) {
    const unsigned r=i/packs,p=i%packs,row=edge*rows+r;
    ulong2 value;
    if constexpr(LL128) {
      const unsigned word=2*p-(p>>3);
      value.x=source_word(q,k,v,row,word);
      value.y=(p%8==7) ? uint64_t(epoch) : source_word(q,k,v,row,word+1);
    } else {
      const auto data=source_word(q,k,v,row,p);
      value.x=(uint64_t(epoch)<<32)|uint32_t(data);
      value.y=(uint64_t(epoch)<<32)|uint32_t(data>>32);
    }
    store_pair(out+i*2,value);
  }
}
// Fewer resident CTAs need more independent memory requests per warp.
// Prefetch a register tile before issuing its NVLink stores; the two phases
// preserve memory-level parallelism without occupying one CTA per copy stripe.
template<int WarpsPerPeer,int Depth,int Threads>
__global__ __launch_bounds__(Threads) void publish_qkv_pipelined_kernel(const uint64_t* q,const uint64_t* k,
    const uint64_t* v,Mailboxes boxes,int peers,int rows,int epoch) {
  const unsigned warp=blockIdx.x*(Threads/32)+threadIdx.x/32,edge=warp/WarpsPerPeer;
  if(edge>=peers)return;
  const unsigned lane=(warp%WarpsPerPeer)*32+threadIdx.x%32;
  constexpr unsigned stride=WarpsPerPeer*32;
  auto* out=boxes.ptr[edge]+(epoch%2)*18432;
  for(unsigned base=lane;base<rows*1232;base+=Depth*stride) {
    ulong2 tile[Depth];
#pragma unroll
    for(unsigned u=0;u<Depth;++u) {
      const unsigned i=base+u*stride;
      if(i<rows*1232) {
        const unsigned row=edge*rows+i/1232,p=i%1232,word=2*p-(p>>3);
        tile[u].x=source_word(q,k,v,row,word);
        tile[u].y=(p%8==7) ? uint64_t(epoch) : source_word(q,k,v,row,word+1);
      }
    }
#pragma unroll
    for(unsigned u=0;u<Depth;++u) {
      const unsigned i=base+u*stride;
      if(i<rows*1232)store_pair(out+2*i,tile[u]);
    }
  }
}
// Two LL128 lines contain 30 data words. Load their 15 aligned 16B pairs,
// then shuffle the upper half-group by one word to open both flag positions.
// Padding is emitted only to the final 128B line of each Q/K/V segment.
template<int Words,int Warps>
__device__ __forceinline__ void push_segment(const uint64_t* source,uint64_t* inbox,
    int peer,int rows,int epoch,unsigned warp,unsigned offset) {
  constexpr unsigned packs=((Words+14)/15)*8;
  const unsigned lane=threadIdx.x%32;
  for(unsigned row=0;row<rows;++row) {
    const auto* src=source+(peer*rows+row)*Words;
    auto* dst=inbox+(epoch%2)*18432+(row*1240+offset)*2;
    for(unsigned p=warp*32+lane;p<((packs+31)/32)*32;p+=Warps*32) {
      const unsigned l=p%16,word=(p/16)*30+l*2;
      ulong2 value{};
      if(l<15 && word<Words)value=*reinterpret_cast<const ulong2*>(src+word);
      const uint64_t shifted=__shfl_up_sync(0xffffffff,value.y,1,16);
      if(l>=8) {value.y=value.x;value.x=shifted;}
      if(l%8==7)value.y=epoch;
      if(p<packs)store_pair(dst+p*2,value);
    }
  }
}
// Balance work across rows instead of restarting a partially-filled warp stripe
// for every row. Eight-lane packing tolerates row boundaries at any 128B line.
template<int Words,int Warps>
__device__ __forceinline__ void push_segment_balanced(const uint64_t* source,uint64_t* inbox,
    int peer,int rows,int epoch,unsigned warp,unsigned offset) {
  constexpr unsigned packs=((Words+14)/15)*8;
  const unsigned lane=threadIdx.x%32;
  for(unsigned i=warp*32+lane;i<((rows*packs+31)/32)*32;i+=Warps*32) {
    const unsigned row=i/packs,p=i%packs,line=p/8,l=p%8;
    const unsigned word=((line*15)&~1u)+l*2;
    ulong2 value{};
    if(row<rows && word<Words)value=*reinterpret_cast<const ulong2*>(source+(peer*rows+row)*Words+word);
    const uint64_t next=__shfl_down_sync(0xffffffff,value.x,1,8);
    if(line%2) {value.x=value.y;value.y=next;}
    if(l==7)value.y=epoch;
    if(row<rows)store_pair(inbox+(epoch%2)*18432+(row*1240+offset+p)*2,value);
  }
}
template<int WarpsPerPeer,bool Balanced=false>
__global__ __launch_bounds__(1024) void publish_qkv_segmented_kernel(const uint64_t* q,
    const uint64_t* k,const uint64_t* v,Mailboxes boxes,int peers,int rows,int epoch) {
  const unsigned warp=blockIdx.x*32+threadIdx.x/32,peer=warp/WarpsPerPeer,w=warp%WarpsPerPeer;
  if(peer>=peers)return;
  if constexpr(Balanced) {
    if(w<WarpsPerPeer-2)push_segment_balanced<2048,WarpsPerPeer-2>(q,boxes.ptr[peer],peer,rows,epoch,w,0);
    else if(w==WarpsPerPeer-2)push_segment_balanced<128,1>(k,boxes.ptr[peer],peer,rows,epoch,0,1096);
    else push_segment_balanced<128,1>(v,boxes.ptr[peer],peer,rows,epoch,0,1168);
    return;
  }
  if(w<WarpsPerPeer-2)push_segment<2048,WarpsPerPeer-2>(q,boxes.ptr[peer],peer,rows,epoch,w,0);
  else if(w==WarpsPerPeer-2)push_segment<128,1>(k,boxes.ptr[peer],peer,rows,epoch,0,1096);
  else push_segment<128,1>(v,boxes.ptr[peer],peer,rows,epoch,0,1168);
}
template<bool LL128,bool FlagsOnly=false,bool Segmented=false>
__global__ void receive_lamport_kernel(const uint64_t* inbox,uint64_t* q,uint64_t* kv,
    int64_t* done,int* counter,int64_t* times,int rows,int source,int cache_rows,
    int layers,int epoch,int* errors) {
  constexpr int packs=Segmented ? 1240 : LL128 ? 1232 : 2304;
  const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned cta=blockIdx.x;
  auto* in=inbox+(epoch%2)*18432;
  int64_t first=0;const int64_t deadline=stamp()+10000000000LL;
  // Round to a warp so even the final partial warp participates in votes.
  for(unsigned i=lane;i<((rows*packs+31)/32)*32;i+=gridDim.x*blockDim.x) {
    const bool valid=i<rows*packs;
    const unsigned r=i/packs,p=i%packs;
    ulong2 value{};bool pending;unsigned spins=0;
    do {
      if constexpr(FlagsOnly) {
        if(valid && p%8==7)asm volatile("ld.volatile.global.u64 %0,[%1];"
            : "=l"(value.y) : "l"(in+i*2+1) : "memory");
      } else if(valid)value=load_pair(in+i*2);
      pending=valid && (LL128 ? (p%8==7 && value.y!=uint64_t(epoch))
          : ((value.x>>32)!=uint32_t(epoch) || (value.y>>32)!=uint32_t(epoch)));
      pending=__any_sync(0xffffffff,pending);
      if((++spins&255)==0 && stamp()>deadline) {atomicAdd(errors,1);asm volatile("trap;");}
    } while(pending);
    // Reload after observing all flags: lanes may have loaded before flag lanes.
    if constexpr(LL128) if(valid)value=load_pair(in+i*2);
    if(!first)first=stamp();
    if(valid) {
      if constexpr(LL128) {
        const unsigned segment=p<1096 ? 0 : p<1168 ? 1 : 2;
        const unsigned local_p=Segmented ? p-(segment==0 ? 0 : segment==1 ? 1096 : 1168) : p;
        const unsigned local_word=2*local_p-(local_p>>3);
        const unsigned word=local_word+(Segmented ? (segment==0 ? 0 : segment==1 ? 2048 : 2176) : 0);
        const unsigned limit=Segmented ? (segment==0 ? 2048 : 128) : 2304;
        if(local_word<limit)destination_word(q,kv,r,word,value.x,source,cache_rows,layers,epoch);
        if(p%8!=7 && local_word+1<limit)destination_word(q,kv,r,word+1,value.y,source,cache_rows,layers,epoch);
      } else destination_word(q,kv,r,p,uint32_t(value.x)|(uint64_t(uint32_t(value.y))<<32),
                             source,cache_rows,layers,epoch);
    }
  }
  __syncthreads();
  if(threadIdx.x==0) {
    times[cta*2]=first;times[cta*2+1]=stamp();
    int arrived;asm volatile("atom.acq_rel.gpu.global.add.s32 %0,[%1],1;"
        : "=r"(arrived) : "l"(counter) : "memory");
    if(arrived==int(gridDim.x*gridDim.y)-1) {atomicExch(counter,0);store_release_system(done+source,epoch);}
  }
}

__global__ void receiver_poll_gate_kernel(int64_t* signal,int64_t epoch) {
  store_release_system(signal,epoch);
}

// Both endpoints use the publisher's clock. No cross-GPU clock subtraction.
__global__ void timing_mark_kernel(int64_t* times) { times[0]=stamp(); }
__global__ void completion_time_kernel(const int64_t* done,int64_t* times,int peers,int64_t epoch) {
  const int lane=threadIdx.x;const int64_t deadline=stamp()+10000000000LL;
  bool pending=lane<peers;unsigned spins=0;
  do {
    if(pending) {
      int64_t value;asm volatile("ld.acquire.sys.global.u64 %0,[%1];"
          : "=l"(value) : "l"(done+lane) : "memory");
      pending=value!=epoch;
    }
    if((++spins&255)==0 && stamp()>deadline)asm volatile("trap;");
  } while(__any_sync(0xffffffff,pending));
  if(lane==0)times[1]=stamp();
}

// Matched receiver-ready observation for the existing flag protocol.
__global__ void receive_ready_ack_kernel(const int64_t* ready,int64_t* done,int64_t epoch) {
  const int64_t deadline=stamp()+10000000000LL;unsigned spins=0;int64_t value;
  do {
    asm volatile("ld.acquire.sys.global.u64 %0,[%1];" : "=l"(value) : "l"(ready) : "memory");
    if((++spins&255)==0 && stamp()>deadline)asm volatile("trap;");
  } while(value!=epoch);
  store_release_system(done,epoch);
}
struct QkvBench {
  using T=tvm::ffi::TensorView;
  static void mark(T times) {
    host::LaunchKernel(1,1,times.device())(timing_mark_kernel,static_cast<int64_t*>(times.data_ptr()));
  }
  static void complete(T done,T times,int64_t peers,int64_t epoch) {
    host::RuntimeCheck(peers>0 && peers<=32);
    host::LaunchKernel(1,32,times.device())(completion_time_kernel,
        static_cast<const int64_t*>(done.data_ptr()),static_cast<int64_t*>(times.data_ptr()),int(peers),epoch);
  }
  static void push(T q,T k,T v,T host_boxes,int64_t epoch,int64_t rows,int64_t variant,
      int64_t blocks,int64_t threads) {
    host::RuntimeCheck(host_boxes.size(0)<=32 && epoch>0 && epoch<0x80000000LL);
    Mailboxes boxes{};std::memcpy(&boxes,host_boxes.data_ptr(),host_boxes.size(0)*sizeof(uint64_t*));
    if(variant>=44) {
      const int warps=(variant==44 || variant==46) ? 16 : 8;
      host::LaunchKernel((host_boxes.size(0)*warps+31)/32,1024,q.device())(
          variant==44 ? publish_qkv_segmented_kernel<16> : variant==45 ? publish_qkv_segmented_kernel<8>
          : variant==46 ? publish_qkv_segmented_kernel<16,true> : publish_qkv_segmented_kernel<8,true>,
          static_cast<const uint64_t*>(q.data_ptr()),static_cast<const uint64_t*>(k.data_ptr()),
          static_cast<const uint64_t*>(v.data_ptr()),boxes,int(host_boxes.size(0)),int(rows),int(epoch));
      return;
    }
    if(variant>=40) {
      const int warps=variant==40 ? 16 : variant==42 ? 4 : 8;
      const int cta_threads=variant==43 ? 512 : 1024;
      const auto kernel=variant==40 ? publish_qkv_pipelined_kernel<16,8,1024>
          : variant==41 ? publish_qkv_pipelined_kernel<8,8,1024>
          : variant==42 ? publish_qkv_pipelined_kernel<4,8,1024> : publish_qkv_pipelined_kernel<8,16,512>;
      host::LaunchKernel((host_boxes.size(0)*warps+cta_threads/32-1)/(cta_threads/32),cta_threads,q.device())(kernel,
          static_cast<const uint64_t*>(q.data_ptr()),static_cast<const uint64_t*>(k.data_ptr()),
          static_cast<const uint64_t*>(v.data_ptr()),boxes,int(host_boxes.size(0)),int(rows),int(epoch));
      return;
    }
    host::LaunchKernel(blocks*host_boxes.size(0),threads,q.device())(
      variant==31 ? publish_qkv_lamport_kernel<false> : publish_qkv_lamport_kernel<true>,
      static_cast<const uint64_t*>(q.data_ptr()),static_cast<const uint64_t*>(k.data_ptr()),
      static_cast<const uint64_t*>(v.data_ptr()),boxes,int(rows),int(blocks),int(epoch));
  }
  static void receive(T inbox,T q,T kv,T done,T counter,T times,T errors,int64_t source,
      int64_t epoch,int64_t rows,int64_t variant,int64_t blocks,int64_t threads) {
    host::LaunchKernel(blocks,threads,q.device())(
      variant>=44 ? receive_lamport_kernel<true,false,true> : variant==31 ? receive_lamport_kernel<false>
      : variant==36 ? receive_lamport_kernel<true,true> : receive_lamport_kernel<true>,
      static_cast<const uint64_t*>(inbox.data_ptr()),static_cast<uint64_t*>(q.data_ptr()),
      static_cast<uint64_t*>(kv.data_ptr()),static_cast<int64_t*>(done.data_ptr()),
      static_cast<int*>(counter.data_ptr()),static_cast<int64_t*>(times.data_ptr()),int(rows),
      int(source),int(kv.size(2)),int(kv.size(1)),int(epoch),static_cast<int*>(errors.data_ptr()));
  }
  static void observe(T local,int64_t signal,int64_t done,int64_t epoch) {
    host::LaunchKernel(1,1,local.device())(receive_ready_ack_kernel,
        reinterpret_cast<const int64_t*>(signal),reinterpret_cast<int64_t*>(done),epoch);
  }
  static void gate(T signal,int64_t epoch) {
    host::LaunchKernel(1,1,signal.device())(receiver_poll_gate_kernel,static_cast<int64_t*>(signal.data_ptr()),epoch);
  }
  static void signal(T local,int64_t address,int64_t epoch) {
    host::LaunchKernel(1,1,local.device())(receiver_poll_gate_kernel,reinterpret_cast<int64_t*>(address),epoch);
  }
  static void fill(T q,T k,T v,int64_t epoch) {
    host::LaunchKernel(128,256,q.device())(fill_input,static_cast<__nv_bfloat16*>(q.data_ptr()),
      static_cast<__nv_bfloat16*>(k.data_ptr()),static_cast<__nv_bfloat16*>(v.data_ptr()),int(q.size(0)),int(epoch));
  }
  static void check(T q,T kv,T ack,T errors,int64_t source,int64_t epoch) {
    host::LaunchKernel(1,256,q.device())(check_ack,static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const __nv_bfloat16*>(kv.data_ptr()),static_cast<int64_t*>(ack.data_ptr()),
      static_cast<int*>(errors.data_ptr()),int(q.size(1)),int(q.stride(0)),int(source),int(kv.size(2)),int(kv.size(1)),int(epoch));
  }
  static void publish(T q,T k,T v,T loc,T offsets,T qd,T kd,T ready,T counter,T timing,T host_edges,
      int64_t epoch,int64_t layers,int64_t variant,int64_t blocks,int64_t threads) {
    PublishQKVParams p{.q=static_cast<const __nv_bfloat16*>(q.data_ptr()),
      .k=static_cast<const __nv_bfloat16*>(k.data_ptr()),.v=static_cast<const __nv_bfloat16*>(v.data_ptr()),
      .out_loc=loc.data_ptr(),.source_offsets=static_cast<const int64_t*>(offsets.data_ptr()),
      .q_desc=static_cast<const int64_t*>(qd.data_ptr()),.kv_desc=static_cast<const int64_t*>(kd.data_ptr()),
      .ready_desc=static_cast<const int64_t*>(ready.data_ptr()),.completion_counter=static_cast<int*>(counter.data_ptr()),
      .ready_value=epoch,.q_stride=q.stride(0),.k_stride=k.stride(0),.v_stride=v.stride(0),
      .q_edges=int(qd.size(0)),.kv_edges=int(kd.size(0)),.ready_edges=int(ready.size(0)),
      .layer=int(epoch%layers),.slot=int(epoch%2),.rows=int(q.size(0)),.loc_i64=false};
    if(variant==2) {
      AfdFmhaTransportKernel<128>::publish_qkv(q,k,v,loc,offsets,qd,kd,ready,counter,
                                            epoch%layers,epoch%2,epoch);
    } else if(variant==0) {
      int grid=(p.rows*8192/4+threads-1)/threads;
      if(blocks>0)grid=std::min(grid,int(blocks));
      host::LaunchKernel(grid,threads,q.device())(publish_qkv_kernel<128,false,true>,p);
    } else if(variant==1) {
      host::LaunchKernel(blocks*p.q_edges,threads,q.device())(publish_qkv_edges_kernel<128>,p,int(blocks));
    } else if(variant==28) {
      host::RuntimeCheck(p.q_edges<=32 && p.rows%p.q_edges==0 && threads==256);
      PackedEdges edges{};std::memcpy(&edges,host_edges.data_ptr(),p.q_edges*5*sizeof(int64_t));
      auto* signal=static_cast<int64_t*>(timing.data_ptr());
      const host::LaunchKernel launch(4*p.q_edges,256,q.device());
      switch(p.rows/p.q_edges) {
        case 2:launch(publish_qkv_packed_kernel<2>,p,edges,signal);break;
        case 3:launch(publish_qkv_packed_kernel<3>,p,edges,signal);break;
        case 4:launch(publish_qkv_packed_kernel<4>,p,edges,signal);break;
        default:host::RuntimeCheck(false,"Packed study requires 2/3/4 rows per edge");
      }
    } else host::RuntimeCheck(false,"Unknown publication variant");
  }
};
'''


def prepare(args):
    (args.root/'tmp').mkdir(exist_ok=True)
    os.environ.update(TVM_FFI_CACHE_DIR=str(args.root / 'tvm'),
                      TVM_FFI_CUDA_ARCH_LIST='10.0a', MAX_JOBS='8',TMPDIR=str(args.root/'tmp'))
    from tvm_ffi.cpp import load_inline
    from minisgl.kernel.utils import KERNEL_PATH
    source = '#include "' + str(KERNEL_PATH / 'jit/afd_fmha_transport.cu') + '"'
    code = CUDA.replace('#include "afd_fmha_transport.cu"', source)
    code += '\n' + '\n'.join(
        f'TVM_FFI_DLL_EXPORT_TYPED_FUNC({name}, ({target}));'
        for name, target in [('publish','QkvBench::publish'),('fill','QkvBench::fill'),
                             ('signal','QkvBench::signal'),
                             ('mark','QkvBench::mark'),('complete','QkvBench::complete'),
                             ('check','QkvBench::check'),('observe','QkvBench::observe'),('gate','QkvBench::gate'),('push','QkvBench::push'),('receive','QkvBench::receive'),('wait','AfdFmhaTransportKernel<128>::wait_ready')])
    load_inline('qkv_fanout_bench',cuda_sources=code,
                extra_include_paths=[str(KERNEL_PATH/'include')],
                extra_cuda_cflags=['-std=c++20','-O3','--expt-relaxed-constexpr'])
    libs=list((args.root/'tvm').glob('*/qkv_fanout_bench.so'))
    if len(libs)!=1:
        raise RuntimeError(f'Expected one prepared module: {libs}')
    for flag,name in [('sass','sass.txt'),('resource-usage','resources.txt')]:
        with (args.root/name).open('w') as out:
            subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-'+flag,str(libs[0])],check=True,stdout=out)
    # Reject register spilling before allocating GPUs for the pooled designs.
    import re
    usage=(args.root/'resources.txt').read_text()
    for match in re.finditer(r'Function [^\n]*publish_qkv_(?:pipelined|segmented)[^\n]*\n([^\n]*)',usage):
        fields=dict(re.findall(r'(REG|STACK|LOCAL):(\d+)',match[1]))
        if any(int(fields[k]) for k in ('STACK','LOCAL')):
            raise RuntimeError('Pipelined communication spills: '+match[0])
    prepared=dict(module=str(libs[0]),fabric_lib=args.fabric_lib)
    if args.experiment=='integration':
        from minisgl.kernel.afd_qkv_lamport import module
        module()
        built=list((args.root/'tvm').glob('*/minisgl__afd_qkv_lamport*.so'))
        if len(built)!=1:raise RuntimeError(f'Expected one production Lamport module: {built}')
        prepared['lamport_module']=str(built[0])
    (args.root/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
    from minisgl.kernel.afd_fmha_transport import ensure_afd_fmha_transport_built
    ensure_afd_fmha_transport_built(128)
    print('PREPARED',libs[0],flush=True)


def worker(args):
    import importlib.util
    import torch
    import torch.distributed as dist
    from tvm_ffi import load_module
    import minisgl.kernel.fabric_memory as fabric
    global_rank=int(os.environ['SLURM_PROCID']);total=int(os.environ['SLURM_NTASKS'])
    world=args.group_size or total
    if total%world:raise ValueError('Group size must divide the allocation')
    rank=global_rank%world
    torch.cuda.set_device(int(os.environ['SLURM_LOCALID']))
    dist.init_process_group('gloo',rank=global_rank,world_size=total,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    group=None
    for start in range(0,total,world):
        candidate=dist.new_group(list(range(start,start+world)),backend='gloo')
        if start<=global_rank<start+world:group=candidate
    prepared=json.loads((args.root/'prepared.json').read_text())
    spec=importlib.util.spec_from_file_location('_minisgl_deepep_moe',prepared['fabric_lib'])
    ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
    fabric._load_extension=lambda:ext
    mod=load_module(prepared['module']);fanin=world-1
    integrated=load_module(prepared['lamport_module']) if args.experiment=='integration' else None
    if fanin<1:raise ValueError('Need a publisher and at least one consumer')
    device=torch.device('cuda',torch.cuda.current_device())
    def tensor(data,dtype=torch.int64):return torch.tensor(data,dtype=dtype,device=device)
    def describe(obj):return dict(handle=obj.handle,shape=list(obj.tensor.shape),offset=obj.byte_offset)
    imports=[];allocations={}
    def remote(d,dtype):
        obj=fabric.import_fabric_tensor(d['handle'],d['shape'],dtype=dtype,byte_offset=d['offset'],
                                       allocation=allocations.get(d['handle']))
        allocations[d['handle']]=obj.allocation
        imports.append(obj);return obj.tensor
    if rank==0:
        ack_owner,sender_ready_owner,done_owner=fabric.allocate_fabric_tensors([((fanin,),torch.int64),((2,8),torch.int64),((fanin,),torch.int64)])
        record={'ack':describe(ack_owner),'ready':describe(sender_ready_owner),'done':describe(done_owner)}
    else:
        # Model-like 94-layer / 128K-cache strides; initialize only touched rows.
        q_owner,kv_owner,ready_owner,inbox_owner=fabric.allocate_fabric_tensors([
            ((2,4,64,128),torch.bfloat16),
            ((2,args.layers,args.cache_rows,4,128),torch.bfloat16),((2,8),torch.int64),((2,18432),torch.int64)])
        record={k:describe(v) for k,v in [('q',q_owner),('kv',kv_owner),('ready',ready_owner),('inbox',inbox_owner)]}
    records=[None]*world;dist.all_gather_object(records,record,group=group)
    ack=ack_owner.tensor if rank==0 else remote(records[0]['ack'],torch.int64)
    done=done_owner.tensor if rank==0 else remote(records[0]['done'],torch.int64)
    receive_counter=torch.zeros(1,dtype=torch.int32,device=device)
    receive_times=torch.zeros((args.iterations+1,32,2),dtype=torch.int64,device=device)
    sender_ready=sender_ready_owner.tensor if rank==0 else remote(records[0]['ready'],torch.int64)
    shared_ready=ready_owner.tensor if rank==1 else remote(records[1]['ready'],torch.int64)
    errors=torch.zeros(1,dtype=torch.int32,device=device)
    timeout=torch.zeros(2,dtype=torch.int64,device=device)
    if rank==0:
        qs=[remote(r['q'],torch.bfloat16) for r in records[1:]]
        kvs=[remote(r['kv'],torch.bfloat16) for r in records[1:]]
        counters=torch.zeros(2,dtype=torch.int32,device=device)
        ready_desc=tensor([[shared_ready.data_ptr(),0,8]])
        sender_desc=tensor([[sender_ready.data_ptr(),0,8]])
        inboxes=[remote(r['inbox'],torch.int64) for r in records[1:]]
        if any(x.data_ptr()%128 for x in inboxes):raise ValueError('LL128 needs aligned 128-byte lines')
        host_boxes=torch.tensor([x.data_ptr() for x in inboxes],dtype=torch.int64)
        production_boxes=torch.tensor([[x.data_ptr(),x.stride(0)] for x in inboxes],dtype=torch.int64)
    results=[];receiver_records={}
    final_timing=args.experiment=='final'
    completion_times=torch.zeros((args.iterations+1,2),dtype=torch.int64,device=device)
    graph_times=torch.zeros(2,dtype=torch.int64,device=device)
    # Fixed resources: compare mechanisms, not launch-parameter combinations.
    # Production's existing 8-CTA launch is retained as the reference.
    variants={
        'protocol':[(1,8,256),(28,4,256),(31,4,256),(32,4,256),(36,4,256),(1,8,256)],
        'polling':[(32,4,256),(36,4,256),(39,4,256)],
        'production':[(0,0,256),(2,8,256),(2,8,256),(0,0,256)],
        'balance':[(32,4,256),(44,0,1024),(46,0,1024),(47,0,1024),(46,0,1024),(32,4,256)],
        'final':[(32,4,256),(46,0,1024),(-32,4,256)],
        'integration':[(48,4,256)],
        'packing':[(32,4,256),(40,0,1024),(44,0,1024),(45,0,1024),(44,0,1024),(32,4,256)],
        'resources':[(1,8,256),(28,4,256),(32,4,256),(-32,4,256),(40,0,1024),(41,0,1024),(42,0,1024),(43,0,512),(32,4,256)],
        'measurement':[(1,8,256),(28,4,256),(32,4,256),(-32,4,256),(36,4,256),(32,4,256),(28,4,256)],
    }[args.experiment]
    if args.profile and global_rank==0:torch.cuda.profiler.start()
    case_index=0
    for rows in (2,3,4):
        receiver_loc=tensor([(r*7919+(rank-1)*17+5)%args.cache_rows for r in range(rows)],torch.int32)
        if rank==0:
            q=torch.empty((fanin*rows,8192),dtype=torch.bfloat16,device=device)
            k=torch.empty((fanin*rows,512),dtype=torch.bfloat16,device=device);v=torch.empty_like(k)
            loc=tensor([(r*7919+s*17+5)%args.cache_rows for s in range(fanin) for r in range(rows)],torch.int32)
            offsets=tensor([s*rows for s in range(fanin+1)])
            qd=tensor([[x.data_ptr(),0,0,64,64,4,s] for s,x in enumerate(qs)])
            kd=tensor([[x.data_ptr(),0,0,4,4,args.cache_rows,args.layers,s] for s,x in enumerate(kvs)])
            host_edges=torch.tensor([[x.data_ptr(),y.data_ptr(),args.cache_rows,args.layers,4]
                for x,y in zip(qs,kvs)],dtype=torch.int64)
        for pass_id,sequence in enumerate((variants,list(reversed(variants)))):
            for variant,blocks,threads in sequence:
                event_control=variant<0;variant=abs(variant)
                lamport=variant>=31
                at_sender=variant==28
                if args.profile and global_rank==0:torch.cuda.nvtx.range_push(f'qkv-case-{len(results)}')
                def reset():
                    errors.zero_();timeout.zero_();receive_counter.zero_();receive_times.zero_()
                    completion_times.zero_();graph_times.zero_()
                    if rank==0:ack.zero_();done.zero_();counters.zero_();sender_ready.zero_()
                    if rank>0:ready_owner.tensor.zero_();inbox_owner.tensor.zero_()
                    torch.cuda.synchronize();dist.barrier()
                def roundtrip(epoch,events=None):
                    slot=epoch%2
                    if rank==0:
                        mod.fill(q,k,v,epoch)
                        if events:events[0].record()
                        if final_timing:mod.mark(completion_times[epoch])
                        if variant==48:
                            integrated.publish(q,k,v,production_boxes,slot,epoch)
                        elif lamport:
                            mod.push(q,k,v,host_boxes,epoch,rows,variant,blocks,threads)
                        else:
                            mod.publish(q,k,v,loc,offsets,qd,kd,sender_desc if at_sender else ready_desc,
                                        counters[slot:slot+1],sender_ready[slot],host_edges,
                                        epoch,args.layers,variant,blocks,threads)
                        if variant==39:mod.gate(sender_ready[slot],epoch)
                        if final_timing:
                            mod.complete(done,completion_times[epoch],fanin,epoch)
                            if events:events[1].record()
                        else:
                            if events:events[1].record()
                            mod.wait(done.data_ptr(),fanin,timeout,30000,epoch)
                        mod.wait(ack.data_ptr(),fanin,timeout,30000,epoch)
                    else:
                        if variant==48:
                            integrated.receive(inbox_owner.tensor,q_owner.tensor,kv_owner.tensor,receiver_loc,
                                               epoch%args.layers,slot,epoch,30000)
                            mod.signal(timeout,done[rank-1].data_ptr(),epoch)
                        elif lamport:
                            if variant==39:mod.wait(sender_ready[slot].data_ptr(),1,timeout,30000,epoch)
                            mod.receive(inbox_owner.tensor,q_owner.tensor,kv_owner.tensor,done,
                                receive_counter,receive_times[epoch],errors,rank-1,epoch,rows,variant,4 if variant>=40 else blocks,256 if variant>=40 else threads)
                        else:
                            signal=sender_ready if at_sender else shared_ready
                            mod.observe(timeout,signal[slot].data_ptr(),done[rank-1].data_ptr(),epoch)
                        # Q slot stride is four rows, independently of live payload rows.
                        view=q_owner.tensor[:, :rows]
                        mod.check(view,kv_owner.tensor,ack,errors,rank-1,epoch)
                reset();roundtrip(1);torch.cuda.synchronize();dist.barrier()
                if errors.item():raise RuntimeError(f'Warmup payload error rank={rank}')
                reset()
                events=[(torch.cuda.Event(enable_timing=True,external=True),
                         torch.cuda.Event(enable_timing=True,external=True)) for _ in range(args.iterations)] if rank==0 and (event_control or (not args.profile and not final_timing)) else []
                graph_events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    if final_timing:mod.mark(graph_times)
                    for epoch in range(1,args.iterations+1):roundtrip(epoch,events[epoch-1] if events else None)
                    if final_timing:mod.mark(graph_times[1:])
                # Capture records commands without executing them; reset before replay.
                if final_timing:
                    reset();graph_events[0].record();graph.replay();graph_events[1].record()
                    torch.cuda.synchronize();dist.barrier()
                    if errors.item() or timeout.any().item():raise RuntimeError('Graph warmup payload/timeout error')
                    cold_event_us=graph_events[0].elapsed_time(graph_events[1])*1000
                    g=graph_times.cpu().tolist();cold_device_us=(g[1]-g[0])/1000
                reset();graph_events[0].record();graph.replay();graph_events[1].record()
                torch.cuda.synchronize();dist.barrier()
                if errors.item() or timeout.any().item():raise RuntimeError(f'Payload/timeout error rank={rank}')
                if lamport and rank>0:
                    t=receive_times[1:,:4 if variant>=40 else blocks].cpu()
                    spans=(t[:,:,1].max(1).values-t[:,:,0].min(1).values).tolist()
                    receiver_records[str(case_index)]=dict(rows=rows,variant=variant,blocks=blocks,threads=threads,
                        first_arrival_to_local_store_us=[x/1000 for x in spans])
                if rank==0:
                    if counters.any().item() or not torch.all(ack==args.iterations).item():
                        raise RuntimeError('Publication counter or receiver ACK mismatch')
                    values=[a.elapsed_time(b)*1000 for a,b in events]
                    row=dict(rows=rows,variant=variant,blocks=blocks,threads=threads,pass_id=pass_id,cuda_events=bool(events),
                             graph_cuda_event_us=graph_events[0].elapsed_time(graph_events[1])*1000,
                             median_us=statistics.median(values) if values else None,
                             mean_us=statistics.mean(values) if values else None,samples_us=values)
                    if final_timing:
                        t=completion_times[1:].cpu().tolist();g=graph_times.cpu().tolist()
                        if any(a<=0 or b<=a for a,b in t) or g[1]<=g[0]:raise RuntimeError('Invalid GPU timing endpoints')
                        spans=[(b-a)/1000 for a,b in t]
                        row.update(measurement='receiver_done_roundtrip_upper_bound',profiled=args.profile,
                            completion_samples_us=spans,completion_median_us=statistics.median(spans),
                            completion_p95_us=sorted(spans)[int(.95*(len(spans)-1))],
                            checked_graph_device_us=(g[1]-g[0])/1000,
                            cold_graph_event_boundary_overhead_us=cold_event_us-cold_device_us,
                            graph_event_boundary_overhead_us=row['graph_cuda_event_us']-(g[1]-g[0])/1000,
                            event_scope='completion' if events else None)
                        if events:row['event_minus_device_median_us']=statistics.median(a-b for a,b in zip(values,spans))
                    if variant>=31:
                        peer_warps=16 if variant in (40,44,46) else 4 if variant==42 else 8
                        ctas=(fanin*peer_warps+threads//32-1)//(threads//32) if 40<=variant<48 else fanin*blocks
                        row.update(sender_ctas=ctas,sender_sm_upper_bound=ctas,sender_warps=ctas*(threads//32),
                                   register_tile_depth=1 if variant>=44 else 16 if variant==43 else 8 if variant>=40 else 1)
                    results.append(row);print(json.dumps({k:v for k,v in row.items() if 'samples' not in k}),flush=True)
                    (args.root/f'results-g{world}-p{global_rank//world}.json').write_text(json.dumps(results,indent=2)+'\n')
                case_index+=1
                del graph,events,graph_events
                if args.profile and global_rank==0:torch.cuda.nvtx.range_pop()
    if receiver_records:(args.root/f'receiver-r{global_rank}.json').write_text(json.dumps(receiver_records)+'\n')
    if args.profile and global_rank==0:torch.cuda.profiler.stop()
    dist.barrier();dist.destroy_process_group()


def submit(args):
    args.root.mkdir(parents=True,exist_ok=True)
    script=args.root/Path(__file__).name
    shutil.copyfile(__file__,script)
    base=['sbatch','--parsable','--propagate=NONE','--account=coreai_comparch_sysarch',
          '--no-requeue','--time=00:30:00']
    container=['srun',f'--container-image={args.image}','--container-mounts=/lustre:/lustre,/scratch:/scratch',
               '--container-mount-home','--no-container-remap-root','--kill-on-bad-exit=1']
    common=['env',f'PYTHONPATH={args.source}/python:{args.source}',args.python,str(script)]
    opts=['--root',str(args.root),'--source',str(args.source),'--fabric-lib',args.fabric_lib,
          '--iterations',str(args.iterations),'--cache-rows',str(args.cache_rows),'--layers',str(args.layers),
          '--experiment',args.experiment,'--group-size',str(args.group_size)]
    if args.profile:opts+=['--profile']
    cpu=base+['--partition=cpu','--qos=cpu-short','--nodes=1','--ntasks=1','--cpus-per-task=16','--mem=96G',
              '--job-name=qkv:prepare',f'--output={args.root}/prepare-%j.log',
              '--wrap='+shlex.join(container+common+['prepare']+opts)]
    if args.prepared:
        (args.root/'prepared.json').write_text(args.prepared.read_text())
        cpu_id=None
    else:
        cpu_id=subprocess.check_output(cpu,text=True).strip()
    hosts=subprocess.check_output(['scontrol','show','hostnames',args.nodes],text=True).split()
    racks={h.rsplit('-T',1)[0] for h in hosts}
    if len(racks)!=1:raise ValueError(f'Expected one rack: {racks}')
    gpu=base+['--partition=batch','--qos=short',f'--nodes={len(hosts)}','--ntasks-per-node=4',
              '--cpus-per-task=8','--gres=gpu:4','--exclusive',f'--nodelist={args.nodes}',
              f'--segment={len(hosts)}','--consolidate-segments',
              '--kill-on-invalid-dep=yes','--job-name=qkv:fanout',f'--output={args.root}/gpu-%j.log',
              '--wrap='+shlex.join(container+['env',f'MASTER_ADDR={hosts[0]}','MASTER_PORT=8072']+common+['worker']+opts)]
    if args.group_sizes:
        gpu[-1]='--wrap='+' && '.join(shlex.join(container+['env',f'MASTER_ADDR={hosts[0]}',
            f'MASTER_PORT={8072+i}']+common+['worker']+opts+['--group-size',str(g)])
            for i,g in enumerate(map(int,args.group_sizes.split(','))))
    if cpu_id:gpu.insert(-1,f'--dependency=afterok:{cpu_id}')
    gpu_id=subprocess.check_output(gpu,text=True).strip()
    record=dict(cpu=cpu_id,gpu=gpu_id,cpu_command=cpu,gpu_command=gpu)
    if args.profile:
        analysis=base+['--partition=cpu','--qos=cpu-short','--nodes=1','--ntasks=1',
            '--cpus-per-task=16','--mem=96G',f'--dependency=afterok:{gpu_id}',
            '--kill-on-invalid-dep=yes','--job-name=qkv:trace',f'--output={args.root}/analysis-%j.log',
            '--wrap='+shlex.join(container+common+['trace']+opts)]
        record.update(analysis=subprocess.check_output(analysis,text=True).strip(),analysis_command=analysis)
    (args.root/'jobs.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


def model_prepare(args):
    """Reuse a CPU-prepared model plan for a matched control/candidate pair."""
    template=args.model_template
    base=json.loads((template/'env.json').read_text())
    import csv
    case=list(csv.DictReader((template/'case.csv').open()))[0]
    ratio=args.model_ratio;ep=32//(ratio+1);batch=int(case['batch_per_attention_dp_lane'])
    if ep*(ratio+1)!=32 or ep<2:raise ValueError('Model comparison requires a 32-GPU, EP>=2 geometry')
    context=int(case['context_tokens']);case_id=f'i{context}-fep{ep}-r{ratio}-atp1-b{batch}-fp4'
    case.update(case_id=case_id,normalized_af_ratio=str(ratio),ffn_ep=str(ep),
                run_afd_argv=f'qwen3 {context} {batch} {ratio}:1 1 {ep}')
    build=args.root/'build';(build/'jit/cache').mkdir(parents=True);(build/'tmp').mkdir()
    os.environ.update(base,MINISGL_DEEPGEMM_BUILD_DIR=str(template.parent/'build/native'),
        DG_JIT_CACHE_DIR=str(build/'jit'),DG_JIT_USE_NVRTC='0',MAX_JOBS='8',
        CUDA_HOME='/usr/local/cuda',CUDA_PATH='/usr/local/cuda',CUDA_NVCC_EXECUTABLE='/usr/local/cuda/bin/nvcc',
        LIBRARY_PATH='/usr/local/cuda/lib64/stubs',TMPDIR=str(build/'tmp'))
    from minisgl.kernel import deepgemm
    reset=ratio*((batch+1)//2)
    for rows in range(ratio,reset+1,ratio):
        code,_,block,wave=deepgemm.generate_megamoe_kernel(ep,128,1152,rows,8,4096,1536,148,True,True,True)
        if rows+((rows+block-1)//block)*block*wave>1152:raise ValueError('Decode materialization exceeds capacity')
        if 'true, 0\n' not in code:raise ValueError('Unexpected reset template')
        code=code.replace('true, 0\n',f'true, {reset}\n')
        deepgemm.precompile_kernel('sm100_fp8_fp4_mega_moe',code,'100f')
    for scope in Path(base['DG_JIT_CACHE_DIR']).glob('ray-node-*/rank*/cache'):
        for kernel in (build/'jit/cache').glob('kernel.*'):
            target=scope/kernel.name
            if not target.exists():target.symlink_to(kernel,target_is_directory=True)
    os.environ.update(TVM_FFI_CACHE_DIR=str(build/'tvm'),TVM_FFI_CUDA_ARCH_LIST='10.0a')
    from minisgl.kernel.afd_fmha_transport import ensure_afd_fmha_transport_built
    ensure_afd_fmha_transport_built(128)
    modules=list((build/'tvm').glob('minisgl__afd_fmha_transport_edges*'))
    if len(modules)!=1:raise ValueError(f'Expected one prepared transport module: {modules}')
    cache=Path(base['TVM_FFI_CACHE_DIR'])
    scopes=list(cache.glob('ray-node-*'))+list(cache.glob('ray-node-*/rank*'))
    if not scopes:raise ValueError('Template has no prepared worker caches')
    for scope in scopes:
        target=scope/modules[0].name
        if target.exists():raise ValueError(f'Refuse to replace {target}')
        target.symlink_to(modules[0],target_is_directory=True)
    plans=[]
    for index,(name,enabled) in enumerate([('control','0'),('edge','1')]):
        plan=args.root/name;plan.mkdir()
        env=dict(base,MINISGL_AFD_QKV_EDGE=enabled,FASTAFD_DEFER_CUDA_EXTRACTION='1',
            AFD_ATTENTION_GPU_COUNT=str(ep*ratio),AFD_MLP_EP_SIZE=str(ep),FASTAFD_CASE_ID=case_id,
            FASTAFD_TASK_ROOT=str(plan),FASTAFD_RESULTS_ROOT=str(plan/'results'),
            RUN_DIR=str(plan/'results'/case_id),FASTAFD_MINISGL_PORT=str(8040+index*8),
            FASTAFD_CUDA_METRIC_PLAN=str(plan/'case.csv'),
            FASTAFD_CUDA_METRICS_ROOT=str(plan/'report/metrics'),
            FASTAFD_CUDA_EXTRACT_TEMP_ROOT=str(plan/'extract'))
        (plan/'env.json').write_text(json.dumps(env,indent=2)+'\n')
        launch=json.loads((template/'launch.json').read_text());launch[5]=f'{ratio}:1';launch[7]=str(ep)
        (plan/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        with (plan/'case.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(case));writer.writeheader();writer.writerow(case)
        plans.append(str(plan))
    (args.root/'plans.json').write_text(json.dumps(plans,indent=2)+'\n')
    text=(args.source/'scripts/experiments/afd/oci_hsg/run_afd.sh').read_text()
    start=text.index('from pathlib import Path\nimport fcntl',text.index('PROMPT_CACHE_ROOT='))
    builder=text[start:text.index('\nPY\nexport FASTAFD_SOURCE_REPO',start)]
    prompts=args.root/'prompts';prompts.mkdir()
    sys.argv=['prompt-cache',base['FASTAFD_PROMPT_BASE_FILE'],str(prompts/'prompts.txt'),str(prompts/'transform.json'),
        base['FASTAFD_PROMPT_CACHE_ROOT'],str(ep*ratio*batch),str(context),str(context),'qwen3',base['FASTAFD_MODEL_PATH'],
        '39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35','uniform','8192',str(ep*ratio),','.join([str(context)]*batch),'1']
    exec(compile(builder,'maintained_prompt_builder','exec'),{'__name__':'__main__'})
    from minisgl.afd_support import runtime_env
    os.environ['MINISGL_AFD_QKV_EDGE']='0'
    if runtime_env()['env_vars']['MINISGL_AFD_QKV_EDGE']!='0':raise RuntimeError('Control selector not forwarded')
    print('MODEL_PLANS_READY',plans,flush=True)


def model(args):
    args.root.mkdir(parents=True,exist_ok=True)
    source=args.source;script=source/'codex_scripts'/Path(__file__).name
    command=['srun',f'--container-image={args.image}','--container-mounts=/lustre:/lustre,/scratch:/scratch',
             '--container-mount-home','--no-container-remap-root','env',f'PYTHONPATH={source}/python:{source}',
             args.python,str(script),'model_prepare','--root',str(args.root),'--source',str(source),
             '--fabric-lib',args.fabric_lib,'--prepared',str(args.prepared),'--model-template',str(args.model_template),
             '--model-ratio',str(args.model_ratio)]
    cpu=['sbatch','--parsable','--propagate=NONE','--account=coreai_comparch_sysarch','--partition=cpu',
         '--qos=cpu-short','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=32G','--time=00:10:00',
         '--job-name=qkv:model-prepare',f'--output={args.root}/prepare-%j.log','--wrap='+shlex.join(command)]
    cpu_id=subprocess.check_output(cpu,text=True).strip()
    env=dict(os.environ,FASTAFD_PLAN_SEQUENCE=str(args.root/'plans.json'),
        FASTAFD_IMAGE=args.image,FASTAFD_SEQUENCE_ROOT=str(args.root),
        FASTAFD_SEQUENCE_SCRIPT=str(source/'codex_scripts/run_afd_plan_sequence.sbatch'),
        FASTAFD_CONTAINER_MOUNTS='/lustre:/lustre,/scratch:/scratch')
    gpu=['sbatch','--parsable','--propagate=NONE','--nodes=8',f'--nodelist={args.nodes}',
         '--segment=8','--consolidate-segments','--time=00:30:00',f'--dependency=afterok:{cpu_id}',
         '--kill-on-invalid-dep=yes','--job-name=qkv:model-compare',f'--output={args.root}/gpu-%j.log',
         str(source/'codex_scripts/run_afd_plan_sequence.sbatch')]
    gpu_id=subprocess.check_output(gpu,env=env,text=True).strip()
    record=dict(cpu=cpu_id,gpu=gpu_id,cpu_command=cpu,gpu_command=gpu)
    (args.root/'jobs.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


def analyze(args):
    from analyze_afd_attention_gap import graphs
    summary=[]
    for name in ['control','edge']:
        plan=args.root/name;env=json.loads((plan/'env.json').read_text())
        run=Path(env['RUN_DIR'])/'experiment'
        subprocess.run(json.loads((run/'cuda-extract-command.json').read_text()),check=True)
        metric=json.loads((Path(env['FASTAFD_CUDA_METRICS_ROOT'])/(env['FASTAFD_CASE_ID']+'.json')).read_text())
        model=next(v for v in metric['profiled_roles'].values() if v['role']=='model')
        db=plan/'model.sqlite'
        subprocess.run(['/usr/local/cuda/bin/nsys','export','--type=sqlite','--output='+str(db),model['report_path']],check=True)
        durations=[];names=set()
        for graph in graphs(db)[1:]:
            for k in graph:
                if 'publish_qkv' in k['name']:
                    durations.append((k['end']-k['start'])/1e3);names.add(k['name'])
        if len(durations)!=2820 or len(names)!=1:raise ValueError('Unexpected publication coverage')
        if ('publish_qkv_edges_kernel' in next(iter(names)))!=(name=='edge'):
            raise ValueError('Control/candidate kernel selection mismatch')
        summary.append(dict(case=name,graph_ms=metric['cuda_wall_time_ms'],
            samples=metric['sample_count'],outliers=metric['outlier_count'],
            qkv_mean_us=statistics.mean(durations),qkv_median_us=statistics.median(durations),
            qkv_count=len(durations),kernel=next(iter(names))))
    module=next((args.root/'build/tvm').glob('minisgl__afd_fmha_transport_edges*/*.so'))
    resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(module)],text=True)
    (args.root/'kernel-resources.txt').write_text(resources)
    (args.root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def trace(args):
    import sqlite3
    db=args.root/'publisher.sqlite'
    subprocess.run(['/usr/local/cuda/bin/nsys','export','--type=sqlite','--output='+str(db),
                    str(args.root/'publisher.nsys-rep')],check=True,stdout=subprocess.DEVNULL)
    con=sqlite3.connect(db)
    cases=con.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'qkv-case-%' ORDER BY start").fetchall()
    kernels=con.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
    path=args.root/f'results-g{args.group_size or 32}-p0.json'
    records=json.loads(path.read_text())
    if len(cases)!=len(records):raise ValueError('Missing NVTX case coverage')
    for index,((start,end,label),record) in enumerate(zip(cases,records)):
        selected=[(a,b) for a,b,name in kernels if start<=a and b<=end and 'publish_qkv' in name]
        if label!=f'qkv-case-{index}' or len(selected)!=args.iterations+1:raise ValueError((label,len(selected)))
        publications=selected
        samples=[(b-a)/1000 for a,b in selected[1:]]
        record.update(kernel_samples_us=samples,kernel_median_us=statistics.median(samples),kernel_mean_us=statistics.mean(samples))
        if args.experiment!='final':record['measurement']='push_kernel_only' if record['variant']>=31 else 'publication_kernel'
        waits=[(a,b) for a,b,name in kernels if start<=a and b<=end and 'wait_ready' in name]
        if args.experiment=='final':
            done=[(a,b) for a,b,name in kernels if start<=a and b<=end and 'completion_time_kernel' in name]
            if len(done)!=len(publications) or len(waits)!=len(publications):raise ValueError('Missing timed completion waits')
        else:
            if len(waits)!=2*len(publications):raise ValueError('Missing done/check ACK waits')
            done=waits[::2]
        spans=[(done[i][1]-a)/1000 for i,(a,b) in enumerate(publications)][1:]
        record['receiver_done_roundtrip_upper_bound_us']=statistics.median(spans)
        fills=[(a,b) for a,b,name in kernels if start<=a and b<=end and 'fill_input' in name]
        if len(fills)!=args.iterations+1:raise ValueError('Missing producer generation coverage')
        record['checked_graph_kernel_span_us']=(waits[-1][1]-fills[1][0])/1000
        record['graph_event_minus_kernel_span_us']=record['graph_cuda_event_us']-record['checked_graph_kernel_span_us']
        if record['variant']>=31:
            reports=[json.loads(p.read_text())[str(index)] for p in args.root.glob('receiver-r*.json')]
            if len(reports)!=(args.group_size or 32)-1:raise ValueError('Missing receiver measurements')
            per_epoch=[max(r['first_arrival_to_local_store_us'][i] for r in reports) for i in range(args.iterations)]
            record['max_receiver_arrival_to_store_median_us']=statistics.median(per_epoch)
    (args.root/'trace-results.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if 'samples' not in k} for r in records]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['submit','prepare','worker','model','model_prepare','analyze','trace'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--fabric-lib',required=True);p.add_argument('--image');p.add_argument('--python')
    p.add_argument('--nodes');p.add_argument('--iterations',type=int,default=32)
    p.add_argument('--experiment',choices=['protocol','polling','production','measurement','resources','packing','balance','final','integration'],default='protocol')
    p.add_argument('--profile',action='store_true',help='Nsight publisher kernel timings; one group size per run')
    p.add_argument('--group-size',type=int,default=0)
    p.add_argument('--group-sizes',help='Sequential group sizes in one allocation, e.g. 4,8,16,32')
    p.add_argument('--prepared',type=Path,help='Reuse an explicitly prepared, unchanged CUDA module')
    p.add_argument('--model-template',type=Path,help='Existing prepared model env.json/launch.json/case.csv plan')
    p.add_argument('--model-ratio',type=int,default=7)
    p.add_argument('--cache-rows',type=int,default=131072);p.add_argument('--layers',type=int,default=94)
    args=p.parse_args();sys.path.insert(0,str(args.source/'python'))
    if args.mode=='submit' and not all((args.image,args.python,args.nodes)):p.error('submit needs --image, --python, --nodes')
    if args.profile and args.group_sizes:p.error('--profile requires one --group-size')
    if args.mode=='worker' and args.profile and os.environ['SLURM_PROCID']=='0' and not os.environ.get('QKV_NSYS_CHILD'):
        command=['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
            '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop',
            '--output='+str(args.root/'publisher'),sys.executable,__file__]+sys.argv[1:]
        sys.exit(subprocess.call(command,env=dict(os.environ,QKV_NSYS_CHILD='1',TMPDIR=str(args.root/'tmp'))))
    globals()[args.mode](args)


if __name__=='__main__':main()
