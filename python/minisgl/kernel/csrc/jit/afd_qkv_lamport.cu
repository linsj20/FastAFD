#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

// Blackwell NVLink LL128: 120 payload bytes followed by an 8-byte epoch.
// This uses ordered 128-byte NVLink lines, not vector-store atomicity.
namespace {
constexpr int kPacks = 1232;  // Padded whole lines for 18,432 useful bytes/row.
struct QkvMailboxes { uint64_t* ptr[32]; int64_t stride[32]; };
__device__ __forceinline__ int64_t qkv_clock() {
  int64_t t;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));return t;
}
__device__ __forceinline__ ulong2 qkv_load(const uint64_t* p) {
  ulong2 v;asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
      : "=l"(v.x),"=l"(v.y) : "l"(p) : "memory");return v;
}
__device__ __forceinline__ uint64_t qkv_word(const uint64_t* q,const uint64_t* k,
    const uint64_t* v,int row,int word,int64_t qs,int64_t ks,int64_t vs) {
  return word<2048 ? q[row*qs+word] : word<2176 ? k[row*ks+word-2048]
       : word<2304 ? v[row*vs+word-2176] : 0;
}
__global__ void publish_qkv_lamport_kernel(const uint64_t* q,const uint64_t* k,
    const uint64_t* v,QkvMailboxes boxes,int rows,int slot,int64_t epoch,
    int64_t qs,int64_t ks,int64_t vs) {
  const unsigned peer=blockIdx.x/4,lane=(blockIdx.x%4)*256+threadIdx.x;
  auto* out=boxes.ptr[peer]+slot*boxes.stride[peer];
  for(unsigned i=lane;i<rows*kPacks;i+=1024) {
    const unsigned p=i%kPacks,word=2*p-p/8,row=peer*rows+i/kPacks;
    const uint64_t a=qkv_word(q,k,v,row,word,qs,ks,vs);
    const uint64_t b=p%8==7 ? uint64_t(epoch) : qkv_word(q,k,v,row,word+1,qs,ks,vs);
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};"
        :: "l"(out+i*2),"l"(a),"l"(b) : "memory");
  }
}
template<bool Loc64>
__global__ void receive_qkv_lamport_kernel(const uint64_t* inbox,uint64_t* q,uint64_t* kv,
    const void* locations,int rows,int layer,int slot,int64_t epoch,int64_t inbox_stride,
    int64_t q_stride,int64_t cache_rows,int layers,int64_t timeout_ns) {
  const unsigned lane=blockIdx.x*256+threadIdx.x;
  const auto* in=inbox+slot*inbox_stride;
  const int64_t deadline=qkv_clock()+timeout_ns;
  for(unsigned i=lane;i<((rows*kPacks+31)/32)*32;i+=1024) {
    const bool valid=i<rows*kPacks;
    const unsigned r=i/kPacks,p=i%kPacks,word=2*p-p/8;
    ulong2 value{};bool pending;unsigned spins=0;
    do {
      if(valid)value=qkv_load(in+i*2);
      pending=valid && p%8==7 && value.y!=uint64_t(epoch);
      pending=__any_sync(0xffffffff,pending);
      if((++spins&255)==0 && qkv_clock()>deadline)asm volatile("trap;");
    } while(pending);
    // A data lane can have loaded before the flag lane. Reload after the vote.
    if(valid) {
      value=qkv_load(in+i*2);
      const int64_t loc=Loc64 ? static_cast<const int64_t*>(locations)[r]
                             : static_cast<const int32_t*>(locations)[r];
      const int64_t base=(int64_t(layer)*cache_rows+loc)*128;
#pragma unroll
      for(int j=0;j<2;++j) {
        const unsigned w=word+j;const uint64_t x=j ? value.y : value.x;
        if(j && p%8==7)continue;
        if(w<2048)q[slot*q_stride+r*2048+w]=x;
        else if(w<2304)kv[base+(w-2048)%128+(w>=2176 ? int64_t(layers)*cache_rows*128 : 0)]=x;
      }
    }
  }
  // Kernel completion makes Q/K/V available to following local attention.
  // No receiver-to-publisher acknowledgement is required here.
}
}

struct AfdQkvLamport {
  using T=tvm::ffi::TensorView;
  static void publish(T q,T k,T v,T host_boxes,int64_t slot,int64_t epoch) {
    using namespace host;
    RuntimeCheck(host_boxes.device().device_type==kDLCPU && host_boxes.ndim()==2 && host_boxes.size(1)==2);
    const int peers=host_boxes.size(0);
    RuntimeCheck(peers>0 && peers<=32 && q.ndim()==2 && q.size(0)%peers==0);
    const int rows=q.size(0)/peers;
    RuntimeCheck(rows>0 && rows<=4 && slot>=0 && slot<2 && epoch>0);
    RuntimeCheck(q.size(1)==8192 && k.size(1)==512 && v.size(1)==512);
    RuntimeCheck(q.stride(0)%4==0 && k.stride(0)%4==0 && v.stride(0)%4==0);
    QkvMailboxes boxes{};const auto* src=static_cast<const int64_t*>(host_boxes.data_ptr());
    for(int p=0;p<peers;++p) {
      RuntimeCheck(src[p*2]%128==0 && src[p*2+1]>=rows*kPacks*2 && src[p*2+1]%16==0);
      boxes.ptr[p]=reinterpret_cast<uint64_t*>(src[p*2]);boxes.stride[p]=src[p*2+1];
    }
    LaunchKernel(4*peers,256,q.device())(publish_qkv_lamport_kernel,
        static_cast<const uint64_t*>(q.data_ptr()),static_cast<const uint64_t*>(k.data_ptr()),
        static_cast<const uint64_t*>(v.data_ptr()),boxes,rows,int(slot),epoch,
        q.stride(0)/4,k.stride(0)/4,v.stride(0)/4);
  }
  static void receive(T inbox,T q,T kv,T loc,int64_t layer,int64_t slot,int64_t epoch,int64_t timeout_ms) {
    using namespace host;
    const int rows=loc.numel(),layers=kv.size(1);
    RuntimeCheck(rows>0 && rows<=4 && loc.ndim()==1 && loc.stride(0)==1);
    RuntimeCheck(slot>=0 && slot<2 && layer>=0 && layer<layers && epoch>0 && timeout_ms>0);
    RuntimeCheck(inbox.size(0)==2 && inbox.stride(0)>=rows*kPacks*2 && inbox.stride(0)%16==0);
    RuntimeCheck(q.size(0)==2 && q.stride(0)>=rows*8192 && q.stride(0)%4==0);
    RuntimeCheck(loc.dtype().code==kDLInt && (loc.dtype().bits==32 || loc.dtype().bits==64));
    const int64_t cache_rows=kv.numel()/(2*layers*512);
    LaunchKernel(4,256,q.device())(loc.dtype().bits==64 ? receive_qkv_lamport_kernel<true> : receive_qkv_lamport_kernel<false>,
        static_cast<const uint64_t*>(inbox.data_ptr()),static_cast<uint64_t*>(q.data_ptr()),
        static_cast<uint64_t*>(kv.data_ptr()),loc.data_ptr(),rows,int(layer),int(slot),epoch,
        inbox.stride(0),q.stride(0)/4,cache_rows,layers,timeout_ms*1000000);
  }
};
