#!/usr/bin/env python3
"""Prepare or analyze a Qwen3 rack-mapped run using an existing AGA environment.

The root supplies mapping.json; template supplies a previous env.json/case.csv.
CPU preparation builds kernels and seeds only the actual mapped worker scopes.
Run the generated plans with run_afd_plan_sequence.sbatch.
"""
import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(a):
    import torch
    from map_afd_racks import map_racks
    from minisgl.afd_protocol import AfdTopology
    from minisgl.kernel import deepgemm
    from minisgl.kernel.afd_qkv_lamport import module
    from minisgl.kernel.afd_fmha_transport import ensure_afd_fmha_transport_built
    from seed_native_cache import seed_host, seed_tvm, seed_device
    t=a.root; b=t/'build'; b.mkdir(exist_ok=True)
    (b/'tmp').mkdir(exist_ok=True);(b/'jit/cache').mkdir(parents=True,exist_ok=True)
    mapping=json.loads((t/'mapping.json').read_text())
    assert mapping==map_racks(mapping['inventory'],[x['rack'] for x in mapping['ffn']])
    attention=mapping['attention_gpus'];ep=mapping['ffn_gpus'];fanin=mapping['fanin_per_ffn']
    topology=AfdTopology(attn_dp_size=attention,mlp_dp_size=ep,attn_tp_size=1,mlp_tp_size=1,ep_size=ep)
    assert all(topology.mlp_dp_for_attn_dp(x['dp_rank'])==x['ffn_dp_rank'] for x in mapping['attention'])
    base=json.loads((a.template/'env.json').read_text());s=Path(base['FASTAFD_SOURCE_REPO'])
    oldcase=list(csv.DictReader((a.template/'case.csv').open()))[0]
    cache=t/'cache';plans=[]
    for batch in a.batches:
        p=t/f'b{batch}';p.mkdir(exist_ok=True);case_id=f'i131072-fep{ep}-r{fanin}-atp1-b{batch}-fp4'
        env=dict(base,**mapping['environment'])
        for key in ('TRITON_CACHE_DIR','TVM_FFI_CACHE_DIR','MINISGL_DEEPGEMM_BUILD_DIR','MINISGL_DEEPEP_BUILD_DIR',
                    'DG_JIT_CACHE_DIR','EP_JIT_CACHE_DIR','FLASHINFER_WORKSPACE_BASE'):
            env[key]=str(cache/Path(base[key]).name)
        env.update(FASTAFD_TASK_ROOT=str(p),FASTAFD_PLAN_NODES=str(len(mapping['inventory'])),
            FASTAFD_MAX_TOTAL_TRAYS=str(len(mapping['inventory'])),FASTAFD_CASE_ID=case_id,
            FASTAFD_RESULTS_ROOT=str(p/'results'),RUN_DIR=str(p/'results'/case_id),
            FASTAFD_CUDA_METRIC_PLAN=str(p/'case.csv'),FASTAFD_CUDA_METRICS_ROOT=str(p/'report/metrics'),
            FASTAFD_CUDA_EXTRACT_TEMP_ROOT=str(p/'extract'),FASTAFD_PROMPT_CACHE_ROOT=str(t/'prompt-cache'),
            MINISGL_AFD_QKV_LAMPORT='1',FASTAFD_MINISGL_PORT=str(8040+8*(batch-min(a.batches))))
        reset=fanin*((batch+1)//2)
        capacity=max(1024,reset+((reset+31)//32)*32*(128//ep))
        env['MINISGL_MEGAMOE_ALL_PUSH_CAPACITY']=str(capacity)
        env.pop('FASTAFD_CASE_PORT_SLOT',None);env.pop('FASTAFD_CASE_ORDINAL',None)
        write(p/'env.json',env)
        write(p/'launch.json',['bash',env['JOB_SCRIPT'],'qwen3','131072',str(batch),f'{fanin}:1','1',str(ep)])
        case=dict(oldcase,case_id=case_id,normalized_af_ratio=str(fanin),ffn_ep=str(ep),
            batch_per_attention_dp_lane=str(batch),required_target_batch_per_attention_dp_lane=str(batch),
            allocated_trays=str(len(mapping['inventory'])),allocated_gpus=str(4*len(mapping['inventory'])),
            active_gpus=str(attention+ep),run_afd_argv=f'qwen3 131072 {batch} {fanin}:1 1 {ep}')
        with (p/'case.csv').open('w') as out:
            w=csv.DictWriter(out,fieldnames=list(case));w.writeheader();w.writerow(case)
        plans.append(str(p))
    write(t/'plans.json',plans)
    env=json.loads((Path(plans[0])/'env.json').read_text())
    os.environ.update({k:v for k,v in env.items() if k.startswith(('MINISGL_MEGAMOE_','DG_MEGAMOE_'))})
    os.environ.update(TMPDIR=str(b/'tmp'),MINISGL_DEEPGEMM_BUILD_DIR=str(a.promotion/'build/native'),
        DG_JIT_CACHE_DIR=str(b/'jit'),DG_JIT_USE_NVRTC='0',CUDA_HOME='/usr/local/cuda',CUDA_PATH='/usr/local/cuda',
        CUDA_NVCC_EXECUTABLE='/usr/local/cuda/bin/nvcc',LIBRARY_PATH='/usr/local/cuda/lib64/stubs',MAX_JOBS='8',
        TVM_FFI_CACHE_DIR=str(b/'tvm'),TVM_FFI_CUDA_ARCH_LIST='10.0',TORCH_CUDA_ARCH_LIST='10.0')
    for old in (a.promotion/'build/jit/cache').glob('kernel.*'):
        target=b/'jit/cache'/old.name
        if not target.exists():target.symlink_to(old,target_is_directory=True)
    records=[];seen=set()
    for reset in sorted({fanin*((batch+1)//2) for batch in a.batches}):
        capacity=max(1024,reset+((reset+31)//32)*32*(128//ep))
        bucket=((capacity+383)//384)*384
        for rows in range(fanin,reset+1,fanin):
            code,config,block,wave=deepgemm.generate_megamoe_kernel(ep,128,bucket,rows,8,4096,1536,148,True,True,True)
            if rows+((rows+block-1)//block)*block*wave>bucket:raise ValueError(('all-push capacity',rows,block,wave))
            if 'true, 0\n' not in code:raise ValueError('Reset specialization not found')
            code=code.replace('true, 0\n',f'true, {reset}\n')
            if code in seen:continue
            seen.add(code)
            target=deepgemm.precompile_kernel('sm100_fp8_fp4_mega_moe',code,'100f')
            records.append(dict(rows=rows,reset=reset,bucket=bucket,target=target));print('KERNEL_READY',rows,reset,bucket,flush=True)
    module();ensure_afd_fmha_transport_built(128)
    write(b/'kernels.json',records)
    ips=[x['address'] for x in mapping['inventory']]
    for kind,donor,key in [
        ('deepgemm',next((a.promotion/'build/native/lib').glob('_minisgl_deepgemm.*.so')),'MINISGL_DEEPGEMM_BUILD_DIR'),
        ('deepep',Path(a.fabric_lib),'MINISGL_DEEPEP_BUILD_DIR')]:
        seed_host(SimpleNamespace(kind=kind,donor=donor,cache_root=[Path(env[key])],node_ip=ips,copy=False,output=b/f'seed-{kind}.json'))
    scopes={'ray-node-'+ip.replace('.','_') for ip in ips}
    for offset,slots in [(1,mapping['attention']),(attention+1,mapping['ffn'])]:
        scopes.update('ray-node-'+x['address'].replace('.','_')+f"/rank{offset+x['dp_rank']}" for x in slots)
    # Preserve the ordinary donors, then add this run's newly compiled specializations.
    for i,scope in enumerate(sorted(scopes)):
        with contextlib.redirect_stdout(io.StringIO()):
            for donor in (a.promotion/'build/tvm',b/'tvm'):
                seed_tvm(SimpleNamespace(donor=donor,cache_root=[Path(env['TVM_FFI_CACHE_DIR'])/scope],node_ip=None,rank_count=0,output=b/f'tvm-{i}-{donor.parent.name}.json'))
            seed_device(SimpleNamespace(donor=b/'jit',cache_root=[Path(env['DG_JIT_CACHE_DIR'])/scope],node_ip=None,rank_count=0,output=b/f'device-{i}.json'))
    text=(s/'scripts/experiments/afd/oci_hsg/run_afd.sh').read_text()
    start=text.index('from pathlib import Path\nimport fcntl',text.index('PROMPT_CACHE_ROOT='))
    builder=text[start:text.index('\nPY\nexport FASTAFD_SOURCE_REPO',start)]
    for batch,path in zip(a.batches,plans):
        p=Path(path);en=json.loads((p/'env.json').read_text())
        dry={**os.environ,**en,'FASTAFD_DRY_RUN':'1'};dry.pop('SLURM_JOB_ID',None)
        with (p/'dry-run.log').open('w') as out:
            subprocess.run(json.loads((p/'launch.json').read_text()),env=dry,stdout=out,stderr=subprocess.STDOUT,check=True)
        model=Path(en['FASTAFD_REPRO_ROOT'])/'model-profiles/qwen3-235b-a22b-fp8-39eb2b06-yarn-f4p001953125-original32768-max131136'
        out=p/'prepared-prompts';out.mkdir(exist_ok=True)
        sys.argv=['prompt-cache',en['FASTAFD_PROMPT_BASE_FILE'],str(out/'prompts.txt'),str(out/'transform.json'),
            en['FASTAFD_PROMPT_CACHE_ROOT'],str(attention*batch),'131072','131072','qwen3',str(model),
            '39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35','uniform','8192',str(attention),','.join(['131072']*batch),'1']
        exec(compile(builder,'maintained_prompt_builder','exec'),{'__name__':'__main__'})
        print('PLAN_READY',batch,flush=True)
    if torch.cuda.is_initialized():raise RuntimeError('CPU preparation initialized CUDA')
    validate(a)
    write(t/'cpu-ready.json',dict(attention=attention,ffn=ep,racks=ep,allocated_gpus=4*len(ips),kernels=len(records),scopes=len(scopes),batches=a.batches))


def validate(a):
    """Exercise the actual driver placement prefix on CPU, with explicit nodes."""
    import torch
    from unittest.mock import patch
    import minisgl.kernel.fabric_memory as fabric
    t=a.root;mapping=json.loads((t/'mapping.json').read_text())
    plans=json.loads((t/'plans.json').read_text());env=json.loads((Path(plans[0])/'env.json').read_text())
    source=Path(env['FASTAFD_SOURCE_REPO'])
    text=(source/'scripts/experiments/afd/oci_hsg/run_afd_qwen3_parallel.sh').read_text()
    prefix=text[:text.index('export NUM_PROMPTS=')]
    script=t/'build/validate-layout.sh';script.write_text(prefix)
    out=t/'build/layout';out.mkdir(exist_ok=True)
    runtime={**os.environ,**env,'AFD_TOTAL_NODES':str(len(mapping['inventory'])),
             'AFD_GPUS_PER_NODE':'4','AFD_NODE_LIST':','.join(x['address'] for x in mapping['inventory']),
             'RUN_DIR':str(out)}
    subprocess.run(['bash',str(script)],env=runtime,check=True)
    placement=json.loads((out/'afd-placement.json').read_text())
    for key,slots in [('attention_dp_rank_nodes',mapping['attention']),('mlp_dp_rank_nodes',mapping['ffn'])]:
        if placement[key]!=[[x['address']] for x in slots]:raise RuntimeError('Driver rank map mismatch')
    # Run the real allocator's offset logic without allocating GPU/HBM storage.
    allocator=SimpleNamespace(FabricAllocation=SimpleNamespace(
        allocate=lambda size:SimpleNamespace(tensor=lambda:None)))
    specs=(((2,94,14401,64,4,128),torch.bfloat16),((2,516,64,128),torch.bfloat16),
           ((2,1),torch.int64),((2,516,2464),torch.int64))
    with patch.object(fabric,'_load_extension',lambda:allocator),patch.object(fabric,'_typed_view',lambda *args,**kw:None):
        legacy=fabric.allocate_fabric_tensors(specs)
        aligned=fabric.allocate_fabric_tensors(specs,alignment=128)
    if legacy[-1].byte_offset%128!=16 or any(x.byte_offset%128 for x in aligned):
        raise RuntimeError('Lamport model arena alignment check failed')
    if torch.cuda.is_initialized():raise RuntimeError('Layout check initialized CUDA')
    print('LL128_ARENA_ALIGNMENT_PASS',aligned[-1].byte_offset,flush=True)
    print('DRIVER_LAYOUT_PASS',mapping['attention_gpus'],mapping['ffn_gpus'],flush=True)


def analyze(a):
    import sqlite3
    from collections import defaultdict
    t=a.root;mapping=json.loads((t/'mapping.json').read_text());attention=mapping['attention_gpus']
    results=[]
    for path in json.loads((t/'plans.json').read_text()):
        if int(Path(path).name.removeprefix('b')) not in a.batches:continue
        p=Path(path);env=json.loads((p/'env.json').read_text());run=Path(env['RUN_DIR'])/'experiment'
        result=json.loads((run/'afd-result.json').read_text())
        for key,slots in [('attention_dp_rank_nodes',mapping['attention']),('mlp_dp_rank_nodes',mapping['ffn'])]:
            if result['placement'][key]!=[[x['address']] for x in slots]:raise RuntimeError('Observed placement differs')
        entry=dict(batch=result['batch_per_attention_dp_lane'],placement='passed',traces=[])
        paired={}
        for rank,kind in [(1,'attention')]+[(attention+1+i,'ffn') for i in range(mapping['ffn_gpus'])]:
            reports=list((run/'ray_logs').glob(f'minisgl_rank{rank}_*.nsys-rep'))
            if len(reports)!=1:raise RuntimeError(('Missing trace',rank,reports))
            db=p/f'{kind}-{rank}.sqlite'
            if not db.exists():
                subprocess.run(['/usr/local/cuda/bin/nsys','export','--type=sqlite','--output='+str(db),str(reports[0])],check=True,stdout=subprocess.DEVNULL)
            with sqlite3.connect(db) as con:
                kernels=con.execute('SELECT k.start,k.end,k.correlationId,k.graphId,s.value,k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
                origin=con.execute('SELECT startTime FROM ANALYSIS_DETAILS').fetchone()[0]
                utc=con.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()[0]
            groups=defaultdict(list)
            for k in kernels:
                if k[3] is not None:groups[k[2]].append(k)
            graphs=sorted(groups.values(),key=lambda g:g[0][0])[1:]
            needle='publish_qkv_lamport_kernel' if kind=='ffn' else 'receive_qkv_lamport_kernel'
            samples=[(k[1]-k[0])/1000 for graph in graphs for k in graph if needle in k[4]]
            if len(graphs)!=15 or len(samples)!=15*94*2:raise RuntimeError(('Lamport trace coverage',rank,len(graphs),len(samples)))
            per_step=[statistics.mean((k[1]-k[0])/1000 for k in g if needle in k[4]) for g in graphs]
            entry['traces'].append(dict(role=kind,rank=rank,path=str(reports[0]),kernel=needle,count=len(samples),median_us=statistics.median(samples),mean_us=statistics.mean(samples),p95_us=sorted(samples)[int(.95*(len(samples)-1))],per_step_mean_us=per_step))
            if rank in (1,attention+1):paired[kind]=(origin,utc,[[k for k in g if needle in k[4]] for g in graphs])
        # Compare only the mapped sender/receiver on the same host. Independent
        # captures have different t=0; both monotonic and UTC offsets must agree.
        if mapping['attention'][0]['address']!=mapping['ffn'][0]['address']:
            raise RuntimeError('Readiness timing requires a same-host trace pair')
        ao,au,ag=paired['attention'];fo,fu,fg=paired['ffn']
        clock_difference=(ao-fo)-(au-fu)
        if abs(clock_difference)>100:raise RuntimeError(('Trace clock offsets differ by >0.1us',clock_difference))
        readiness=[];prearmed=[];per_step=[]
        for receivers,senders in zip(ag,fg):
            # MB0/MB1 use two streams, each with 94 ordered layer operations.
            lanes=[]
            for kernels in (receivers,senders):
                streams=defaultdict(list)
                for k in kernels:streams[k[5]].append(k)
                ordered=sorted(streams.values(),key=lambda ks:ks[0][0])
                if sorted(map(len,ordered))!=[94,94]:raise RuntimeError('Unexpected microbatch stream coverage')
                lanes.append(ordered)
            step=[]
            for recv,send in zip(*lanes):
                for r,s in zip(recv,send):
                    delta=(r[1]+ao-s[0]-fo)/1000
                    if delta<0:raise RuntimeError('Noncausal sender/receiver pairing')
                    step.append(delta);prearmed.append(r[0]+ao<=s[0]+fo)
            readiness.extend(step);per_step.append(statistics.median(step))
        entry['same_host_readiness']=dict(sender_rank=attention+1,receiver_rank=1,count=len(readiness),
            clock_origin_disagreement_ns=clock_difference,
            method='same-host trace origins; paired microbatch streams and layer order; sender start to receiver end, including unpack',
            median_us=statistics.median(readiness),p95_us=sorted(readiness)[int(.95*(len(readiness)-1))],
            max_us=max(readiness),receiver_prearmed_fraction=statistics.mean(prearmed),per_step_median_us=per_step)
        metric_log=p/'cuda-metric-extraction.log'
        with metric_log.open('w') as out:
            metric=subprocess.run(json.loads((run/'cuda-extract-command.json').read_text()),check=False,stdout=out,stderr=subprocess.STDOUT)
        mp=Path(env['FASTAFD_CUDA_METRICS_ROOT'])/(env['FASTAFD_CASE_ID']+'.json')
        entry.update(metric_status='passed' if metric.returncode==0 else 'rejected',metric_path=str(mp) if metric.returncode==0 else None,metric_log=str(metric_log))
        if metric.returncode==0:entry['metric']=json.loads(mp.read_text())
        results.append(entry)
        write(t/('SUMMARY.json' if len(a.batches)==4 else 'SUMMARY-b'+'-'.join(map(str,a.batches))+'.json'),results)
        print(json.dumps({k:v for k,v in entry.items() if k not in ('traces','metric')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','validate','analyze']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--template',type=Path);p.add_argument('--promotion',type=Path);p.add_argument('--fabric-lib')
    p.add_argument('--batches',type=int,nargs='+',default=[4,5,6,7]);a=p.parse_args()
    globals()[a.mode](a)
