#!/usr/bin/env python3
"""Monitor recorded AFD jobs and bounded log tails without copying bulk traces."""

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


REMOTE = '''import datetime,json,os,re,subprocess,sys
from pathlib import Path
records=json.load(sys.stdin)
queue=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],
 '-o','%i|%r'],text=True)
queue_reasons=dict(line.split('|',1) for line in queue.splitlines() if '|' in line)
state=subprocess.check_output(['sacct','-n','-X','-j',','.join(r['job'] for r in records),'-o',
 'JobIDRaw,State,ExitCode,Elapsed,NodeList,Reason','-P'],text=True)
state_rows=[line.split('|') for line in state.splitlines()]
for record in records:
 job=record['job']; root=Path(record.get('run_dir',record['output']))
 rows=[row for row in state_rows if row[0]==job]
 if len(rows)!=1: raise RuntimeError('Expected one scheduler record for '+job)
 row=rows[0]; logs={}
 paths=list(root.glob('*'+job+'*.out'))+list(root.glob('*'+job+'*.err'))
 paths += [Path(path) for path in record.get('log_paths',[])]
 paths += [root/'experiment/afd.log',root/'experiment/driver.stderr']
 node_ips=set()
 for run in sorted(root.glob('*/results/*')):
  if not run.is_dir(): continue
  paths += [run/'experiment/afd.log',run/'experiment/driver.stderr']
  paths += sorted(run.glob('plan-rank-*.log'))
  for log in run.glob('ray-start-*.log'):
   node_ips.update(re.findall(r'Local node IP:\\s*([0-9.]+)',log.read_text(errors='replace')))
 seeds=[]
 if row[1] == 'RUNNING' and node_ips:
  for seed in record.get('extension_seeds',[]):
   command=['python3',record['extension_seed_script'],seed['donor'],
            '--cache-root',seed['cache_root'],'--output',seed['output']]
   for ip in sorted(node_ips): command += ['--node-ip',ip]
   result=subprocess.run(command,text=True,capture_output=True)
   seeds.append({'cache_root':seed['cache_root'],'exit_code':result.returncode,
                 'stdout':result.stdout,'stderr':result.stderr})
 log_bytes={}; log_counts={}
 for path in dict.fromkeys(paths):
  if not path.is_file(): continue
  log_bytes[str(path)]=path.stat().st_size
  if record.get('log_count_pattern'):
   pattern=re.compile(record['log_count_pattern'])
   with path.open(errors='replace') as f:
    log_counts[str(path)]=sum(bool(pattern.search(line.rstrip('\\n'))) for line in f)
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-8192)); tail=f.read().decode(errors='replace')
  logs[str(path)]=tail.splitlines()[-12:]
 sample=root/'experiment/sample.json'
 progress=[]
 if row[1] == 'RUNNING' and record.get('sweep_plan'):
  plan=json.loads(Path(record['sweep_plan']).read_text())
  for group in plan['groups']:
   sweep=Path(group['env']['MEGAMOE_SWEEP_ROOT'])
   cases=group['env']['MEGAMOE_SWEEP_CASES'].split(',')
   done=[]; started=[]
   ordinary_only=group['env'].get('FASTAFD_IKET_CAPTURE_PHASE') == 'normal'
   for case in cases:
    runs=list((sweep/case).glob('*/source-snapshot.txt'))
    if runs: started.append(case)
    if any((p.parent/'baseline.json').is_file() and (ordinary_only or ((p.parent/'profiled.json').is_file() and len(list((p.parent/'ranks/rank-0/iket').glob('*.trace.json'))) == 1)) for p in runs):
     done.append(case)
   progress.append(dict(model=group['model'],completed=len(done),expected=len(cases),last_started=started[-1:]))
 graph_progress=[]
 if row[1] == 'RUNNING':
  for filename in record.get('graph_plans',[]):
   plan=json.loads(Path(filename).read_text())
   completed=sum((Path(run['path'])/'result.json').is_file() for run in plan['runs'])
   graph_progress.append(dict(plan=filename,completed=completed,expected=len(plan['runs'])))
 print(json.dumps({'job':job,'state':row[1],'exit_code':row[2],
  'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'elapsed':row[3],'nodes':row[4],'reason':row[5],
  'queue_reason':queue_reasons.get(job),'logs':logs,'log_bytes':log_bytes,'log_counts':log_counts,
  'ray_node_ips':sorted(node_ips),'extension_seeds':seeds,'sweep_progress':progress,'graph_progress':graph_progress,
  'sample_bytes':sample.stat().st_size if sample.is_file() else 0}),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--host", default="oci-hsg")
    parser.add_argument("--watch-seconds", type=int, default=300)
    parser.add_argument("--log-count-pattern", help="Count matching lines in each log without copying it")
    args = parser.parse_args()
    if args.watch_seconds < 0:
        raise ValueError("Watch interval cannot be negative")
    active = {json.loads((root / "job.json").read_text())["job"]: root
              for root in args.roots}
    while active:
        records = [json.loads((root / "job.json").read_text())
                   for root in active.values()]
        for record, root in zip(records, active.values()):
            submission = root / "submit.json"
            if submission.is_file():
                command = json.loads(submission.read_text())
                record['log_paths'] = [arg.split('=', 1)[1].replace('%j', record['job'])
                                       for arg in command if arg.startswith(('--output=', '--error='))]
            record['log_count_pattern'] = args.log_count_pattern
        output = subprocess.check_output(
            ["ssh", args.host, "python3 -c " + shlex.quote(REMOTE)],
            input=json.dumps(records), text=True)
        for line in output.splitlines():
            result = json.loads(line)
            root = active[result["job"]]
            (root / "monitor_snapshot.json").write_text(
                json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "logs"}),
                  flush=True)
            if result["state"].split()[0] in {
                "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}:
                del active[result["job"]]
        if not args.watch_seconds:
            break
        if active:
            time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
