#!/usr/bin/env python3
"""Monitor recorded AFD jobs and bounded log tails without copying bulk traces."""

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


REMOTE = '''import json,os,re,subprocess,sys
from pathlib import Path
records=json.load(sys.stdin)
queue=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],
 '-o','%i|%r'],text=True)
queue_reasons=dict(line.split('|',1) for line in queue.splitlines() if '|' in line)
for record in records:
 job=record['job']; root=Path(record.get('run_dir',record['output']))
 state=subprocess.check_output(['sacct','-n','-X','-j',job,'-o',
  'JobIDRaw,State,ExitCode,Elapsed,NodeList,Reason','-P'],text=True)
 rows=[line.split('|') for line in state.splitlines() if line.startswith(job+'|')]
 if len(rows)!=1: raise RuntimeError('Expected one scheduler record for '+job)
 row=rows[0]; logs={}
 paths=list(root.glob('*'+job+'*.out'))+list(root.glob('*'+job+'*.err'))
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
 for path in paths:
  if not path.is_file(): continue
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-8192)); tail=f.read().decode(errors='replace')
  logs[str(path)]=tail.splitlines()[-12:]
 sample=root/'experiment/sample.json'
 print(json.dumps({'job':job,'state':row[1],'exit_code':row[2],
  'elapsed':row[3],'nodes':row[4],'reason':row[5],
  'queue_reason':queue_reasons.get(job),'logs':logs,
  'ray_node_ips':sorted(node_ips),'extension_seeds':seeds,
  'sample_bytes':sample.stat().st_size if sample.is_file() else 0}),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--host", default="oci-hsg")
    parser.add_argument("--watch-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.watch_seconds < 0:
        raise ValueError("Watch interval cannot be negative")
    active = {json.loads((root / "job.json").read_text())["job"]: root
              for root in args.roots}
    while active:
        records = [json.loads((root / "job.json").read_text())
                   for root in active.values()]
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
