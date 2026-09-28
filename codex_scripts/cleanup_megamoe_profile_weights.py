#!/usr/bin/env python3
"""Review or delete exactly manifested weight caches from completed ordinary runs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = '''import json,os,re,stat,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['remote_root'])
assert root.is_absolute() and root.parent.name=='artifacts'
campaigns={x['campaign']:x for x in p['campaigns']}
assert len(campaigns)==len(p['campaigns']) and campaigns
jobs=[x['gpu_job'] for x in campaigns.values()]
raw=subprocess.check_output(['sacct','-n','-X','-j',','.join(jobs),'-o','JobIDRaw,State,ExitCode','-P'],text=True)
states={x.split('|')[0]:x.split('|')[1:3] for x in raw.splitlines() if x}
assert all(states[j]==['COMPLETED','0:0'] for j in jobs),states
for name,campaign in campaigns.items():
 assert '/' not in name and name not in ('.','..')
 assert json.loads((root/name/'gpu/job.json').read_text())['job']==campaign['gpu_job']
 plan=json.loads((root/name/'plan.json').read_text())
 assert plan['groups'] and all(g['env']['FASTAFD_IKET_CAPTURE_PHASE']=='normal' for g in plan['groups'])
files=p['files'];assert files and len({x['path'] for x in files})==len(files)==p['file_count']
assert sum(x['bytes'] for x in files)==p['total_bytes']
for item in files:
 f=Path(item['path']);relative=f.relative_to(root)
 assert relative.parts[0] in campaigns and '..' not in relative.parts
 assert f.name=='profile-rank-weights.pt' and re.fullmatch(r'rank-\\d+',f.parent.name) and f.parent.parent.name=='ranks'
 s=f.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid()
 assert (s.st_size,s.st_mtime_ns,s.st_ino)==(item['bytes'],item['mtime_ns'],item['inode']),str(f)
 assert (f.parent.parent.parent/'baseline.json').is_file()
deleted=0;freed=0
try:
 if p['delete']:
  for item in files:
   Path(item['path']).unlink();deleted+=1;freed+=item['bytes']
finally:
 print(json.dumps(dict(status='DELETE_REQUESTED' if p['delete'] else 'PASS_DRY_RUN',validated_files=len(files),validated_bytes=p['total_bytes'],deleted_files=deleted,unlinked_bytes=freed)),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--host', default='oci-hsg')
    parser.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    payload['delete'] = args.delete
    receipt = args.manifest.parent / ('deletion_receipt.json' if args.delete else 'dry_run.json')
    if args.delete and receipt.exists():
        raise ValueError('Deletion receipt already exists; review it before any further cleanup')
    result = subprocess.run(['ssh', args.host, 'python3 -B -c ' + shlex.quote(REMOTE)],
                            input=json.dumps(payload), text=True, capture_output=True)
    receipt.write_text(json.dumps(dict(exit_code=result.returncode, stdout=result.stdout,
                                      stderr=result.stderr), indent=2) + '\n')
    print(result.stdout, end='')
    print(result.stderr, end='')
    result.check_returncode()


if __name__ == '__main__':
    main()
