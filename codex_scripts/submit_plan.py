#!/usr/bin/env python3
"""Submit a prepared sbatch or AFD plan once, preserving logs and its job ID.

The caller must satisfy correctness gates and the outstanding-job limit first.
AFD plans support a launcher dry run; sbatch plans require explicit submission.
"""

import argparse
import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('sbatch', 'afd'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--host', default='oci-hsg')
    parser.add_argument('--data-host', help='Host used to copy the plan; submission still uses --host')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run and args.kind != 'afd':
        parser.error('--dry-run requires an AFD launcher plan')
    root = args.root
    if (root / 'job.json').exists():
        raise ValueError('Plan already has a job ID; use a separate recovery plan')
    env = json.loads((root / 'env.json').read_text())
    command_file = 'submit.json' if args.kind == 'sbatch' else 'launch.json'
    command = json.loads((root / command_file).read_text())
    if (not isinstance(command, list) or not command
            or any(not isinstance(arg, str) or not arg for arg in command)):
        raise ValueError('Expected a nonempty command argument list')
    if args.kind == 'sbatch':
        if command[:2] != ['sbatch', '--parsable']:
            raise ValueError('Expected an sbatch --parsable argument list')
        (root / 'submit.sh').write_text('#!/bin/bash\nset -euo pipefail\n' +
                                       shlex.join(command) + '\n')
        remote = env['OUTPUT_ROOT'].rstrip('/')
        directories = [remote]
    else:
        remote = env['FASTAFD_TASK_ROOT'].rstrip('/')
        directories = [remote, env['FASTAFD_CUDA_EXTRACT_TEMP_ROOT']]
        env['FASTAFD_DRY_RUN'] = '1' if args.dry_run else '0'
    if not remote or remote == '/':
        raise ValueError('Require a dedicated remote output directory')
    subprocess.run(['ssh', args.host, shlex.join(['mkdir', '-p', *directories])], check=True)
    copy_host = args.data_host if args.data_host is not None else args.host
    subprocess.run(['rsync', '-a', str(root) + '/', copy_host + ':' + remote + '/'],
                   check=True)
    program = ('import json,os,subprocess,sys; d=json.load(sys.stdin); '
               "os.environ.update(d['env']); subprocess.run(d['cmd'],check=True)")
    result = subprocess.run(
        ['ssh', args.host, 'python3 -c ' + shlex.quote(program)],
        input=json.dumps({'env': env, 'cmd': command}), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log = result.stdout + (result.stderr or '')
    (root / ('dry_run.log' if args.dry_run else 'submission.log')).write_text(log)
    print(log, end='', flush=True)
    result.check_returncode()
    if args.dry_run:
        return
    if args.kind == 'sbatch':
        match = re.fullmatch(r'(\d+)(?:;\S+)?', result.stdout.strip())
        if match is None:
            raise ValueError('Inspect submission.log and scheduler before retrying')
        record = {'job': match[1], 'output': remote}
    else:
        matches = re.findall(r'^submitted job=(\d+)(?:;\S+)? .*?run_dir=(\S+)$',
                             result.stdout, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError('Inspect submission.log and scheduler before retrying')
        job, run = matches[0]
        record = dict(job=job, run_dir=run, output=run, task_root=remote,
                      source=env['FASTAFD_SOURCE_REPO'])
    record['submitted_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root / 'job.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    # Analysis may resolve its input job from the remote plan directory.
    # Persist locally first so a failed transfer cannot trigger resubmission.
    subprocess.run(['rsync', '-a', str(root / 'job.json'),
                    copy_host + ':' + remote + '/job.json'], check=True)


if __name__ == '__main__':
    main()
