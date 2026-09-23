#!/usr/bin/env bash
# Run prepared AFD env.json/launch.json plans sequentially in one allocation.
# Each plan must fit the whole allocation and supply a unique RUN_DIR.
# Required: FASTAFD_PLAN_SEQUENCE (JSON list of plan directories), FASTAFD_IMAGE,
# FASTAFD_SEQUENCE_ROOT, FASTAFD_SEQUENCE_SCRIPT (this file's durable path).
# Optional: FASTAFD_SEQUENCE_FABRIC_SCOPE=domain for one AGA NVL576 domain,
# FASTAFD_SEQUENCE_RACK_MAP (map_afd_racks.py output), FASTAFD_CONTAINER_MOUNTS.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --no-requeue

set -euo pipefail
: "${FASTAFD_PLAN_SEQUENCE:?}" "${FASTAFD_IMAGE:?}" \
  "${FASTAFD_SEQUENCE_ROOT:?}" "${FASTAFD_SEQUENCE_SCRIPT:?}"
if [[ ${FASTAFD_SEQUENCE_INSIDE:-0} != 1 ]]; then
    mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    export HEAD_HOST=${hosts[0]}
    case ${FASTAFD_SEQUENCE_FABRIC_SCOPE:-rack} in
        rack) suffix='-T[0-9]+$' ;;
        domain)
            for host in "${hosts[@]}"; do
                [[ $host =~ ^nvl576d[0-9]+-R[0-9]+-T[0-9]+$ ]] || exit 2
            done
            suffix='-R[0-9]+-T[0-9]+$' ;;
        *) echo 'Unknown FASTAFD_SEQUENCE_FABRIC_SCOPE' >&2; exit 2 ;;
    esac
    mapfile -t fabrics < <(printf '%s\n' "${hosts[@]}" | sed -E "s/$suffix//" | sort -u)
    (( ${#fabrics[@]} == 1 )) || { echo 'AFD sequence spans the requested fabric scope' >&2; exit 2; }
    FASTAFD_SEQUENCE_ALLOCATED_HOSTS=$(IFS=,; echo "${hosts[*]}")
    export FASTAFD_SEQUENCE_ALLOCATED_HOSTS
    exec srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
        --ntasks-per-node=1 --gres=gpu:4 --mem=0 --kill-on-bad-exit=1 \
        --container-image="$FASTAFD_IMAGE" --container-mount-home \
        --container-mounts="${FASTAFD_CONTAINER_MOUNTS:-/lustre:/lustre}" --no-container-remap-root \
        env FASTAFD_SEQUENCE_INSIDE=1 bash "$FASTAFD_SEQUENCE_SCRIPT"
fi

python3 - "$FASTAFD_PLAN_SEQUENCE" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys
import time

plans = json.loads(Path(sys.argv[1]).read_text())
if not plans or len(plans) != len(set(plans)):
    raise ValueError('Sequence requires distinct plan directories')
settings = {directory: json.loads((Path(directory) / 'env.json').read_text())
            for directory in plans}
required = {'MODEL_KEY', 'PROMPT_MODE', 'RUN_DIR', 'JOB_SCRIPT', 'FASTAFD_SOURCE_REPO'}
for directory, values in settings.items():
    if missing := required - values.keys():
        raise ValueError(f'AFD plan {directory} missing explicit environment: {sorted(missing)}')
if len({values['RUN_DIR'] for values in settings.values()}) != len(plans):
    raise ValueError('AFD plans must use distinct RUN_DIR values')
if mapping_path := os.environ.get('FASTAFD_SEQUENCE_RACK_MAP'):
    mapping = json.loads(Path(mapping_path).read_text())
    expected = {node['host'] for node in mapping['inventory']}
    actual = set(os.environ['FASTAFD_SEQUENCE_ALLOCATED_HOSTS'].split(','))
    if actual != expected:
        raise ValueError('Allocation differs from the prepared rack mapping')
    for directory, values in settings.items():
        if any(values.get(key) != value for key, value in mapping['environment'].items()):
            raise ValueError(f'AFD plan {directory} differs from the prepared rank mapping')
rank, nodes = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_JOB_NUM_NODES'])
state = Path(os.environ['FASTAFD_SEQUENCE_ROOT']) / 'state' / os.environ['SLURM_JOB_ID']
state.mkdir(parents=True, exist_ok=True)
for index, directory in enumerate(plans, 1):
    root = Path(directory)
    env = dict(os.environ)
    env.update(settings[directory])
    command = json.loads((root / 'launch.json').read_text())
    env.update(FASTAFD_IN_CONTAINER='1', FASTAFD_DRY_RUN='0',
               FASTAFD_CASE_ORDINAL=str(index), FASTAFD_CASE_PORT_SLOT=str(index))
    run = Path(env['RUN_DIR'])
    run.mkdir(parents=True, exist_ok=True)
    print(f'AFD_PLAN_BEGIN index={index} rank={rank} plan={root}', flush=True)
    with (run / f'plan-rank-{rank}.log').open('x') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f'AFD plan {root}, rank {rank} failed: {result.returncode}; see {log.name}')
    (state / f'{index}.rank-{rank}.done').touch(exist_ok=False)
    deadline = time.monotonic() + 180
    while not all((state / f'{index}.rank-{r}.done').exists() for r in range(nodes)):
        if time.monotonic() > deadline:
            raise TimeoutError(f'AFD plan {root}: ranks did not finish together')
        time.sleep(1)
    print(f'AFD_PLAN_COMPLETE index={index} rank={rank} plan={root}', flush=True)
print(f'AFD_SEQUENCE_SUCCESS rank={rank} plans={len(plans)}', flush=True)
PY
