#!/usr/bin/env bash
# Run prepared sweep arms on one allocation, using the existing per-EP worker.
# Each arm has env.json and its own manifest/output/cache paths. No submissions.
# Required: MEGAMOE_SWEEP_ABBA_ROOT and MEGAMOE_SWEEP_ABBA_ORDER.
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${MEGAMOE_SWEEP_ABBA_ROOT:?}" "${MEGAMOE_SWEEP_ABBA_ORDER:?}"
for arm in $MEGAMOE_SWEEP_ABBA_ORDER; do
    python3 - "$MEGAMOE_SWEEP_ABBA_ROOT/$arm/env.json" <<'PY'
import json
import os
import subprocess
import sys

with open(sys.argv[1]) as stream:
    settings = json.load(stream)
os.environ.update(settings)
subprocess.run(["bash", settings["MEGAMOE_SWEEP_SCRIPT"]], check=True)
PY
done
