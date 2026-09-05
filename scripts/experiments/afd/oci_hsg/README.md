# OCI-HSG AFD experiment controls

These are the reusable controls promoted from the corrected Qwen3 sweeps and
the optimized-attention single-case runs. Generated plans, source manifests,
job-specific submit wrappers, node exclusions, logs, traces, and results stay
in the caller's experiment workspace rather than this source directory.

## Single case

`run_afd.sh` validates topology, KV capacity, source provenance, capture
policy, and result extraction before submitting or running one Qwen3 or
MiniMax case. Qwen3 uses `run_afd_qwen3_parallel.sh` as the in-allocation
preset and supports the opt-in exact-page pair used by the 128K batch-7 memory
experiment:

```bash
# First export the common absolute workspace variables from the complete
# workflow below.
FASTAFD_TASK_ROOT=/path/to/experiment \
FASTAFD_CUDA_METRIC_PLAN=/path/to/experiment/CASES.csv \
FASTAFD_SWEEP_CONTRACT=comprehensive \
FASTAFD_AFD_MEMORY_RATIO=0.90 \
FASTAFD_AFD_NUM_PAGES=14344 \
FASTAFD_AFD_KV_CAPACITY_TOKENS=918016 \
FASTAFD_REQUIRE_CAPACITY_MAX=1 \
./run_afd.sh qwen3 128k 7 16:1 1 4
```

The source checkout must be clean by default. To run a deliberate patch set,
pass both `FASTAFD_EXPECTED_HEAD` and a strict
`FASTAFD_EXPECTED_SOURCE_MANIFEST`.

## Sweep worker

`run_afd_bundle.sh` repeatedly claims same-tray cases from a CSV plan through
`afd_online_case_pool.py`. Set at least `FASTAFD_TASK_ROOT`, `FASTAFD_PLAN`,
`FASTAFD_ALLOCATED_TRAYS`, and `FASTAFD_SUBMIT_QOS`; source, image, venv,
result, metric, and pool paths are separately overrideable. The worker stops
claiming with 15 minutes left, fails closed after two case failures, and never
submits another allocation itself.

`extract_cuda_wall.py` and `cuda_execution_span.py` implement the corrected
warmup-plus-15 target-batch CUDA metric. Outlier limits come from each plan row
rather than case-specific exceptions.

## Baseline and model profiles

`run_vllm.sh` is the generalized pinned vLLM baseline runner retained from the
earlier OCI-HSG reproduction and wide-EP sweeps. `prepare_model_profile.py`
builds deterministic long-context model profiles. `run_afd.sh` is the single
canonical AFD launcher: it covers Qwen3 and MiniMax, direct one-case submission,
in-allocation execution, topology sweeps, all supported placement policies,
DeepEP/MegaMoE backends, production/perfect routing, and strict NSYS metrics.
Historical scratch or worktree copies are evidence only and must not be used as
launchers. The retired fixed-layout launcher used `FASTAFD_SLURM_QOS`; use the
canonical `FASTAFD_SUBMIT_QOS` interface instead.

The retired launcher's explicit `nodes` argument maps to the canonical pmap
ratio when FFN EP is 4: `ratio = nodes - 1`. Thus its published Qwen3 8K and
16K layouts are `7:1` and `11:1`, and the MiniMax layout is `17:1`. MiniMax is
intentionally fail-closed to the proven legacy-placement, TP1/EP4, 18-tray
contract; Qwen3 retains the wider explicit topology interface.

The unified launcher does not hide a backend choice behind a separate wrapper.
Legacy pmap/reproduction runs default to `megamoe_m2n`, legacy comprehensive
sweeps default to `deepep`, and attention-only placement defaults to same-rank
`megamoe`. Set `MINISGL_AFD_MOE_BACKEND` explicitly whenever a non-default
comparison is intended. All four choices—placement, backend, model, and routing
mode—are recorded in the result and incompatible combinations fail before
submission; the launcher never retries with another backend.

Every direct and bundled run derives a shape-complete case ID such as
`i8192-fep4-r7-atp1-b96` (or `rMIN-MAX-...` for irregular prompts).
Comprehensive MegaMoE plans may append the explicitly validated weight dtype;
the pmap path does not use the old ambiguous `single` result ID.

Reference reusable scripts from this source directory; do not copy or fork
`run_afd.sh` into a task workspace. Task-specific plans, state, logs, and
results belong under that workspace, not in Git.

## Reproduce a topology sweep on OCI-HSG

Keep source, live results, and archives separate. On the shared remote, use the
single canonical checkout and keep it immutable until every submitted job is
terminal; do not create a per-run source worktree:

- source checkout: `/lustre/.../users/$USER/github/FastAFD`;
- live task root: `/lustre/.../users/$USER/<sweep-name>`;
- completed archives: `fastafd_reproduce/archives/completed_sweeps/`;
- local checkout: source, reusable controls, project memory, and visualizations
  only. Do not copy full result trees into local `scratch/`.

The commands below show the required structure. Replace the abbreviated Lustre
prefix once and keep all derived paths absolute:

```bash
USER_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/$USER
SOURCE_REPO=$USER_ROOT/github/FastAFD
TASK_ROOT=$USER_ROOT/afd_128k_sweep_YYYYMMDD
CONTROL_DIR=$SOURCE_REPO/scripts/experiments/afd/oci_hsg
REPRO_ROOT=$USER_ROOT/fastafd_reproduce
PLAN=$TASK_ROOT/CASES.csv
POOL=$TASK_ROOT/state/pools/CASES
IMAGE=$USER_ROOT/oci-hsg_onboarding/images/pytorch-25.10-py3-aarch64.sqsh
EP_VENV_DIR=$REPRO_ROOT/envs/minisgl-3c716194-cuda130.artifact-5175891
EXPECTED_HEAD=$(git -C "$SOURCE_REPO" rev-parse HEAD)

export FASTAFD_TASK_ROOT="$TASK_ROOT"
export FASTAFD_REPRO_ROOT="$REPRO_ROOT"
export FASTAFD_SOURCE_REPO="$SOURCE_REPO"
export FASTAFD_IMAGE="$IMAGE"
export FASTAFD_EP_VENV_DIR="$EP_VENV_DIR"
export FASTAFD_EXPECTED_HEAD="$EXPECTED_HEAD"
export FASTAFD_CUDA_METRIC_PLAN="$PLAN"
export FASTAFD_CUDA_METRICS_ROOT="$TASK_ROOT/report/metrics"
export FASTAFD_CUDA_EXTRACT_TEMP_ROOT="$TASK_ROOT/cuda_extract/tmp"

test -z "$(git -C "$SOURCE_REPO" status --porcelain)"
test "$EXPECTED_HEAD" = "$(git -C "$SOURCE_REPO" rev-parse HEAD)"
test -f "$IMAGE"
test -x "$EP_VENV_DIR/bin/python"
test -f "$PLAN"
mkdir -p "$TASK_ROOT/logs" "$TASK_ROOT/report/metrics" "$TASK_ROOT/cuda_extract/tmp"
"$EP_VENV_DIR/bin/python" "$CONTROL_DIR/afd_online_case_pool.py" \
  --plan "$PLAN" --state-root "$POOL" init
```

`FASTAFD_REPRO_ROOT`, `FASTAFD_SOURCE_REPO`, `FASTAFD_IMAGE`, and
`FASTAFD_EP_VENV_DIR` are required. The launcher intentionally has no
home-directory fallback for these paths; the former stale image fallback was
the cause of the first perfect-routing preflight failure.

Always record and pass the full 40-character source SHA. A matching short SHA
is insufficient. The perfect-routing sweep first failed preflight because its
stored full SHA was stale even though the displayed short SHA looked correct.
Also use the Lustre image path above: the old home-directory image path no
longer exists and caused an earlier Pyxis prelaunch failure. Both failures
happened before any case ran and consumed no benchmark result.

Before submitting the matrix, dry-run at least one row from every distinct
topology/capacity class. The example below is the 128K, A1:F1, EP4, batch-7
capacity-maximum row:

```bash
FASTAFD_DRY_RUN=1 \
FASTAFD_TASK_ROOT="$TASK_ROOT" \
FASTAFD_SOURCE_REPO="$SOURCE_REPO" \
FASTAFD_SWEEP_CONTRACT=comprehensive \
FASTAFD_AFD_MODEL_PLACEMENT=attn-only \
FASTAFD_AFD_NUM_MB=2 \
FASTAFD_AFD_MEMORY_RATIO=0.90 \
FASTAFD_AFD_NUM_PAGES=14344 \
FASTAFD_AFD_KV_CAPACITY_TOKENS=918016 \
FASTAFD_REQUIRE_CAPACITY_MAX=1 \
MINISGL_AFD_MOE_BACKEND=megamoe \
MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp4 \
"$CONTROL_DIR/run_afd.sh" qwen3 131072 7 1:1 1 4
```

Submit one online worker for each `allocated_trays` group in the plan. Keep the
number of simultaneously runnable allocations within the current short-QoS
limit; use dependencies or submit additional groups only as slots free up.
This is the direct worker form used by the completed 42-case sweeps:

```bash
TRAYS=2
sbatch --parsable \
  --nodes="$TRAYS" --segment="$TRAYS" --qos=short --time=02:00:00 \
  --output="$TASK_ROOT/logs/trays${TRAYS}-%j.out" \
  --error="$TASK_ROOT/logs/trays${TRAYS}-%j.err" \
  --export="ALL,FASTAFD_TASK_ROOT=$TASK_ROOT,FASTAFD_PLAN=$PLAN,FASTAFD_POOL_ROOT=$POOL,FASTAFD_ALLOCATED_TRAYS=$TRAYS,FASTAFD_SUBMIT_QOS=short,FASTAFD_REPRO_ROOT=$REPRO_ROOT,FASTAFD_SOURCE_REPO=$SOURCE_REPO,FASTAFD_CONTROL_DIR=$CONTROL_DIR,FASTAFD_EP_VENV_DIR=$EP_VENV_DIR,FASTAFD_IMAGE=$IMAGE,FASTAFD_EXPECTED_HEAD=$EXPECTED_HEAD,FASTAFD_CASE_TIMEOUT_SECONDS=3600,FASTAFD_AFD_MODEL_PLACEMENT=attn-only,FASTAFD_AFD_NUM_MB=2,FASTAFD_POOL_SELECTION_MODE=auto,FASTAFD_AUTO_RELEASE_FAILED=0,FASTAFD_MIN_REMAINING_SECONDS=900,FASTAFD_MAX_FAILURES=2,MINISGL_AFD_MOE_BACKEND=megamoe,MINISGL_MEGAMOE_EXPERT_WEIGHT_DTYPE=fp4" \
  "$CONTROL_DIR/run_afd_bundle.sh"
```

For the diagnostic perfect-router variant, add
`MINISGL_AFD_PERFECT_ROUTING=1` to `--export`. Leave it unset for the production
router. Do not silently fall back between the two.

Use the pool summary plus a strict campaign audit as the completion gate. A
Slurm `COMPLETED 0:0` state alone is not sufficient. The accepted 42-case
contract is: 42 terminal cases; no claims, pending cases, or failures; 15
retained post-warmup CUDA samples per case; zero unexpected outliers; all
metric/result hashes valid; and 1,512 `.nsys-rep` files.

## Why runs previously appeared to hang

The recurring delay was after measurement, not in the benchmark kernels. In an
audited older runner, all 17 examined cases had already made capture, sample,
and coordinator-trace artifacts durable, then spent 210--214 seconds in
Ray/process-tree teardown. Waiting longer could not improve any metric.

The current `run_afd.sh` therefore watches three durability gates:
`capture-complete.json`, `sample.json`, and
`ray_logs/coordinator_nvtx_cpu.json`. Once all three exist and are nonempty, it
terminates only that allocation's driver, waits 10 seconds, and uses a bounded
KILL if needed. Consequently, a line such as `driver status 137` is expected
when it is paired with `post-capture-driver-termination.txt` and the enclosing
case/bundle completes successfully. Do not classify that pair as a failed or
hung GPU run.

NSYS export can also look idle. It is still progressing when `.qdstrm` or
`.nsys-rep` sizes/mtimes, per-rank trace counts, or the minute-level
`gpu-progress/` records advance. Only treat it as a stall when those bounded
signals stop advancing and the required durability files remain absent; the
per-case watchdog then fails closed. A ZMQ `Address already in use` message is
a separate control-port collision, not a hang: let the pool archive the attempt
and retry only that exact case with a fresh ordinal/port block.

Failures before `capture-complete.json` are not teardown hangs. For example,
merged-launcher validation job `6918910` stopped during M2N graph warmup because
NVCC rejected a malformed scheduler template instantiation. CPU-only NVCC job
`6919340` reproduced the two compiler errors, and recheck `6919396` compiled the
same generated Qwen kernel after the missing explicit
`kRankReadyRecvCounts=false` argument was restored. Diagnose this class from
the first worker traceback and the retained generated `kernel.cu`; do not wait
for NSYS files that the run never reached.

Treat the submitted launcher path as immutable until its Slurm job is terminal.
Bash reads a script incrementally, so replacing that file during execution can
invalidate its current byte offset and surface as a late parse error even after
capture has finished. Validation job `6919438` demonstrated this operational
failure: inference, the sample, the coordinator trace, and all 32 Nsight reports
were durable, but a concurrent launcher update caused the outer shell to parse
the wrong later bytes. Stage the final launcher and source manifest before
submission; never sync into that worktree while one of its jobs is active.
Immutable-file rerun `6919706` subsequently completed the same Qwen case
`0:0`, proving that no benchmark-path retry or fallback was required.

## Archive completed results

Use `codex_scripts/archive_sweep_cases.py` from a CPU data-mover allocation.
For a strict AFD sweep, retain only these payload classes unless a task has an
explicitly broader preservation requirement:

```bash
python3 "$SOURCE_REPO/codex_scripts/archive_sweep_cases.py" \
  --source-root "$TASK_ROOT" \
  --archive "$REPRO_ROOT/archives/completed_sweeps/SWEEP.tar.zst" \
  --sweep-name "SWEEP TITLE" \
  --include 'results/*/experiment/ray_logs/*.nsys-rep' \
  --include 'results/*/experiment/afd-result.json' \
  --include 'report/metrics/*.json' \
  --include 'report/completed-plan-audit.json' \
  --include 'report/completed-plan-audit.tsv' \
  --include 'report/perfect-vs-original.txt' \
  --exclude-dir-name tmp \
  --zstd-threads 16
```

The archive is written through `.partial` and renamed only after `zstd -t` and
exact tar-member validation pass. Its members are only the selected result
payloads: no scripts, README, or other documentation is embedded. Preserve the
external `.manifest.tsv`, `.sha256`, and `.summary.json` sidecars. Delete an
uncompressed task root only after comparing the summary counts with the strict
completion audit.
