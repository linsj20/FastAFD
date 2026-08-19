# OCI-HSG AFD experiment controls

These are the reusable controls promoted from the corrected Qwen3 sweeps and
the optimized-attention single-case runs. Generated plans, source manifests,
job-specific submit wrappers, node exclusions, logs, traces, and results stay
in the caller's experiment workspace rather than this source directory.

## Single case

`run_afd.sh` validates topology, KV capacity, source provenance, capture
policy, and result extraction before submitting or running one Qwen3 case. It
uses `run_afd_qwen3_parallel.sh` as the in-allocation preset and supports the
opt-in exact-page pair used by the 128K batch-7 memory experiment:

```bash
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
builds the deterministic long-context model profiles used by both launchers.
`run_afd_reproduce.sh` retains the older pinned Qwen3/MiniMax published-result
interface; `run_afd.sh` is the newer Qwen3 topology-sweep and one-case runner.

All scripts are intended to be copied or referenced from a task workspace;
task-specific controls and artifacts belong under that workspace, not in Git.
