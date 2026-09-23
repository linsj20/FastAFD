# Cluster operations and project management

**Read this file before any cluster SSH, search, transfer, build, job submission,
monitoring or cleanup, and before organizing sources/results.** This is the
central reference for cluster operations and high-level project management.
Keep common rules here and differences in the cluster sections below;
campaign docs record their plans/results and link here instead of duplicating
operational policy. Add future clusters here. Current user instructions prevail.

Verified 2026-09-15. HSG retains its existing campaigns. AGA bring-up is active;
see [AGA environment and tests](CODEX_PROJECT_aga_bringup.md). AGA dependencies
are downloaded directly and model conversion runs on AGA, without migration.

## Do not do

- **No expensive broad searches:** do not recursively `grep`/`rg`, `find`, `du`,
  `ls -R`, or run Python walks over home, shared mounts, all sources, caches,
  results or dependency trees. Start from project memory, known paths, source
  file lists and run manifests. Bound both directories and file types; cap depth
  when traversing. Piping an expensive traversal to `head` is not a scope limit.
- Do not run builds, compression, large Nsight/SQLite exports, dataset processing
  or mass filesystem cleanup on login nodes. Use CPU allocations for sustained
  computation and data nodes for transfers/bounded file operations. CPU nodes
  still share storage: limit traversal and metadata concurrency there too.
- Do not run repeated tree-wide size/count/checksum audits or broad rsync dry
  runs. Use existing manifests, explicit file lists and a single focused check.
  No `py_check` or hash/version audit unless the user requests it.
- Do not transfer entire workspaces, result/Ray-log trees or environments to a
  new cluster as incidental setup. Copy only the requested files; source mirrors
  exclude generated data and historical versions. Do not follow large-storage
  symlinks during recursive copies/scans.
- User requires no connections or artifact copies between HSG and AGA. Provision
  AGA images/packages/models by direct external downloads; generate converted
  checkpoints on AGA. Maintained source comes from the local authoritative checkout.
- Do not poll Slurm in tight loops, issue unfiltered cluster-wide job queries,
  reserve GPUs for CPU work/transfers/manual waiting, or submit speculative jobs.
- Do not create a source copy per optimization, edit a source consumed by live
  jobs, delete unrelated results, or silently add fallback behavior.

## Shared practices

- Use plain SSH aliases. Login nodes are for lightweight shell work, submission
  and filtered status queries; use data nodes for rsync/scp and Nsight copies.
  Heavy analysis/builds use CPU allocations, never idle GPU reservations.
  Use dedicated VS Code nodes (data copiers only when none exist). Keep Codex
  local; do not launch agents on compute/data nodes.
- Prefer cached MARS Scheduling Tools when available. Otherwise query only your
  job IDs together, ordinarily every 5min; avoid tight scheduler polling.
  Request explicit resources/time and release allocations when work finishes.
- Keep source, generated output and dependencies separate. Local checkout is
  authoritative; merge changes locally before synchronizing its remote mirrors.
  Mirror an explicit source-file list, including intended current local edits.
  Exclude results, caches, environments, binaries and historical source copies.
  Preserve remote-only work before replacing it. Source sync is not validation.

## Project management

### Ownership and layout

The local Git checkout owns maintained code, reusable `codex_scripts/`, and
project documentation. `CODEX_PROJECT.md` is a concise status/index; topic docs
record goals, decisions, attempts, results and next steps with artifact links.
This file owns organization and lifecycle policy. Task-specific manifests carry
execution details; do not turn project memory into logs or duplicate this policy.

```text
FastAFD/
  source/
    local-mirror/               # current local working-tree mirror
    <direction>/                # independently diverging implementation
  artifacts/
    <subproject>/
      <run>/                    # plans, logs, results, traces, caches, provenance
      reports/                  # cross-run summaries, tables and plots
    archives/<subproject>/<run>/ # retained completed result payloads, if archived
```

Create directories only when needed. Existing artifact paths remain valid;
this layout is not a request for bulk migration. Large dependencies/environments
stay in their designated shared locations, referenced by runs rather than copied
into every source. Local `scratch/<subproject>/` holds compact evidence and task
plans; use the established AFD case hierarchy below for retained model results.

### Source lifecycle

Each source represents a **direction, not an optimization/run/date**. Iterate in
place; create another source only when implementations must diverge and coexist.
Record its parent, purpose and merge/retirement condition. Save Git revision and
exact dirty patch with each run; finish queued/running jobs before editing their
source. Delete obsolete source trees instead of making per-optimization archives.

An experiment does not become accepted because it exists remotely. Validate the
relevant correctness/performance contract, merge accepted changes into the local
checkout, then sync its mirrors. Retire merged/rejected directions after unique
useful changes and required provenance are accounted for. Preserve useful
performance paths until a matching comparison supports replacement. A retained
result version does not require another maintained source directory.

### Runs, artifacts and archives

- Give each subproject a stable topic name and each run a descriptive unique
  label; use date/job suffixes for repeats. Keep all generated run data under its
  artifact directory. New parameters/retries create run records, not source trees.
  Never overwrite accepted evidence or relabel a failed run as passing.
- A run manifest records purpose, cluster, source revision/dirty patch, launch
  plan, environment/image, workload, placement, job IDs, status and result paths.
  Keep a compact outcome explaining acceptance, rejection or unresolved questions.
  Cross-run reports link their inputs and preserve the comparison contract.
- When work completes, keep accepted milestones, useful baselines and diagnostic
  evidence. Failed attempts need concise outcomes; retain raw payloads when they
  are still needed for diagnosis/reproduction. Protect active validation and
  unrelated user data during cleanup; remove superseded copies and expendable caches.
- **Archives hold selected completed result payloads, not obsolete source trees.**
  A completed run can remain in place; do not archive every run automatically.
  For an archive, keep its manifest, summary, original/new paths and retrieval
  instructions beside the payload. Link it from the subproject memory/index.
  Verify the intended files are retained before removing an original payload;
  avoid duplicate live/archive copies and do not add hash audits by default.
- Preserve raw traces remotely and copy only compact evidence locally. Compress
  or package only a known bounded file set on an appropriate CPU/data node.
  Archiving must not rewrite metrics or erase provenance. Keep deletion/migration
  records when paths referenced by prior experiments are removed or changed.

Keep persistent payloads on large storage; scope temporary/JIT caches by run and,
where required, node/rank. Do not put experiment payloads in home or `/tmp`.

### Nsight and local evidence

Keep raw Nsight captures remotely. Retrieve only the metric-selected attention
and FFN reports through the data node (`rsync -a --partial`), never whole result/
Ray-log trees. Verify nonempty files and record original paths, roles/ranks,
sizes, job, source/patch, image, placement and workload settings. Follow
[the retained-case hierarchy](CODEX_PROJECT_nsys_traces.md); use distinct cluster
provenance/run labels to avoid overwrites. CPU-only exports stay remote unless
needed. Compare results only under matching workload and measurement contracts.

## Cluster access and storage

| Cluster | Login | Data-copy alias / hostname |
|---|---|---|
| HSG | `ssh oci-hsg` | `oci-hsg-data` / `oci-hsg-cs-001-dc-01.nvidia.com` |
| AGA Slurm 2 | `ssh oci-aga` | `oci-aga-data` / `oci-aga-slurm-2-dc-01.nvidia.com` |

Both data hosts were accessed successfully. User-facing project root on both:
`/lustre/fsw/portfolios/coreai/users/shengjiel/FastAFD`.
Same path, **separate cluster storage**. Physical parent directories:

- HSG: `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel`.
- AGA: `/scratch/fsw/portfolios/coreai/projects/coreai_comparch_sysarch/users/shengjiel`.
  `~/scratch` already links to `/lustre/fsw/portfolios/coreai/users/shengjiel`.

## HSG

Continue execution here. Main source is now `FastAFD/source/local-mirror`;
profiling launcher default updated and remote shell syntax passed. Git linked
worktree paths were repaired after moving the mirror from `FastAFD/local-mirror`.
Its retained Git metadata predates the current mirrored files: use local source
revision/dirty patch for new run provenance, not remote HEAD alone.

Retain independent `source/AFD_P` and profiling dependencies `source/iket` /
`source/iket-smodel`. Deleted 186 obsolete optimization trees at user request.
Deletion record: `artifacts/source_history/20260915/manifest.json`;
no source archive remains. Existing experiment result artifacts are preserved.
Generated mirror cache/scratch/reports moved to `artifacts/mirror_generated_20260915`.
Old historical source paths are unavailable after deletion; new plans must use
retained sources and current paths. Support: `#oci-hsg-cs-001-support`.

## AGA / NVL576

- September22 rack-scale submission: each domain has16physical racks,9trays×4GPUs
  per rack in the topology definition. No domain currently exposes all144trays;
  d002 has141,missingR04-T05/R13-T06/R15-T02. User approved512GPUs(A496:F16),
  eight trays in each rack. Exact128-node request needs `--segment=8
  --consolidate-segments`; omitting these failed topology validation. GPU39892
  accepted underQOSnormal. See [rack scaling](CODEX_PROJECT_aga_bringup.md#rack-scaling-and-communication)
  for the reusable mapper, per-rack routing and batch4–7 Nsight sequence.

- September22 optimized Qwen3 promotion uses `sbatch --propagate=NONE` for
  GPU and analysis allocations. Current login limits are NPROC300/400 and
  AS/RSS8GiB; inheriting them caused Ray startup fork failure before inference.
  CPU39750 confirmed propagation-disabled NPROC1543356 and unlimited AS;
  GPU39751 then completed the32GPU one-rack capture. See
  [promotion](CODEX_PROJECT_aga_bringup.md#qwen3-promotion) for exact plans and evidence.
- `FastAFD/source/local-mirror` holds current source/docs without `.git` or
  historical snapshots. Fresh dependencies now live under `FastAFD/dependencies`;
  active setup/tests under `FastAFD/artifacts/aga_bringup/20260915`. Both original
  FP8 models downloaded directly; Qwen3.5 FP4 conversion25015 completed on AGA
  (4 GPUs,2m24s), with a separate CPU finalizer. Alignment/perf validation pending.
- CPU image import must use node-local `ENROOT_TEMP_PATH` (used `/dev/shm`),
  since Lustre-backed extraction failed the overlay mount. Keep the downloaded
  layer cache and final image on project storage. All CPU-only preparation,
  builds, checkpoint finalization and trace analysis use CPU allocations.
- Account `coreai_comparch_sysarch`. Live partitions: `batch` max 4h,
  `batch_long` max 7d, `cpu` max 7d, `cpu_datamover` for scheduled transfers.
  QOS `short`: max 2h/64 nodes/min 4 GPUs; `normal`: min 4 GPUs.
  `cpu-short`: max 4h/20 nodes; `cpu-normal`: max 1d/2 nodes. CPU-only work uses CPU
  QOS without GPUs. CPU and one-node GPU `sbatch --test-only` passed; no job ran.
- aarch64 compute nodes, 4 GPUs/node; `topology/block`, 9-node base blocks.
  Names encode domain/rack/tray, e.g. `nvl576d002-R01-T01`. MOTD recommends
  `--segment=[1-9] --consolidate-segments`; both pass one-node test-only parsing
  despite omission from help. Exact segment semantics still need the AGA guide.
  Record/verify allocated domain and racks; 144 nodes × 4 GPUs does not guarantee one domain.
- MOTD recommends NCCL 2.30.7+ (`nvcr.io/nvidia/pytorch:26.07-py3`), minimum 2.30.3
  (`26.05-py3`) for NVL576. Pyxis options are present. Pin tested image/dependencies;
  HSG binaries/caches are not assumed compatible with AGA's GPU/ABI.
- MOTD labels cluster M2/pre-production. Support: `#oci-aga-slurm-2-support`.
  Open: exact segment recipe, approved model/image paths, minimal application bring-up.

## References

[MARS support](https://nvidia.atlassian.net/wiki/spaces/HWINFCSSUP/overview),
[Cluster Best Practices](https://nvidia.atlassian.net/wiki/spaces/HWINFCSSUP/pages/2441645950/Cluster+Best+Practices),
[AGA cluster guide](https://nvidia.atlassian.net/wiki/spaces/HWINFCSSUP/pages/3730983883/oci-aga-slurm-2+Cluster-Specific+Information).
Best Practices retrieved via NVInfo/Glean. AGA guide retrieval failed (HTTP502/
not indexed); AGA facts above come from live MOTD, Slurm, storage and SSH checks.
