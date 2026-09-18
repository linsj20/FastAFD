The unchanged remainder helper improves both tested EP16 cases against the previous all-push best. It is still not a general winner: Qwen3.5 remains slightly slower than historical pull, and the larger Qwen3.5 tile is unsupported.

Ordinary MegaMoE-only means in µs, from job7259728 on one four-node NVL72 allocation. Each optimized arm has200 samples across an uninterrupted ABBA (100/pass); historical pull has100. No kernel settings were retuned; initialization/reset remains outside kernel timing.

| Case | Tokens total / rank | Previous all-push best | Helper | Reduction | Historical pull |
|---|---:|---:|---:|---:|---:|
| Qwen3-235B-A22B EP16 | 192 / 12 | 24.22975 | 23.90270 | 1.350% | 25.7731 |
| Qwen3.5-397B-A17B EP16 | 512 / 32 | 49.56815 | 49.30880 | 0.523% | 48.9744 |

Both run orders improve for each model. Qwen3 helper-minus-control differences are-0.3594/-0.2947us; Qwen3.5 differences are-0.2167/-0.3020us. These are measured means, not a statistical-independence claim for graph replay samples. All128 exact-output comparisons against the per-rank historical reference and160 replay checks pass;32 reference records were generated. Six CPU device builds also passed register gates with128 registers,24stack bytes andzero spills.

Qwen3 usesH4096/I1536/E128/topk8; Qwen3.5 usesH4096/I1024/E512/topk10. Both useEP16,150compute CTAs,8senders,bucket3840 and naturally select32-row tiles with256epilogue threads. Steady-buffer capacity guards remain satisfied (268/1056≤3840 upper bounds).

The proposed Qwen3.5 global912/rows57 case selects64-row tiles and128epilogue threads. The current compact implementation requires at most32-row tiles and256epilogue threads, so it was not launched or silently retiled. The512-token case is explicitly a different, supported test; this does not establish support at the larger batch. No kernel or heuristic was modified to fit these tests.

Separate max-rank producer-plus-kernel means in µs (not the table's scope):

| Model | Helper ABBA | Previous best ABBA | Historical pull, preserved outlier |
|---|---:|---:|---:|
| Qwen3-235B-A22B | 42.336160 | 42.759040 | 9238.859863 |
| Qwen3.5-397B-A17B | 70.142561 | 70.125921 | 9484.893799 |

Qwen3 full chain improves; Qwen3.5 is effectively unchanged/slightly worse. No cross-scope subtraction or dispatch/compute attribution is made. This validation used ordinary timings only, so it supplies no new fine-grained phase timings.

All ten ordinary passes, including dispersion and extrema, are retained in [ordinary_7259728.json](ordinary_7259728.json). [run_audit_7259728.json](run_audit_7259728.json) checks plan order, every rank, dimensions, token counts, timing scope, iteration counts and reset exclusion. [comparison.json](comparison.json) records exact aggregates and limitations. Raw Nsight reports remain in the remote artifact tree. GPU7259728 completed14m30s; CPU7259660 completed1m31s. No live jobs, default promotion or commit.
