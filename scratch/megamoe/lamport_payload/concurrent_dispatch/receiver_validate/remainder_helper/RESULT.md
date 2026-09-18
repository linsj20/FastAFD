The narrow remainder helper is retained as the best measured **fixed EP8** ordinary kernel path. It batches two rows' loads before validation while keeping the main compute loop's compiler behavior close to the parallel parent. Markers remain in the caller; the helper contains no profiling markers.

| Ordinary kernel comparison | Helper | Previous best | Change |
|---|---:|---:|---:|
| Initial ABBA,20 instances/arm |37.2416us|37.3328us|-0.0912us|
| Uninterrupted ABBA,200 instances/arm |37.16655us|37.24400us|-0.07745us (-0.208%)|

Both orders improved in both jobs. Confirmation runs used the same source and caches,100 instances/pass, with no intervening instrumented capture. This is a small repeated gain, not a claim of statistical independence among replay samples.

The full [timing decomposition](comparison_7257013.md) keeps all three captures, operation starts, metadata start=0 and history pull at right. Median compute span was25.280us (parallel parent26.080, previous best25.152, history25.024). Remainder issuer wait was0.480/0.768/0.480us versus parent1.312/1.472/1.488us; admission stayed near0.608us after its own MMA. First and median MMA start improved against previous best in all three captures; compute end improved in two.

All144 exact-output and144 replay checks passed across both jobs. Buffer initialization/reset remained outside kernel timing. CPU builds used128 registers,24 stack bytes,zero spills; helper hadzero stack/local-memory instructions.

The separate producer-plus-kernel confirmation was56.181280 vs56.357280us, but the initial comparison regressed, so no consistent full-chain gain is claimed. All outliers remain in the audit/decomposition.

This is an experimental header overlay with the existing prepared-capacity guard and matching compact native descriptor. Preserve previous best1536; no general/default promotion. [Decision and provenance](decision.json).
