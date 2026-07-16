# Project Memory

The OCI HSG reproduction of the public FastAFD GB200 NVL72 performance claims
is complete for Qwen3 and MiniMax-M2.5 at 8K/16K. All four vLLM baselines now
have DP=EP4/8/16/32/64 tables using exact EP8+ KV-capacity ceilings and sparse
corresponding CUDA graph captures. EP16 is the best wide-EP point in every
case. See
[`CODEX_PROJECT_REPRODUCE.md`](CODEX_PROJECT_REPRODUCE.md) for the two retained
launchers, integrated wide-EP baseline support, pinned provenance, measurement
contract, and signed result evidence.
