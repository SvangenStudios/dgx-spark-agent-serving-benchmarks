# Minimal reproduction: decode share pinned at ~7 % during long prefill, independent of chunk size

## Claim

With chunked prefill enabled on this DSpark/speculative stack, an ongoing decode stream
receives ~7 % of its undisturbed rate while a long prefill runs — and this share does
**not** change when `--max-num-batched-tokens` is reduced 4× (8192 → 2048). New requests
are not admitted until the prefill finishes (~150 s observed, vs 1.7 s normal TTFT).

vLLM's chunked-prefill design states that pending decode requests are prioritized and
batched before prefill is scheduled. This stack deviates from that stated intent.

## Environment

- 2× NVIDIA DGX Spark (GB10, sm_121), TP=2 over RoCE
- vLLM `0.21.1rc1.dev339+g1967a5627bc3` with the DSpark overlay
  (`tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark` @ `d728faee`)
- `deepseek-ai/DeepSeek-V4-Flash-0731`, `nvfp4_ds_mla` KV, block-size 256, DSpark k=5
- `--enable-chunked-prefill` on; `--async-scheduling` off (tested both; no difference)

## Steps

1. Serve with `--max-num-batched-tokens 8192`. Wait until warm (several long generations).
2. Run `python3 ../scripts/prefill_decode_overlap.py 256000 chunk-8192`
3. Restart with `--max-num-batched-tokens 2048`. Warm again.
4. Run `python3 ../scripts/prefill_decode_overlap.py 256000 chunk-2048`

The script starts a streaming generation, injects a ~256K-token prefill 5 s later, then a
new short request 20 s after that. It reports the ongoing stream's rate during the
prefill, p50/p95/max token intervals, the new request's TTFT, and the prefill rate.

## Measured results

| Metric | chunk 8192 | chunk 2048 |
|---|---|---|
| Reference decode (undisturbed) | 24.1 tok/s | 23.0 tok/s |
| Decode during prefill | 1.70 tok/s (**7.1 %**) | 1.68 tok/s (**7.3 %**) |
| p95 token gap | 5.197 s | 1.590 s |
| max token gap | 6.417 s | 2.085 s |
| TTFT of new request | 150.7 s | 158.1 s |
| Prefill rate | 1,529 tok/s | 1,469 tok/s |

The p95 gap tracks `chunk_size ÷ prefill_rate` almost exactly (5.36 s and 1.39 s
predicted) — decode gets one scheduling opportunity per chunk. But the decode *share* is
invariant: ~9 tokens per 8192-chunk vs ~2.3 tokens per 2048-chunk. Four times more
opportunities, four times fewer tokens each, net zero.

## Expected behavior

Per the chunked-prefill design, pending decodes should be prioritized and interleaved so
that inter-token latency remains stable when a large request arrives, and new requests
should not wait for the full prefill to complete.

## Notes

- Not yet reproduced on unpatched upstream vLLM (the DSpark NVFP4 deployment does not
  boot there), so the deviation is reported against the overlay stack first.
- Short-prompt concurrency is unaffected (TTFT 0.4 s at N=4) — the issue is specific to
  long prefill saturating compute.
