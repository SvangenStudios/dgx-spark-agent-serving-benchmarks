# Minimal reproduction: one decode step per prefill chunk, and blocked admission

> **2026-08-02: numbers below are the token-weighted re-measurement.** An earlier version of
> this page carried figures from an instrument that counted SSE events as tokens, used no
> prefill window and took a cold reference. Those are superseded. See
> [`../results/CORRECTION-2026-08-02.md`](../results/CORRECTION-2026-08-02.md).

## Claim

With chunked prefill enabled on this DSpark/speculative stack, an ongoing decode stream is
severely starved while a long prefill runs: it receives **one speculative decode step per
prefill chunk**, yielding ~3 accepted tokens regardless of chunk size — on the order of
0.1–0.2% of the token budget. Because the per-step yield does not scale with chunk size,
decode throughput scales with the number of chunks: reducing `--max-num-batched-tokens` 4×
(8192 → 2048) improves decode share 2.9×, from 1.7% to 5.0% of the same stream's
undisturbed rate. New requests are not admitted until the prefill finishes (145–159 s
observed, vs 1.7 s normal TTFT) in **both** profiles.

vLLM's chunked-prefill design states that pending decode requests are prioritized and
batched before prefill is scheduled. This stack appears to deviate from that stated intent.

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

The script starts a streaming generation, lets it run undisturbed for 25 s to establish its
own baseline, injects a ~256K-token prefill, then a new short request 20 s after that. It
reports token-weighted decode throughput inside the prefill window (submission → the long
request's first output token), p50/p95/max output-chunk intervals, the new request's TTFT,
and the prefill rate. Token counts come from streamed `token_ids` and are cross-checked
against the server's own `usage.completion_tokens`; run `../scripts/probe_token_ids.py`
first to confirm the server supports that.

## Measured results

Medians of three repetitions per profile (2048 additionally repeated after a server
restart, n=6; spread in `../results/RESULTS.md` §4).

| Metric | chunk 8192 | chunk 2048 |
|---|---|---|
| Undisturbed baseline (same stream, before submit) | 36.3–39.0 tok/s | 36.3–39.0 tok/s |
| Decode during prefill | 0.65 tok/s (**1.7%**) | 1.90 tok/s (**5.0%**) |
| Decode tokens delivered during prefill | 106 | 332 |
| Prefill chunks for the 262K prompt | 32 | 128 |
| Accepted tokens per chunk | 3.3 | 2.6 |
| p95 output-chunk gap | 6.10 s | 1.64 s |
| max output-chunk gap | 6.23 s | 1.66 s |
| TTFT of new request | 145.4 s | 159.2 s |
| Prefill rate | 1,584 tok/s | 1,462 tok/s |

The observed p95 gaps closely tracked `chunk_size ÷ effective_prefill_rate` (5.17 s and
1.40 s predicted, both overshot by ~18%) — decode gets one scheduling opportunity per
chunk. What that opportunity yields is ~`k` accepted speculative tokens
(`MTP_NUM_TOKENS=5`), and crucially it does **not** scale with chunk size. Hence:

```
decode_tokens ≈ (prefill_tokens / chunk_size) × accepted_tokens_per_step
```

predicting 4× improvement from 8192 → 2048; measured 3.1×.

An earlier version of this page claimed the decode share was invariant across chunk sizes.
That was an artifact of counting SSE events rather than tokens: the event *rate* was
nearly identical between profiles (1.70 vs 1.68 events/s) while the token content per
event was not.

## Expected behavior

Per the chunked-prefill design, pending decodes should be prioritized and interleaved so
that inter-token latency remains stable when a large request arrives, and new requests
should not wait for the full prefill to complete.

## Notes

- Not yet reproduced on unpatched upstream vLLM (the DSpark NVFP4 deployment does not
  boot there), so the deviation is reported against the overlay stack first.
- Short-prompt concurrency is unaffected (TTFT 0.4 s at N=4) — the issue is specific to
  long prefill saturating compute.
