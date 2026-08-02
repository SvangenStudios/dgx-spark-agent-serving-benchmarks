# Agent Serving on DGX Spark — benchmarks, tools and operational findings

Reproducible measurements of how **DeepSeek V4 Flash 0731** (vLLM/DSpark, TP=2 over RoCE)
and **Laguna S 2.1** (vLLM 0.25.1, single node) behave on NVIDIA DGX Spark (GB10, sm_121,
128 GB unified memory) — with a focus on what matters for **agent workloads**: KV-cache
capacity, scheduling fairness under mixed load, prompt cache locality, and startup/recovery
behavior.

> **Status: pre-release.** Measured 2026-08-01/02 on the same three-node DGX Spark
> cluster, using two distinct model and runtime stacks. Includes two real coding-agent
> tasks (Hermes v0.19.0). In the captured 10-turn bug-fix run, the client achieved
> **97–99% prefix-cache reuse per turn**, with only ~200–450 tokens re-prefilled at each
> step — the framework builds prompts append-only.

---

## Novelty and prior work

**Established prior work — none of this is claimed as new:**

- Prefix caching benefits from stable prefixes; "static first, dynamic last" is documented
  practice by API vendors ([Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
  [OpenAI](https://developers.openai.com/api/docs/guides/prompt-caching)).
- Prefill and decode have fundamentally different performance characteristics; chunked
  prefill exists precisely to interleave them
  ([vLLM docs](https://docs.vllm.ai/en/v0.8.2/performance/optimization.html)).
- Larger `max_num_batched_tokens` uses more activation memory, leaving less for KV cache
  (vLLM docs).
- Multilingual speculative decoding often has lower draft acceptance
  ([Speculative Decoding Across Languages, arXiv:2605.30580](https://arxiv.org/abs/2605.30580)).
- The UMA memory-reporting problem on GB10 is documented in
  [vLLM #48140](https://github.com/vllm-project/vllm/issues/48140) (closed, not planned).
- Related prompt-divergence tooling exists in the
  [VS Code Cache Explorer](https://code.visualstudio.com/docs/agents/agent-troubleshooting/cache-explorer).
- DeepSeek V4 Flash on 2× DGX Spark at 1M context
  ([guide thread](https://forums.developer.nvidia.com/t/guide-deepseek-v4-flash-on-2x-dgx-spark-gb10-reproducible-vllm-serving-recipe-up-to-1m-token-context/374742)),
  and ~60–67 tok/s on code with DSpark
  ([results thread](https://forums.developer.nvidia.com/t/deepseek-v4-flash-dspark-on-2x-dgx-spark-gb10-big-single-stream-speed-boost-60-67-tok-s-1m-context-now-with-concurrency/374846)),
  were previously published on the NVIDIA DGX Spark forum.

**Our contribution:**

1. **Quantified the `max_num_batched_tokens` 8192 → 2048 trade-off on 2× GB10** in one
   controlled run: **+67% KV pool, −69% p95 token-gap jitter, −3.9% prefill — and an
   unchanged ~7% decode share under concurrent prefill.**
2. **Documented a likely scheduling deviation:** vLLM's chunked-prefill design states that
   pending decodes are prioritized before prefill; on this DSpark/speculative stack, decode
   received ~7% of capacity during long prefill regardless of chunk size, and new requests
   were admitted only after prefill completion (221× normal TTFT). A minimal reproduction is included in `repro/`. Issue filing against the patched
   stack is pending; upstream vLLM should only be targeted if the behavior is reproduced
   without the DSpark patches.
3. **Measured prefill scaling, decode scaling and admission separately** on the same
   configuration: prefill aggregate flat from N=1→6 (serialized); Swedish-prose decode aggregate
   increased 107% by N=4 (N=6 added only 9% more while maximum TTFT rose sharply); code
   content reached 126 tok/s aggregate at N=4. Admission blocked during long prefill.
4. **Systematically reproduced #48140** across three boots (4.22 / 5.18 / 5.34 GiB reported
   available KV regardless of process history and JIT state; 262K context requires
   18.35 GiB → deterministic refusal; 32K requires ~2.3 GiB → boots), verified a working
   32K profile, and documented two candidate mitigation paths for 262K (`drop_caches` or
   a local UMA-aware memory-accounting patch). The 262K causality test remains pending.
5. **Replicated prompt-locality behavior across two substantially different vLLM stacks**
   (model, tokenizer, chat template, parser, FP4 backend, vLLM version, speculation method,
   topology) with pre-registered hypotheses — including one honestly failed prediction.
6. **Released standalone tools:** a wire-capture proxy for arbitrary OpenAI-compatible
   clients, and a locality analyzer targeting vLLM-compatible servers that expose a
   `/tokenize` endpoint (reports vLLM-style block reuse).
7. **Field data for multilingual speculative decoding** on this stack: draft acceptance
   71% (code) / 28.4% (English prose) / **19.5% (Swedish prose)** — a practical
   replication of the arXiv findings above, on concrete hardware.

---

## Hardware and runtime

| | |
|---|---|
| Hardware | 2× NVIDIA DGX Spark (GB10 Grace-Blackwell, sm_121), 128 GB unified LPDDR5X each |
| Interconnect | RoCE, direct-attached, MTU 9000 |
| Model A | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `7872f01b`, 156 GB checkpoint on disk, TP=2 |
| Runtime A | vLLM `0.21.1rc1.dev339` + DSpark overlay ([recipe `tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) @ `d728faee`) |
| Model B | `poolside/Laguna-S-2.1-NVFP4`, 67 GB checkpoint on disk, single node |
| Runtime B | vLLM 0.25.1 upstream, DFlash speculation |

Full details: [`results/system-info.txt`](results/system-info.txt).

## Main results

See [`results/RESULTS.md`](results/RESULTS.md). Highlights:

- **Chunk size is a jitter and KV-capacity lever, not a fairness lever.** Token gaps under
  prefill are exactly `chunk_size ÷ prefill_rate`; decode share stays ~7% regardless.
- **The 1M serving configuration boots successfully; retrieval quality was verified at
  32K, 128K, 512K and 900K with 12/12 needles recovered and no distractor hits.** It is
  batch capacity, not an interactive mode — a 900K prefill takes ~15 minutes and starves
  everything else on the server.
- **Moving the same mutating field from the bottom to the top of the prompt increased
  per-turn cost by 39.6× on DS4 and 13.7× on Laguna** (relative to each clean baseline,
  dirty-top cost 27.7× and 15.1× respectively; 0% vs ~98% reusable prefix). Cache reuse
  is aligned to each runtime's prefix-cache block granularity — a divergence in the first
  block resulted in 0% reusable prefix on both stacks.
- **Cold vs warm start** (Laguna): ~24 min FP4 JIT (one-time, persistent cache) + ~12 min
  single-threaded weight loading (every start) + capture. Readiness should be measured to
  the first successful generation, not the open port.

## Tools

| Tool | Purpose |
|---|---|
| `scripts/capture_proxy.py` | Records the exact serialized payload an agent client sends (what goes over the wire is what determines the cache). 0700/0600 permissions, capture dir gitignored. |
| `scripts/prompt_locality.py` | Given two consecutive prompts: first divergent token (server's own tokenizer), invalidated block index, reusable cache fraction, which message/field changed. `REDACT=1` suppresses content snippets. Three confidence tiers: exact / positional approximation / calibrated time estimate. |
| `scripts/cache_locality.py` | TTFT cost as a function of *where* a mutating field sits (clean / bottom / middle / top). |
| `scripts/decode_concurrency.py` | Decode scaling N=1,2,4,6 with TTFT, p50/p95 token intervals, draft acceptance per level. |
| `scripts/prefill_concurrency.py` | Prefill scaling N=1,2,4,6. |
| `scripts/prefill_decode_overlap.py` | What happens to ongoing decode and new requests during a large prefill. |
| `scripts/context_depth_retrieval.py` | Needle retrieval at 32K/128K/512K/900K with distractors. |

## Measurement discipline (learned the hard way)

1. On this DSpark build, use `"stream": false` for throughput measurement — streamed
   output represented speculative decode steps rather than accepted output tokens.
2. Warm up with **long** generations; three short calls are not "warm", and the effect decays after ~30 min idle.
3. Measure p50/p95 token intervals, not just the largest gap — partial starvation looks "green" otherwise.
4. Salt every prompt uniquely, or you are measuring the prefix cache.
5. Filter `repetition_penalty` on this DSpark build — the recipe documents an
   illegal-memory-access crash. We verified `presence_penalty` and `frequency_penalty` as
   safe, but did not repeat the destructive `repetition_penalty` test.
6. Do not judge model loading or memory leaks from process RSS on GB10 — CUDA/UVM allocations don't show there.
7. Do not read an instantaneous ETA as a trend.
8. Compare the same phase between runs (early shards vs early shards).
9. `pkill -f` / `pgrep -f` self-match — even across SSH, where the pattern sits in the remote
   shell's command line. Use `[b]racket` patterns or PID files.

## Threats to validity

See [`results/THREATS-TO-VALIDITY.md`](results/THREATS-TO-VALIDITY.md) — including what is
**not** threatened, one pre-registration flaw we left in place rather than rewriting, and
the experiment that would settle each open point.

## Pending before v1.0

- [x] Real agent workload capture (Hermes) — done, see results §9
- [x] English translation of all script output strings and documents
- [ ] 262K + `drop_caches` causality test for #48140
- [ ] 72-hour soak on the final 2048 agent profile
- [ ] File the scheduling issue against the patched DSpark stack
- [x] Minimal reproduction for the scheduling deviation — see `repro/`
