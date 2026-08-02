# Reply to post in the NVIDIA DGX Spark forum thread

Paste the block below as a reply. Keep the original post as it is — nothing needs to be
deleted. If the forum allows editing, add a one-line bold pointer at the top of the original
post linking to this reply and to `results/CORRECTION-2026-08-02.md` in the repository.

---

**Correction: my decode-share numbers were wrong, and one of my conclusions was too**

Following up on my own post. I re-reviewed the overlap script, found three faults in how it
measured decode share during prefill, repaired it, and re-ran the full A/B. One published
conclusion does not survive.

**What was wrong with the instrument**

*1. SSE events were counted as tokens.* The script timestamped each streamed chunk with
non-empty content and computed `1 / mean(gap)` — an *event* rate — then divided it by a
*token* rate. On this speculative-decoding stack one chunk carries several accepted tokens.
Measured on the same server, 300 tokens arrived in 118 chunks:

```
tokens per chunk: {1: 31, 2: 36, 3: 25, 4: 14, 5: 6, 6: 6}
mean: 2.50 tokens per non-empty chunk
```

*2. There was no prefill window.* The line meant to select intervals during the prefill was
`during = [g for g in gaps]` — a no-op copy of the entire stream, including output before
the long request was submitted and after its prefill had finished. This turned out to be the
dominant error: it mixed the undisturbed head and tail of a 600-token stream into a
measurement of a 179-second prefill.

*3. The reference was cold and included TTFT.* It was the first request the script issued.
Warm, first-token-to-last-token, decode reads 21.3 tok/s where the old method read 17.1.

**The corrected numbers**

Three repetitions per profile, plus three more for 2048 after a server restart to rule out
warm-up effects. Each stream is now its own baseline — measured undisturbed for 25 s before
the prefill is submitted — and tokens are counted from streamed `token_ids`, cross-checked
against the server's `usage.completion_tokens`.

| | published | corrected |
|---|---|---|
| Decode share during prefill, 8192 | 7.1% | **1.7%** (1.6 / 1.7 / 1.8) |
| Decode share during prefill, 2048 | 7.3% | **5.0%** (4.5 / 4.8 / 5.0 / 5.0 / 5.1 / 5.2) |
| Prefill cost of 2048 | −3.9% | **−7.7%** (1,462 vs 1,584 tok/s) |
| p95 output-chunk gap, 8192 → 2048 | 5.20 → 1.59 s | 6.10 → 1.64 s |

The original three-run 8192 arm measured 1.6–1.8% decode share, with a 1.7% median. A
subsequent verification run using the fully hardened instrument measured 1.5%, with all
integrity guards passing. The six-run 2048 median is 5.0%, independently verified at 5.1%.
The multi-run A/B therefore remains **2.9×**, while the hardened verification pair measured
3.4×. Raw output for all eleven runs is committed in the repository.

**If you run this stack for agent workloads, this is the actionable part:**
`--max-num-batched-tokens` is a fairness lever, and 8192 — the value I originally
recommended in this thread, and a common default in recipes built on this runtime — costs
roughly 3× the decode throughput of 2048 while a long prefill is running. In absolute terms
that is ~0.6 tok/s against ~1.9 tok/s for whatever else is mid-generation. Both are bad;
the difference between them is not.

**The conclusion that does not survive.** I published that decode share is invariant to
chunk size, and concluded "chunk size is a jitter lever, not a fairness lever". That is
wrong. **2048 delivers 2.9× the decode share and 3.1× the decode tokens** of 8192 during the
same prefill. The fairness benefit I measured as zero is in fact the largest single effect
of the setting.

My reasoning at the time was: "at 8192 decode fits ~9 tokens per chunk, at 2048 ~2.3 — four
times more opportunities, four times fewer tokens each, net zero." Half of that was right.
Decode does get one scheduling opportunity per prefill chunk. But the step yields ~`k`
accepted speculative tokens **regardless of chunk size** — measured 3.3 per chunk at 8192
and 2.6 at 2048, not 9 and 2.3. So total decode scales with the *number* of chunks:

```
decode_tokens ≈ (prefill_tokens / chunk_size) × accepted_tokens_per_step
```

For the 262K prompt that is 32 chunks at 8192 against 128 at 2048. The model predicts 4×;
measured 3.1×, the shortfall explained by the higher per-step yield at the larger chunk.
This is falsifiable — it predicts decode share roughly proportional to `1/chunk_size` until
the per-step yield saturates at `k`.

Practical upshot for anyone running agents on this stack: `--max-num-batched-tokens 2048`
costs about 7.7% prefill throughput, nearly double what I first reported, and buys a 67%
larger KV pool, 73% less output jitter, **and** roughly 3× the decode throughput while a
long prefill is running. It is a better trade than my original write-up implied.

Worth noting the direction of the correction: the corrected decode shares are *lower* than
published — the starvation is worse than I reported — and the corrected prefill cost is
*higher*. Both sides of the trade-off got sharper.

**Unaffected**

- KV pool +67% (2,669,829 tokens on 2048, re-read from the restarted server's own startup
  accounting; 2,671,557 published, 0.06% apart)
- Admission delay: new short requests waited 145–159 s against a 1.66 s warm baseline,
  released only when the prefill completed — in **both** profiles. Chunk size does not fix
  admission; the difference between profiles is just how long the prefill takes.
- Long-context retrieval 12/12 at 32K/128K/512K/900K
- The GB10 UMA results and the #48140 reproduction
- The synthetic prefix-cache locality experiments on both stacks — plain-text prompts with
  no tools, so the chat-template fault below does not apply

**Also corrected: the agent-capture cache reuse.** That analysis did not apply the server's
chat template, and those bodies carry 20 tool definitions — the template expands one tool
definition to ~266 tokens where my flattening produced ~21. Re-measured against the real
template, reuse is **95.7 / 98.4 / 98.1%**, not 97–99%. I had predicted these could only
move *up*; wrong for the first transition, because my flattening also dropped serialized
assistant tool_calls, which sit in the appended tail being re-prefilled. The append-only
conclusion is unchanged and better supported: first divergence lands on the last token of
the previous prompt in every transition.

**What changed in the instrument**

Streamed `return_token_ids` with a hard abort when visible output carries no token_ids; every
stream cross-checked against the server's own `usage.completion_tokens`; an explicit prefill
window; each stream used as its own baseline; token throughput and output-chunk jitter
reported as separate metrics; output-chunk gaps clipped to the window rather than filtered
into it, so the stall that straddles the boundary is no longer discarded; all measurement
prompts salted, since this build exposes no `/reset_prefix_cache` (404). 22 regression tests
in CI, and a standalone `probe_token_ids.py` that answers "can this server's stream be
counted?" with an exit code:

```
non-empty content events : 95   <- what a naive stream counter would report
tokens via token_ids     : 200
usage.completion_tokens  : 200
VERDICT: Counting SSE events would understate decode by 2.11x on this server.
```

The strongest change is the smallest: an event count can never match the server's reported
`completion_tokens` on a speculative stack, so the original bug is now impossible to ship. A
written measurement rule is not a guard; an assertion is — I had the rule written down in my
own repository and violated it in one script anyway.

Everything is in the repository, including `results/CORRECTION-2026-08-02.md` which
enumerates exactly what changed and what did not.
