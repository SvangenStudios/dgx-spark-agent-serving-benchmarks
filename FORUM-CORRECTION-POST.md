# Reply to post in the NVIDIA DGX Spark forum thread

Paste the block below as a reply. Keep the original post as it is — nothing needs to be
deleted or edited.

---

**Measurement correction — decode-share figures under revalidation**

Following up on my own post: I re-reviewed the overlap script and found three faults in how
the decode share during prefill was measured. The reported values of **7.1% and 7.3% are
invalid** and I am re-measuring them.

**1. SSE events were counted as tokens.** The script timestamped each streamed chunk with
non-empty content and computed `1 / mean(gap)` — an *event* rate — then divided it by a
*token* rate. On this speculative-decoding stack one chunk carries several accepted tokens.
Measured on the same server, 300 generated tokens arrived in 118 chunks:

```
tokens per chunk: {1: 31, 2: 36, 3: 25, 4: 14, 5: 6, 6: 6}
mean: 2.50 tokens per non-empty chunk
```

**2. There was no prefill window.** The line meant to select the intervals during the
prefill was `during = [g for g in gaps]` — a no-op copy of the entire stream, including the
5 s before the long request was submitted and everything after the prefill had finished.

**3. The reference was cold and included TTFT.** It was the first request the script issued,
measured as `completion_tokens ÷ total_request_time`. On the same server:

```
cold, total time incl. TTFT (the method used):  17.1 tok/s
warm, total time incl. TTFT:                    20.9 tok/s
warm, first token -> last token (pure decode):  21.3 tok/s
```

TTFT explains only ~1.9% of that. The rest is GPU clock ramp on a cold GB10.

These faults do not all point the same way — event-counting understates, the cold reference
and the missing window overstate — so I am not publishing a corrected number until it is
measured. It will remain far below the 25% threshold I used for a pass, so the qualitative
finding (decode is severely starved during long prefill) is not in question. The exact
percentage is.

**The comparison between the two chunk sizes is also withdrawn**, not just the absolute
values. My first instinct was that both runs carried the same bias so the comparison
survived. That does not hold: tokens per SSE chunk may differ between chunk sizes, the
unbounded window diluted each run by a different amount, and the two reference calls were
cold to different degrees.

The irony is not lost on me: my own measurement-discipline list said streamed output on this
build represents speculative decode *steps* rather than accepted tokens — and this script
did it anyway. A written rule is not a guard; a regression test is.

**Also corrected: the agent-capture cache reuse.** That analysis did not apply the server's
chat template, and those bodies carry 20 tool definitions — the template expands one tool
definition to ~266 tokens where my flattening produced ~21. I have now re-measured the
saved captures against the real template:

| Transition | Published | Corrected |
|---|---|---|
| turn 1→2 | 97.1% | 95.7% |
| turn 5→6 | 98.7% | 98.4% |
| turn 9→10 | 97.5% | 98.1% |

So the headline is **95.7–98.4%**, not 97–99%. Worth noting that I predicted these could
only move *up* — divergence sits at the last token, and reuse grows with prompt length. I
was wrong for turn 1→2: my flattening also dropped serialized `assistant` tool_calls, and
those sit in the appended tail being re-prefilled, which outweighed the longer prefix. The
append-only conclusion is unchanged and now better supported: the first divergence lands on
the last token of the previous prompt in every transition.

**Unaffected — measured without the event/token ratio:**

- KV pool +67% (1,598,763 → 2,671,557 tokens), from the runtime's own startup accounting
- visible output-chunk p95 gap 5.20 s → 1.59 s (−69%). The observed gaps closely tracked
  `chunk_size ÷ effective_prefill_rate` (predicted 5.36 s / 1.39 s) — wall-clock timing
- prefill throughput −3.9% (1,529 → 1,469 tok/s), from `usage.prompt_tokens`
- admission delay: new short requests at 367 s against a 1.66 s warm baseline, released only
  when the prefill completed
- prefill vs decode concurrency scaling, computed from `usage` token counts
- long-context retrieval 12/12 at 32K/128K/512K/900K
- the GB10 UMA results and the #48140 reproduction
- the synthetic prefix-cache locality experiments on both stacks — those send plain-text
  prompts with no tools, so the chat-template fault does not apply

**What I changed in the instrument:** streamed `return_token_ids: true` with a hard abort if
the server returns null instead of a silent fallback; tokens counted even when
`delta.content` is empty; the prefill window defined explicitly from submission of the long
request to its first output token; a warm, streamed, token-weighted reference measured first
token to last token; and token throughput reported separately from output-chunk jitter
rather than conflated. Every measurement prompt is now salted, not just the large prefill —
this build exposes no `/reset_prefix_cache` endpoint (404), so salting is the only reset
short of a restart. Regression tests cover all of it, and there is a standalone
`probe_token_ids.py` that answers "can this server's stream be counted?" with an exit code:

```
non-empty content events : 95   <- what a naive stream counter would report
tokens via token_ids     : 200
usage.completion_tokens  : 200
tokens per event         : 2.11
VERDICT: token_ids reported and consistent with usage.
         Counting SSE events would understate decode by 2.11x on this server.
```

The strongest change is the smallest one: every stream is now cross-checked against the
server's own `usage.completion_tokens`. An event count can never match that on a
speculative stack, so the original bug would have been impossible to ship. A written
measurement rule is not a guard; an assertion is.

Filing the scheduling issue is on hold until the minimal reproduction runs on the repaired
instrument. Corrected numbers and the updated scripts will go into v0.1.1 in the repo, and
I will post them in this thread.

Full write-up of exactly what is and is not affected:
`results/CORRECTION-2026-08-02.md` in the repository.
