"""Regression tests for the four measurement faults found on 2026-08-02.

Each test corresponds to a fault that made it into published numbers. See
results/CORRECTION-2026-08-02.md.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import prefill_decode_overlap as ov  # noqa: E402


def chunk(content=None, token_ids=None):
    """A chat.completion.chunk as vLLM emits it."""
    return {"choices": [{"index": 0,
                         "delta": {"content": content} if content is not None else {},
                         "token_ids": token_ids}]}


# --- Fault 1: SSE events were counted as tokens -------------------------------

def test_chunk_carrying_four_tokens_counts_as_four():
    """One SSE event can carry several accepted tokens under speculative decoding.

    The published measurement counted this as 1, understating decode by ~2.5x.
    """
    assert ov.chunk_token_count(chunk(" the quick brown fox", [791, 4062, 14198, 39935])) == 4


# --- Fault 1b: tokens arriving without visible text ---------------------------

def test_chunk_with_empty_content_but_token_ids_is_still_counted():
    """vLLM may emit accepted tokens with no visible text delta. They are still tokens."""
    assert ov.chunk_token_count(chunk("", [791, 4062])) == 2


def test_chunk_with_visible_text_but_no_token_ids_is_refused():
    """No silent fallback to event counting — that is exactly how the bug shipped."""
    with pytest.raises(ov.MissingTokenIds):
        ov.chunk_token_count(chunk(" hello", None))


def test_terminal_chunk_carrying_no_tokens_is_not_an_error():
    """vLLM closes a stream with a finish_reason chunk that has no content and no
    token_ids. That is not a capability failure — it carries zero tokens."""
    final = {"choices": [{"index": 0, "delta": {}, "token_ids": None,
                          "finish_reason": "stop"}]}
    assert ov.chunk_token_count(final) == 0


# --- the guard that would have caught the original bug ------------------------

def test_counted_tokens_are_cross_checked_against_the_servers_own_usage():
    """The server reports completion_tokens. If our count disagrees, we are wrong."""
    samples = [(1.0, 2), (2.0, 3)]           # we counted 5
    ov.verify_against_usage(samples, {"completion_tokens": 5})   # agrees: no raise
    with pytest.raises(ov.TokenCountMismatch):
        ov.verify_against_usage(samples, {"completion_tokens": 12})


# --- Fault 2: there was no prefill window -------------------------------------

def test_samples_outside_the_prefill_window_are_excluded():
    """`during = [g for g in gaps]` included the whole stream: before, during and after."""
    samples = [(1.0, 3),    # before the long request was submitted
               (6.0, 1),    # inside
               (8.0, 1),    # inside
               (20.0, 4)]   # after the prefill completed
    assert ov.tokens_in_window(samples, 5.0, 10.0) == 2


def test_decode_rate_uses_only_the_window_duration():
    samples = [(1.0, 100), (6.0, 2), (8.0, 3), (20.0, 100)]
    # 5 tokens inside a 5-second window
    assert ov.decode_rate_in_window(samples, 5.0, 10.0) == pytest.approx(1.0)


# --- Fault 3: the reference included TTFT -------------------------------------

def test_reference_rate_measures_first_token_to_last_token_not_total_time():
    """Request sent at t=0, first token at t=10 (TTFT), then one token per second.

    Pure decode is 1.0 tok/s. Dividing 4 tokens by 13 s of wall clock gives 0.31.
    """
    samples = [(10.0, 1), (11.0, 1), (12.0, 1), (13.0, 1)]
    assert ov.reference_decode_rate(samples) == pytest.approx(1.0)


# --- Throughput and jitter must not be conflated ------------------------------

def test_chunk_gaps_are_independent_of_how_many_tokens_a_chunk_carries():
    """Jitter is a wall-clock property of chunks; throughput is a token property.

    Same timing, different token payloads -> identical gaps, different throughput.
    """
    sparse = [(0.0, 1), (1.0, 1), (2.0, 1)]
    dense = [(0.0, 5), (1.0, 5), (2.0, 5)]
    assert ov.chunk_gaps(sparse) == ov.chunk_gaps(dense) == [1.0, 1.0]
    assert ov.reference_decode_rate(dense) == pytest.approx(5.0)
    assert ov.reference_decode_rate(sparse) == pytest.approx(1.0)
