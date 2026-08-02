"""Regression tests for fault 4: prompt_locality did not use the server's chat template.

The old flatten() built its own "<|role|>content" string. On an agent body with one tool
definition that produced 93 tokens where the server's real template produces 338, and it
dropped assistant tool_calls entirely. See results/CORRECTION-2026-08-02.md.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import prompt_locality as pl  # noqa: E402


TOOLS = [{"type": "function",
          "function": {"name": "read_file", "description": "Read a file",
                       "parameters": {"type": "object",
                                      "properties": {"path": {"type": "string"}}}}}]

BODY = {"messages": [{"role": "system", "content": "You are a helpful agent."},
                     {"role": "user", "content": "Read config.yaml"},
                     {"role": "assistant", "content": None,
                      "tool_calls": [{"id": "c1", "type": "function",
                                      "function": {"name": "read_file",
                                                   "arguments": '{"path": "config.yaml"}'}}]},
                     {"role": "tool", "tool_call_id": "c1", "content": "key: value"}],
        "tools": TOOLS}


def test_chat_body_is_tokenized_as_messages_not_as_flattened_text():
    """The server must apply its own chat template; we must not invent one."""
    req = pl.tokenize_request(BODY)
    assert req["messages"] == BODY["messages"]
    assert "prompt" not in req


def test_tool_definitions_are_forwarded_to_the_server():
    """The template expands one tool definition to ~266 tokens; flatten() produced ~21."""
    assert pl.tokenize_request(BODY)["tools"] == TOOLS


def test_generation_prompt_is_included():
    """The generation prompt is part of what gets prefilled, so it must be counted."""
    assert pl.tokenize_request(BODY)["add_generation_prompt"] is True


def test_plain_text_is_still_sent_as_a_prompt():
    """Plain-text mode is unaffected by the chat template and must keep working."""
    req = pl.tokenize_request("just some text")
    assert req["prompt"] == "just some text"
    assert "messages" not in req


# --- approximate mode is opt-in, and must not silently lose tool calls ---

def test_approximate_mode_is_only_reachable_when_explicitly_requested():
    strict = pl.tokenize_request(BODY)
    approx = pl.tokenize_request(BODY, approximate=True)
    assert "messages" in strict and "prompt" not in strict
    assert "prompt" in approx and "messages" not in approx


def test_approximate_flattening_does_not_drop_assistant_tool_calls():
    """The old flatten() emitted an empty string for content=None tool_call messages,
    so a divergence inside a tool call was invisible.

    The marker appears ONLY inside the tool call, so this cannot pass by picking it up
    from the surrounding user or tool messages.
    """
    body = {"messages": [{"role": "user", "content": "Read the config"},
                         {"role": "assistant", "content": None,
                          "tool_calls": [{"id": "c1", "type": "function",
                                          "function": {"name": "read_file",
                                                       "arguments": '{"path": "ONLY-IN-TOOLCALL"}'}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": "key: value"}]}
    assert "ONLY-IN-TOOLCALL" in pl.flatten(body)


def test_unsupported_server_aborts_instead_of_approximating():
    """No silent fallback — that is how the first fault shipped."""
    with pytest.raises(SystemExit):
        pl.require_chat_tokenize({"__http": 400, "__body": "unknown field: messages"})
