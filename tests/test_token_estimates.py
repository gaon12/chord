"""Routing includes tool traffic, multilingual content, and output allowance."""

from chord.token_estimates import estimate_request_tokens


def test_request_budget_includes_history_tool_calls_results_and_completion_cap():
    messages = [{"role": "user", "content": "hello"}]
    baseline = estimate_request_tokens(messages, None, 1024)
    assert estimate_request_tokens(messages, None, 2048) == baseline + 1024
    assert (
        estimate_request_tokens(
            messages + [{"role": "tool", "tool_call_id": "c1", "content": "결과" * 1000}],
            None,
            1024,
        )
        > baseline + 2000
    )
    assert (
        estimate_request_tokens(
            messages,
            [{"type": "function", "function": {"name": "lookup", "description": "x" * 2000}}],
            1024,
        )
        > baseline + 500
    )
