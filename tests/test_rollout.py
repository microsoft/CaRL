import pytest

from carl.core.rollout import RolloutController, rollout
from carl.core.tool import ToolResult


def responses(*outputs):
    outputs = iter(outputs)

    def generate(**request):
        text, finish = next(outputs)
        return {"text": text, "finish_reason": finish,
                "ids": [ord(c) for c in text],
                "prompt_ids": [ord(c) for c in request["prompt"]]}
    return generate


def test_direct_commit_does_not_invoke_tools():
    def forbidden(code):
        pytest.fail("A direct answer must not execute a tool")

    tr = rollout("What is two plus two?", responses((r"\boxed{4}", "eos")), forbidden)
    assert tr.final_answer_text == r"\boxed{4}"
    assert tr.num_segments == 1
    assert tr.total_tool_calls == 0


def test_two_calls_keep_extractions_and_remove_only_owned_observations():
    calls = []

    def execute(code):
        calls.append(code)
        return ToolResult(str(len(calls)), "success", f"RAW_{len(calls)}")

    tr = rollout("The question itself mentions RAW_1.", responses(
        ("```python\nprint(2)\n```", "stop"), ("<context>first fact</context>", "stop"),
        ("```python\nprint(3)\n```", "stop"), ("<context>second fact</context>", "stop"),
        (r"\boxed{5}", "eos")), execute)
    assert len(calls) == 2 and tr.num_segments == 5
    assert "RAW_1" in tr.segments[1].start_context
    assert "RAW_2" in tr.segments[3].start_context
    assert "RAW_2" not in tr.full_context
    assert tr.full_context.count("RAW_1") == 1
    assert "first fact" in tr.full_context and "second fact" in tr.full_context
    assert "[TOOL OUTPUT]" not in tr.full_context
    assert tr.segments[0].prompt_ids == [ord(c) for c in tr.segments[0].start_context]


def test_invocation_comments_are_preserved_for_execution_context_and_training():
    code = "# Compute the product from the question\nprint(347 * 28)"
    invocation = f"I will multiply the two numbers.\n```python\n{code}\n```"
    calls = []

    def execute(actual):
        calls.append(actual)
        return ToolResult("product", "success", "9716\n")

    tr = rollout("What is 347 times 28?", responses(
        (invocation, "stop"), ("<context>The product is 9716.</context>", "stop"),
        (r"\boxed{9716}", "eos")), execute)
    assert calls == [code]
    assert tr.segments[0].generated_text == invocation
    assert tr.segments[0].generated_ids == [ord(c) for c in invocation]
    assert invocation in tr.segments[1].start_context
    assert invocation in tr.segments[2].start_context
    assert tr.final_answer_text == r"\boxed{9716}"


@pytest.mark.parametrize("text,finish,reason", [
    ("```python\nprint(4)", "eos", "protocol_error"),
    ("```python\nprint(4)", "length", "token_limit"),
    ("```python\nprint(4)\n```\nThe answer is 4.", "eos", "protocol_error"),
    ("```javascript\nprint(4)\n```", "stop", "protocol_error"),
    ("```\nprint(4)\n```", "stop", "protocol_error"),
    ("```python\n# comment only\n```", "stop", "protocol_error"),
    ("<context>4</context>", "eos", "protocol_error"),
    ("", "eos", "protocol_error"),
    ("Unterminated answer", "stop", "protocol_error"),
])
def test_invalid_generation_is_not_repaired_or_executed(text, finish, reason):
    tr = rollout("Q", responses((text, finish)), lambda code: pytest.fail("Unexpected execution"))
    assert tr.termination_reason == reason
    assert tr.final_answer_text is None
    assert tr.segments[0].generated_text == text


@pytest.mark.parametrize("text", [
    "<context>unfinished", "<context>a</context><context>b</context>",
    "Explanation <context>a</context>", "<context>a</context>Answer: a",
])
def test_invalid_assimilation_is_not_silently_closed(text):
    tr = rollout("Q", responses(("```python\nprint(1)\n```", "stop"), (text, "stop")),
                 lambda code: ToolResult("test", "success", "raw"))
    assert tr.termination_reason == "protocol_error"
    assert tr.segments[-1].generated_text == text


@pytest.mark.parametrize("result", [
    ToolResult("empty", "success"),
    ToolResult("error", "python_error", error="NameError: x is not defined"),
])
def test_empty_and_failed_tools_still_require_assimilation(result):
    tr = rollout("Q", responses(
        ("```python\nprint(x)\n```", "stop"), ("<context>No new information.</context>", "stop"),
        ("Insufficient information.", "eos")), lambda code: result)
    assert result.observation in tr.segments[1].start_context
    assert tr.total_assimilations == 1
    assert tr.segments[0].tool_result == result
    assert "[TOOL OUTPUT]" not in tr.segments[-1].start_context


def test_segment_limit_does_not_run_an_unassimilatable_call():
    tr = rollout("Q", responses(("```python\nprint(1)\n```", "stop")),
                 lambda code: pytest.fail("No space for assimilation"), max_segments=1)
    assert tr.termination_reason == "segment_limit" and tr.total_tool_calls == 0


def test_invalid_state_order_and_empty_question():
    with pytest.raises(ValueError, match="Question"):
        RolloutController("")
    c = RolloutController("Q")
    with pytest.raises(RuntimeError):
        c.accept_tool_result("unexpected")
    c.generation_request()
    with pytest.raises(RuntimeError):
        c.generation_request()
    with pytest.raises(ValueError):
        c.accept_generation({"text": "Missing termination metadata"})


def test_context_overflow_becomes_an_explicit_terminal_failure():
    from carl.core.rollout import ContextOverflow

    def generate(**kwargs):
        raise ContextOverflow("No space for the next request")

    tr = rollout("Q", generate, lambda code: pytest.fail("No invocation"))
    assert tr.termination_reason == "context_limit"
    assert tr.final_answer_text is None and tr.failure_message


def test_unstructured_tool_result_is_rejected_without_consuming_the_invocation():
    controller = RolloutController("Q")
    controller.generation_request()
    controller.accept_generation({"text": "```python\nprint(4)\n```", "finish_reason": "stop",
                                  "request_id": "request", "engine_finish_reason": "stop",
                                  "engine_stop_reason": 2, "num_cached_tokens": 16})
    context = controller.trajectory.full_context
    with pytest.raises(TypeError, match="ToolResult"):
        controller.accept_tool_result("4")
    assert controller.state == "tool" and controller.trajectory.full_context == context
    assert controller.trajectory.segments[0].tool_result is None
    result = ToolResult("actual", "success", "4\n")
    controller.accept_tool_result(result)
    assert controller.trajectory.total_tool_calls == 1
    assert result.observation in controller.trajectory.full_context
    segment = controller.trajectory.segments[0]
    assert segment.generation_finish_reason == "stop" and segment.engine_finish_reason == "stop"
    assert segment.engine_request_id == "request" and segment.engine_stop_reason == 2
    assert segment.num_cached_tokens == 16
    with pytest.raises(RuntimeError, match="duplicate"):
        controller.accept_tool_result(result)
