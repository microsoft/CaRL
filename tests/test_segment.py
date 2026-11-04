from dataclasses import asdict
import json

from carl.core.segment import Segment, Trajectory
from carl.core.tool import ToolResult


def test_segment_serialization_keeps_one_tool_result_and_debug_metadata():
    result = ToolResult("execution", "python_error", stdout="partial\n",
                        error="NameError: x is not defined")
    segment = Segment(
        "question", "```python\nprint(x)\n```", segment_type="invoke",
        termination="tool_call", tool_result=result, generation_finish_reason="stop",
        engine_request_id="request", engine_finish_reason="stop", engine_stop_reason=2,
        num_cached_tokens=16, value_estimate=0.25)
    record = json.loads(json.dumps(asdict(segment)))
    assert record["tool_result"] == asdict(result)
    assert not {"tool_output", "tool_comments", "value_target", "raw_generated_text"} & record.keys()
    assert record["generation_finish_reason"] == record["engine_finish_reason"] == "stop"
    assert record["engine_request_id"] == "request"
    assert record["engine_stop_reason"] == 2 and record["num_cached_tokens"] == 16
    assert record["value_estimate"] == 0.25
    restored = Segment(**{**record, "tool_result": ToolResult(**record["tool_result"])})
    assert restored == segment
    assert restored.tool_result.observation == "partial\n\nERROR [python_error]: NameError: x is not defined"


def test_tool_count_distinguishes_empty_success_failure_and_unexecuted_invocation():
    segments = [
        Segment("Q", "call", segment_type="invoke", tool_result=ToolResult("empty", "success")),
        Segment("Q", "call", segment_type="invoke",
                tool_result=ToolResult("failed", "python_error", error="NameError")),
        Segment("Q", "call", segment_type="invoke"),
    ]
    trajectory = Trajectory(segments=segments, termination_reason="segment_limit", reward=0)
    assert trajectory.total_tool_calls == 2
    assert trajectory.reward == 0
    assert trajectory.final_answer_text is None
