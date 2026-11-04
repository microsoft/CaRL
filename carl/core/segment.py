"""Segment & Trajectory dataclasses."""
from dataclasses import dataclass, field
from typing import Literal
from carl.core.tool import ToolResult


@dataclass
class Segment:
    """One option. Captured prompt_ids, not re-tokenized text, define replay input."""
    start_context: str
    generated_text: str
    generated_ids: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    segment_type: str = "synthesize"   # invoke | assimilate | synthesize
    termination: str = "eos"           # tool_call | context_block | eos | truncated | protocol_error
    tool_code: str | None = None
    advantage: float | None = None
    value_estimate: float | None = None
    prompt_ids: list[int] | None = None
    log_probs_kind: Literal["policy", "sampling"] | None = None
    generation_finish_reason: str | None = None
    policy_version: int | None = None
    engine_request_id: str | None = None
    engine_finish_reason: str | None = None
    engine_stop_reason: str | int | None = None
    num_cached_tokens: int | None = None
    tool_result: ToolResult | None = None
    decoding_constraints: dict | None = None


@dataclass
class Trajectory:
    segments: list[Segment] = field(default_factory=list)
    full_context: str = ""
    reward: float | None = None
    termination_reason: str | None = None
    failure_message: str | None = None

    @property
    def final_answer_text(self) -> str | None:
        if (self.termination_reason == "eos" and self.segments
                and self.segments[-1].segment_type == "synthesize"):
            return self.segments[-1].generated_text
        return None

    @property
    def total_tool_calls(self) -> int:
        return sum(1 for s in self.segments if s.segment_type == "invoke" and s.tool_result is not None)

    @property
    def total_assimilations(self) -> int:
        return sum(1 for s in self.segments if s.segment_type == "assimilate")

    @property
    def num_segments(self) -> int:
        return len(self.segments)
