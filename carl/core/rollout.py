"""Invoke, assimilate and commit transitions shared by collection and inference.

Generation returns text, ids, log_probs, and finish_reason (eos/stop/length).
Training also requires exact prompt_ids and log_probs_kind. No delimiter is
invented and no observation is removed by searching for its text.
"""
from __future__ import annotations
from typing import Callable

from carl.core.segment import Segment, Trajectory
from carl.core.tool import ToolResult
from carl.core.code_block_detector import detect_code_block
from carl.core.context_block_detector import detect_context_block
from carl.core.boundaries import python_boundary

GenFn = Callable[..., dict]
ExecFn = Callable[[str], ToolResult]


class ContextOverflow(ValueError):
    """The next model request cannot fit its configured context."""

# Backends must stop on complete generated blocks, not these strings alone.
CODE_STOPS = ["\n```\n", "\n```\r\n"]
PROTOCOL_VERSION = "python-context-eos-v1"
ASSIMILATION_INSTRUCTION = (
    "[END TOOL OUTPUT]\n"
    "[ASSIMILATE]\n"
    "Extract only the facts needed for the question or next step from the actual "
    "tool output above. Preserve relevant names, numbers, relationships and units. "
    "Write only one <context>...</context> block, with plain text inside. "
    "Do not continue the output list, call another tool, or give the final answer yet.\n"
)
CONTINUATION_INSTRUCTION = (
    "[CONTINUE]\n"
    "Use the retained context above. If another tool is necessary, write one Python "
    "block; otherwise commit the final answer inside \\boxed{} and end your response.\n"
)

CARL_SYSTEM_PROMPT = r"""You are a helpful assistant that answers questions, optionally using Python.

To use a tool, write a Python code block:

```python
print(search("query string"))
```

Use a language-labeled opening fence and a closing fence on its own line.
Stop after the code block; never invent its result. The runtime executes it.
Python calls are stateless: variables from a previous call do not exist.
Use standard-library Python and the provided search(query) function only.
Do not import requests, install packages, or implement your own web search.
search(query) returns Wikipedia passages as text; use print() to see results.
For arithmetic, use the given numbers. Financial tables are already in the question.

After EVERY tool result you MUST assimilate it before another call or an answer.
Write only one <context> block distilling the relevant facts from the real output:

<context>key fact 1; key fact 2</context>

Raw output is removed after extraction. Preserve intermediate facts and units
needed for later steps, not just a guessed answer. If output is empty or an error,
record that limitation without inventing facts.
After assimilation, either invoke another tool or COMMIT the final boxed answer
and end your response. You may commit directly without tools only before any
invocation or after completing its assimilation. Never skip assimilation.
"""


class RolloutController:
    """One trajectory; synchronous and asynchronous drivers share these transitions."""

    def __init__(self, question: str, max_segments: int = 15,
                 invoke_max_tokens: int = 2048, assimilate_max_tokens: int = 256,
                 system_prompt: str = CARL_SYSTEM_PROMPT,
                 initial_context: str | None = None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be nonempty")
        if any(type(n) is not int or n <= 0 for n in (
            max_segments, invoke_max_tokens, assimilate_max_tokens,
        )):
            raise ValueError("Rollout segment and token limits must be positive integers")
        if initial_context is not None and (not isinstance(initial_context, str) or not initial_context.strip()):
            raise ValueError("Rendered initial_context must be nonempty")
        self.trajectory = Trajectory(full_context=(
            initial_context if initial_context is not None
            else f"{system_prompt}\n\nQuestion: {question}\n\n"
        ))
        self.max_segments = max_segments
        self.invoke_max_tokens = invoke_max_tokens
        self.assimilate_max_tokens = assimilate_max_tokens
        self.state = "ready"
        self._tool_prefix: str | None = None
        self._tool_segment: Segment | None = None
        self._request_context: str | None = None

    @property
    def finished(self) -> bool:
        return self.state == "done"

    def generation_request(self) -> dict:
        if self.state != "ready":
            raise RuntimeError(f"Cannot request generation while rollout is {self.state}")
        self._request_context = self.trajectory.full_context
        self.state = "generating"
        assimilating = self._tool_prefix is not None
        return {
            "prompt": self._request_context,
            "stop": ["</context>"] if assimilating else list(CODE_STOPS),
            "max_new_tokens": self.assimilate_max_tokens if assimilating else self.invoke_max_tokens,
        }

    def accept_generation(self, out: dict) -> None:
        if self.state != "generating" or self._request_context is None:
            raise RuntimeError(f"Unexpected generation while rollout is {self.state}")
        finish = out.get("finish_reason")
        if finish not in ("eos", "stop", "length") or not isinstance(out.get("text"), str):
            raise ValueError("Generator must return text and finish_reason=eos/stop/length")
        text = out["text"]
        assimilating = self._tool_prefix is not None
        segment = Segment(
            start_context=self._request_context,
            generated_text=text,
            generated_ids=list(out.get("ids", [])),
            log_probs=list(out.get("log_probs", [])),
            prompt_ids=list(out["prompt_ids"]) if out.get("prompt_ids") is not None else None,
            log_probs_kind=out.get("log_probs_kind"),
            generation_finish_reason=finish,
            policy_version=out.get("policy_version"),
            engine_request_id=out.get("request_id"),
            engine_finish_reason=out.get("engine_finish_reason"),
            engine_stop_reason=out.get("engine_stop_reason"),
            num_cached_tokens=out.get("num_cached_tokens"),
            decoding_constraints=out.get("decoding_constraints"),
            segment_type="assimilate" if assimilating else "synthesize",
        )
        self.trajectory.segments.append(segment)
        if finish == "length":
            segment.termination = "truncated"
            self.trajectory.full_context += text
            self._finish("token_limit")
            return

        if assimilating:
            detected = detect_context_block(text)
            if (detected is None or text[:detected.start].strip()
                    or text[detected.end:].strip() or text.count("<context>") != 1):
                self._protocol_error(segment, "Expected exactly one complete <context> block with no surrounding output")
                return
            segment.termination = "context_block"
            self.trajectory.full_context = self._tool_prefix + text + "\n" + CONTINUATION_INSTRUCTION
            self._tool_prefix = None
            self._continue_or_finish()
            return

        detected = detect_code_block(text)
        if detected is not None:
            if (python_boundary(text) is None
                    or text[detected.end:].strip() or not detected.executable):
                self._protocol_error(segment, "Expected one nonempty Python block ending at its closing fence")
                return
            segment.segment_type = "invoke"
            segment.termination = "tool_call"
            segment.tool_code = detected.executable
            self.trajectory.full_context += text + "\n"
            if self.trajectory.num_segments >= self.max_segments:
                self._finish("segment_limit")
                return
            self._tool_segment = segment
            self.state = "tool"
        elif (finish == "eos" and text.strip() and "```" not in text
              and "<context>" not in text and "</context>" not in text):
            segment.termination = "eos"
            self.trajectory.full_context += text
            self._finish("eos")
        else:
            self._protocol_error(segment, "Expected a complete Python block or a nonempty final answer terminated by EOS")

    def tool_request(self) -> str:
        if (self.state != "tool" or self._tool_segment is None
                or self._tool_segment.tool_code is None):
            raise RuntimeError("There is no pending tool invocation")
        return self._tool_segment.tool_code

    def accept_tool_result(self, result: ToolResult) -> None:
        if self.state != "tool" or self._tool_segment is None:
            raise RuntimeError("Unexpected or duplicate tool result")
        if not isinstance(result, ToolResult):
            raise TypeError("Executor must return a ToolResult")
        self._tool_segment.tool_result = result
        self._tool_segment = None
        self._tool_prefix = self.trajectory.full_context
        self.trajectory.full_context += "[TOOL OUTPUT]\n" + result.observation + "\n" + ASSIMILATION_INSTRUCTION
        self.state = "ready"

    def _protocol_error(self, segment: Segment, message: str) -> None:
        segment.termination = "protocol_error"
        self.trajectory.full_context += segment.generated_text
        self.trajectory.failure_message = message
        self._finish("protocol_error")

    def _continue_or_finish(self) -> None:
        if self.trajectory.num_segments >= self.max_segments:
            self._finish("segment_limit")
        else:
            self.state = "ready"

    def _finish(self, reason: str) -> None:
        self.trajectory.termination_reason = reason
        self.state = "done"

    def terminate(self, reason: str, message: str) -> None:
        """Record a driver-imposed terminal limit without inventing model output."""
        if self.finished:
            raise RuntimeError("Cannot overwrite a completed trajectory")
        if not reason or not message:
            raise ValueError("External termination requires a reason and diagnostic message")
        self.trajectory.failure_message = message
        self._finish(reason)


def rollout(
    question: str,
    generate: GenFn,
    execute: ExecFn,
    max_segments: int = 15,
    invoke_max_tokens: int = 2048,
    assimilate_max_tokens: int = 256,
    synthesize_max_tokens: int | None = None,
    system_prompt: str = CARL_SYSTEM_PROMPT,
    initial_context: str | None = None,
) -> Trajectory:
    """Drive the shared controller; failures in backend contracts propagate.

    Invoke versus commit is unknown before generation, so both use the same
    decision budget. If supplied, the old synthesize argument must agree.
    """
    if synthesize_max_tokens is not None and synthesize_max_tokens != invoke_max_tokens:
        raise ValueError("Invoke and commit share a decision budget; their token limits must match")
    controller = RolloutController(
        question, max_segments, invoke_max_tokens, assimilate_max_tokens, system_prompt,
        initial_context,
    )
    while not controller.finished:
        if controller.state == "tool":
            controller.accept_tool_result(execute(controller.tool_request()))
        else:
            try:
                controller.accept_generation(generate(**controller.generation_request()))
            except ContextOverflow as error:
                controller.terminate("context_limit", str(error))
    return controller.trajectory
