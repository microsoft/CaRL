import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from carl.core.rollout import CODE_STOPS
from carl.inference.batch import run_batch
from carl.inference.vllm import BoundaryEOS, VLLMGenerator
from carl.distributed import average_gradients


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return [ord(c) + 3 for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 3) for i in ids if i > 2)


def test_vllm_boundary_uses_only_generated_text_and_forces_control_eos():
    tokenizer = Tokenizer()
    processor = BoundaryEOS(tokenizer, False)
    logits = torch.ones(512)
    prompt = tokenizer.encode("```python\nprint(4)\n```")
    assert torch.equal(processor(prompt, [], logits.clone()), logits)
    generated = tokenizer.encode("```python\nprint(4)\n```")
    forced = processor([], generated, logits.clone())
    assert processor.matched and forced[2] == 0
    assert torch.isneginf(forced[:2]).all() and torch.isneginf(forced[3:]).all()


def test_vllm_record_does_not_supervise_runtime_eos(monkeypatch):
    module = ModuleType("vllm")
    module.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", module)
    tokenizer = Tokenizer()
    text = "```python\nprint(4)\n```"

    class Engine:
        async def generate(self, prompt, parameters, request):
            ids = tokenizer.encode(text)
            parameters.logits_processors[0](prompt["prompt_token_ids"], ids, torch.ones(512))
            yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(token_ids=ids + [2], finish_reason="stop")])

        async def abort(self, request):
            pytest.fail("A completed request must not be aborted")

    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator.tokenizer, generator.engine, generator.limit = tokenizer, Engine(), 8192
    result = asyncio.run(generator("Q", CODE_STOPS, 100))
    assert result["text"] == text and result["ids"] == tokenizer.encode(text)
    assert result["finish_reason"] == "stop"


def test_batch_deadline_cancels_inflight_generation():
    cancelled = []

    class Backend:
        def render_prompt(self, question):
            return question

        async def __call__(self, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

    saved = []
    async def persist(row, trajectory):
        saved.append(trajectory)

    async def unused(code):
        pytest.fail("No invocation")

    asyncio.run(run_batch([{"sample_id": "s", "question": "Q"}], Backend(), unused, persist, timeout=0.01))
    assert cancelled == [True] and saved[0].termination_reason == "timeout"


def test_gradient_reduction_includes_locally_unused_parameters(monkeypatch):
    import carl.distributed as distributed
    monkeypatch.setattr(distributed, "rank_world", lambda: (0, 2))
    calls = []

    def all_reduce(gradient):
        calls.append(gradient.clone())
        gradient.add_(2)

    monkeypatch.setattr(distributed.dist, "all_reduce", all_reduce)
    first, second = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    first.grad = torch.tensor([4.0])
    average_gradients([first, second])
    assert len(calls) == 2
    assert first.grad.item() == 3 and second.grad.item() == 1
