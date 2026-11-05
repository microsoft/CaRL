"""vLLM adapter: each segment is a new request; vLLM manages prefix reuse."""
import uuid
import os

from carl.core.prompt import render_initial_context
from carl.core.rollout import CODE_STOPS, ContextOverflow


class BoundaryEOS:
    def __init__(self, tokenizer, assimilate):
        self.tokenizer = tokenizer
        self.assimilate = assimilate
        self.matched = False

    def __call__(self, prompt_ids, generated_ids, logits):
        import torch
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        from carl.core.boundaries import python_boundary
        from carl.core.context_block_detector import detect_context_block
        self.matched = bool(detect_context_block(text) if self.assimilate else python_boundary(text))
        if self.matched:
            logits.fill_(-torch.inf)
            logits[self.tokenizer.eos_token_id] = 0
        return logits


class VLLMGenerator:
    def __init__(self, model, revision, max_model_len=8192):
        if os.environ.get("VLLM_USE_V1", "0") != "0":
            raise ValueError("This adapter requires the vLLM V0 async engine (VLLM_USE_V1=0)")
        os.environ["VLLM_USE_V1"] = "0"
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=model, revision=revision, tokenizer_revision=revision, trust_remote_code=False,
            dtype="bfloat16", max_model_len=max_model_len, enable_prefix_caching=True,
            tensor_parallel_size=1, enforce_eager=True))
        self.limit = max_model_len

    def render_prompt(self, question):
        return render_initial_context(self.tokenizer, question)

    async def __call__(self, prompt, stop, max_new_tokens):
        from vllm import SamplingParams
        if stop not in (CODE_STOPS, ["</context>"]):
            raise ValueError("Unknown stopping contract")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) + max_new_tokens > self.limit:
            raise ContextOverflow("Prompt plus generation exceeds vLLM context")
        processor = BoundaryEOS(self.tokenizer, stop == ["</context>"])
        params = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=max_new_tokens,
                                logits_processors=[processor])
        request = uuid.uuid4().hex
        completed = False
        try:
            result = None
            async for result in self.engine.generate({"prompt_token_ids": ids}, params, request):
                pass
            if result is None or not result.finished or len(result.outputs) != 1:
                raise RuntimeError("vLLM returned an incomplete generation")
            output = result.outputs[0]
            generated = list(output.token_ids)
            if processor.matched:
                if not generated or generated[-1] != self.tokenizer.eos_token_id:
                    raise RuntimeError("Native structural boundary did not end at control EOS")
                generated.pop()
                finish = "stop"
            else:
                finish = "eos" if output.finish_reason == "stop" else "length"
            completed = True
            return {"text": self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    "ids": generated, "prompt_ids": ids, "finish_reason": finish}
        finally:
            if not completed:
                await self.engine.abort(request)

    def close(self):
        self.engine.shutdown_background_loop()
