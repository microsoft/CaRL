"""Single-sequence Transformers backend with native segment stopping."""
import math

from carl.core.boundaries import python_boundary
from carl.core.context_block_detector import detect_context_block
from carl.core.prompt import render_initial_context
from carl.core.rollout import CODE_STOPS, ContextOverflow


class SegmentBoundary:
    def __init__(self, tokenizer, prompt_length, assimilate):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.assimilate = assimilate
        self.matched = False

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError("This backend accepts one sequence per request")
        generated = self.tokenizer.decode(
            input_ids[0, self.prompt_length:].tolist(), skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        self.matched = (detect_context_block(generated) is not None if self.assimilate
                        else python_boundary(generated) is not None)
        return self.matched


class HFGenerator:
    def __init__(self, policy, tokenizer, temperature=0.0):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        if not tokenizer.chat_template or tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer requires a chat template and padding token")
        self.policy, self.tokenizer, self.temperature = policy, tokenizer, temperature
        self.policy_version = 0

    def render_prompt(self, question):
        return render_initial_context(self.tokenizer, question)

    def __call__(self, prompt, stop, max_new_tokens):
        import torch
        from transformers import GenerationConfig, StoppingCriteriaList

        if stop not in (CODE_STOPS, ["</context>"]):
            raise ValueError("Unsupported segment stopping contract")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("Generation budget must be a positive integer")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False,
                                add_special_tokens=False).to(next(self.policy.parameters()).device)
        length = inputs["input_ids"].shape[1]
        limit = self.policy.config.max_position_embeddings
        if not length or length + max_new_tokens > limit:
            raise ContextOverflow(f"Prompt ({length}) plus generation ({max_new_tokens}) exceeds context ({limit})")
        boundary = SegmentBoundary(self.tokenizer, length, stop == ["</context>"])
        config = GenerationConfig(
            do_sample=self.temperature > 0, max_new_tokens=max_new_tokens,
            num_beams=1, use_cache=True, eos_token_id=self.policy.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id, bos_token_id=self.policy.generation_config.bos_token_id)
        if self.temperature > 0:
            config.temperature, config.top_p, config.top_k = self.temperature, 1.0, 0
        was_training = self.policy.training
        self.policy.eval()
        try:
            with torch.no_grad():
                output = self.policy.generate(**inputs, generation_config=config,
                                              stopping_criteria=StoppingCriteriaList([boundary]))
        finally:
            self.policy.train(was_training)
        ids = output[0, length:].tolist()
        eos = config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos or []
        finish = "stop" if boundary.matched else "eos" if ids and ids[-1] in eos else "length"
        return {
            "text": self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "ids": ids, "prompt_ids": inputs["input_ids"][0].tolist(),
            "finish_reason": finish,
            "policy_version": self.policy_version, "log_probs_kind": None, "log_probs": [],
            "decoding_constraints": {"eos_token_ids": eos},
        }
