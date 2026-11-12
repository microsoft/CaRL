"""Offline generation regression tests."""
import pytest


def test_tiny_model_generation_preserves_prompt_tokens_and_training_mode():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from carl.core.rollout import CODE_STOPS
    from carl.inference.backend import HFGenerator

    class Inputs(dict):
        def to(self, device):
            return Inputs({k: v.to(device) for k, v in self.items()})

    class Tokenizer:
        chat_template = "test-template"
        pad_token_id = 0

        def __call__(self, prompt, **kwargs):
            assert kwargs["return_token_type_ids"] is False
            return Inputs(input_ids=torch.tensor([[3, 4]]), attention_mask=torch.ones(1, 2, dtype=torch.long))

        def decode(self, ids, **kwargs):
            return "".join(chr(65 + i) for i in ids if i not in (0, 1, 2))

    policy = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2,
        bos_token_id=1, eos_token_id=2, pad_token_id=0))
    policy.train()
    generate = HFGenerator(policy, Tokenizer())
    output = generate("test", CODE_STOPS, 2)
    assert policy.training
    assert output["prompt_ids"] == [3, 4]
    assert 1 <= len(output["ids"]) <= 2
    assert output["finish_reason"] in ("eos", "length")
    assert all(p.grad is None for p in policy.parameters())
    with pytest.raises(ValueError, match="exceeds context"):
        generate("test", CODE_STOPS, 32)
