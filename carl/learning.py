"""Model loading, causal supervision and local checkpoint utilities."""
from contextlib import nullcontext
from pathlib import Path

import torch


def load_model(path, revision, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not Path(path).is_dir() and not revision:
        raise ValueError("Remote models require an explicit commit revision")
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, revision=revision, trust_remote_code=False, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    return model, tokenizer


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def target_tokens(segment, tokenizer):
    prompt = segment["prompt_ids"]
    action = list(segment["generated_ids"])
    if not prompt or not action:
        raise ValueError("Training requires captured prompt and action token IDs")
    forced = (segment.get("decoding_constraints") or {}).get("forced_boundary_eos")
    if forced:
        if segment["segment_type"] != "invoke" or action[-1] != tokenizer.eos_token_id:
            raise ValueError("Invalid runtime boundary EOS metadata")
        action.pop()
    eos = (segment.get("decoding_constraints") or {}).get("eos_token_ids", [tokenizer.eos_token_id])
    if segment["segment_type"] == "synthesize" and action[-1] not in eos:
        raise ValueError("Final commit must contain a natural EOS target")
    if not action:
        raise ValueError("Empty action target")
    return prompt, action


def sft_loss(policy, reference, prompt, action, coefficient=0.0):
    """Only action tokens are targets; optional exact KL(reference || policy)."""
    device = next(policy.parameters()).device
    if coefficient < 0 or coefficient and reference is None:
        raise ValueError("Nonnegative reference regularization requires a frozen model")
    ids = torch.tensor([prompt + action], device=device)
    if ids.shape[1] > policy.config.max_position_embeddings:
        raise ValueError("Training context exceeds the model limit; no truncation")
    logits = policy(ids, use_cache=False).logits[:, len(prompt)-1:-1].float()
    targets = torch.tensor([action], device=device)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.flatten())
    if coefficient:
        if reference.training or any(p.requires_grad for p in reference.parameters()):
            raise ValueError("The reference must be immutable and in eval mode")
        with torch.no_grad():
            ref = reference(ids, use_cache=False).logits[:, len(prompt)-1:-1].float().log_softmax(-1)
        loss = loss + coefficient * (ref.exp() * (ref - logits.log_softmax(-1))).sum(-1).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite SFT objective")
    return loss
