"""Exact segment replay and framework-independent clipped actor objective.

Sequences are left padded; generated actions are right aligned in the action
axis. Prompt/observation tokens are attention context, never direct targets.
Critic boundary losses, reference KL and framework wiring are separate concerns.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from carl.core.segment import Segment


@dataclass
class SegmentReplay:
    sequences: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    labels: Tensor
    action_mask: Tensor
    old_log_probs: Tensor
    advantages: Tensor


def pack_segments(segments: list[Segment], pad_token_id: int) -> SegmentReplay:
    """Pack recorded inputs/actions without reconstructing or truncating either.

    Log probabilities must be finite raw-policy scores captured/recomputed
    under the frozen rollout policy. Unknown and sampling-transformed scores
    are rejected rather than mixed with raw training logits.
    """
    if not segments:
        raise ValueError("Segment replay requires at least one segment")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    for i, segment in enumerate(segments):
        if not segment.prompt_ids:
            raise ValueError(f"Segment {i} requires recorded prompt_ids; do not retokenize context")
        if not segment.generated_ids:
            raise ValueError(f"Segment {i} has no generated actions")
        for ids in (segment.prompt_ids, segment.generated_ids):
            if any(type(token) is not int or token < 0 for token in ids):
                raise ValueError(f"Segment {i} token IDs must be nonnegative integers")
        if segment.log_probs_kind != "policy":
            raise ValueError(
                f"Segment {i} requires log_probs_kind='policy'; recompute raw-policy "
                "scores with the frozen rollout weights before PPO replay"
            )
        if len(segment.log_probs) != len(segment.generated_ids):
            raise ValueError(f"Segment {i} needs one old log probability per generated action")
        if any(not math.isfinite(p) or p > 0 for p in segment.log_probs):
            raise ValueError(f"Segment {i} old log probabilities must be finite and nonpositive")
        if segment.advantage is None or not math.isfinite(segment.advantage):
            raise ValueError(f"Segment {i} requires a finite advantage")
        if (any(abs(p) > torch.finfo(torch.float32).max for p in segment.log_probs)
                or abs(segment.advantage) > torch.finfo(torch.float32).max):
            raise ValueError(f"Segment {i} scores and advantage must be representable in float32")

    batch_size = len(segments)
    length = max(len(s.prompt_ids) + len(s.generated_ids) for s in segments)
    n_actions = max(len(s.generated_ids) for s in segments)
    sequences = torch.full((batch_size, length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(sequences)
    labels = torch.full_like(sequences, -100)
    action_mask = torch.zeros((batch_size, n_actions), dtype=torch.bool)
    old_log_probs = torch.zeros((batch_size, n_actions), dtype=torch.float32)
    advantages = torch.zeros_like(old_log_probs)
    for i, segment in enumerate(segments):
        tokens = segment.prompt_ids + segment.generated_ids
        count = len(segment.generated_ids)
        sequences[i, -len(tokens):] = torch.tensor(tokens)
        attention_mask[i, -len(tokens):] = 1
        labels[i, -count:] = torch.tensor(segment.generated_ids)
        action_mask[i, -count:] = True
        old_log_probs[i, -count:] = torch.tensor(segment.log_probs)
        advantages[i, -count:] = segment.advantage
    position_ids = (attention_mask.cumsum(dim=1) - 1).masked_fill(attention_mask == 0, 0)
    return SegmentReplay(
        sequences, attention_mask, position_ids, labels,
        action_mask, old_log_probs, advantages,
    )


def action_log_probs(logits: Tensor, replay: SegmentReplay) -> Tensor:
    """Score every generated token, including the first, at its causal position."""
    if logits.ndim != 3 or logits.shape[:2] != replay.sequences.shape:
        raise ValueError("Logits must have shape (batch, sequence, vocabulary) matching replay")
    n_actions = replay.action_mask.size(1)
    action_logits = logits[:, -(n_actions + 1):-1].float()
    action_logits = action_logits.masked_fill(~replay.action_mask.unsqueeze(-1), 0)
    targets = replay.sequences[:, -n_actions:].masked_fill(~replay.action_mask, 0)
    if (targets >= logits.size(-1)).any():
        raise ValueError("Generated action token is outside the logits vocabulary")
    scores = action_logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return scores.masked_fill(~replay.action_mask, 0)


def clipped_policy_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    action_mask: Tensor,
    clip_range: float = 0.2,
) -> Tensor:
    """Mean of per-segment token-mean PPO losses, with fixed old scores/advantages.

    Only active action positions participate, even if ignored padding values
    are nonfinite. This reduction gives each segment equal weight.
    """
    if (new_log_probs.ndim != 2 or new_log_probs.shape != old_log_probs.shape
            or new_log_probs.shape != advantages.shape
            or new_log_probs.shape != action_mask.shape):
        raise ValueError("PPO scores, advantages and mask must have matching (batch, actions) shapes")
    if action_mask.dtype != torch.bool:
        raise ValueError("PPO action_mask must be boolean")
    if action_mask.size(0) == 0 or not action_mask.any(dim=1).all():
        raise ValueError("Every PPO segment must contain at least one action")
    if not 0 < clip_range < 1:
        raise ValueError("clip_range must be between 0 and 1")
    for values in (new_log_probs, old_log_probs, advantages):
        if not torch.isfinite(values[action_mask]).all():
            raise ValueError("PPO action scores and advantages must be finite")
    new_scores = new_log_probs.masked_fill(~action_mask, 0)
    old_scores = old_log_probs.detach().masked_fill(~action_mask, 0)
    fixed_advantages = advantages.detach().masked_fill(~action_mask, 0)
    ratios = (new_scores - old_scores).exp()
    if not torch.isfinite(ratios[action_mask]).all():
        raise ValueError("PPO probability ratio overflow; inspect policy/log-probability consistency")
    surrogate = torch.minimum(
        ratios * fixed_advantages,
        ratios.clamp(1 - clip_range, 1 + clip_range) * fixed_advantages,
    )
    losses = -surrogate.sum(dim=1) / action_mask.sum(dim=1)
    loss = losses.mean()
    if not torch.isfinite(loss):
        raise ValueError("PPO loss overflow; inspect action scores and advantage magnitudes")
    return loss
