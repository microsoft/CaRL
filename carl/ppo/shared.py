"""Shared-backbone CARL PPO with synchronous distributed optimizer updates.
Actor and critic gradients accumulate at the same weights before one Adam step.
Shared parameters have one optimizer state and one learning rate.
The reference policy is separate and frozen. Collection uses the policy object
directly; no external rollout engine can retain stale weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
import random

import torch
from torch import nn

from carl.core.segment import Segment, Trajectory
from carl.critic.head import ValueModel
from carl.ppo.advantages import segment_advantages
from carl.ppo.replay import action_log_probs, clipped_policy_loss, pack_segments
from carl.critic.validation import passes_calibration_gate
from carl.distributed import average_gradients, rank_world


@dataclass(frozen=True)
class PPOBatch:
    trajectories: tuple[Trajectory, ...]
    policy_version: int


def _to_device(replay, device):
    for name, value in vars(replay).items():
        setattr(replay, name, value.to(device))
    return replay


class SharedPPOTrainer:
    """Collect -> freeze old scores/values -> update -> collect with new weights."""

    def __init__(self, policy, reference, tokenizer, cfg, value_head=None):
        self.cfg = cfg
        if cfg.method not in ("carl", "trajectory_ppo", "grpo"):
            raise ValueError("Method must be carl, trajectory_ppo or grpo")
        for name in ("ppo_epochs", "ppo_mini_batch", "rollout_batch_prompts",
                     "n_samples_per_prompt", "num_steps", "eval_every"):
            if type(getattr(cfg, name)) is not int or getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("actor_lr", "critic_value_head_lr", "grad_clip"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(cfg.warmup_steps) is not int or cfg.warmup_steps < 1:
            raise ValueError("warmup_steps must be a positive integer")
        for name in ("vf_coef", "kl_coef", "ent_coef"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if cfg.gamma != 1 or cfg.gae_lambda != 0 or not 0 < cfg.eps_clip < 1:
            raise ValueError("CARL requires gamma=1, gae_lambda=0 and 0 < eps_clip < 1")
        if cfg.invoke_max_tokens != cfg.synthesize_max_tokens:
            raise ValueError("Invoke and commit use the same generation budget")
        if tokenizer.pad_token_id is None:
            raise ValueError("A padding token ID is required")
        policy_ids = {id(p) for p in policy.parameters()}
        if policy_ids & {id(p) for p in reference.parameters()}:
            raise ValueError("Reference weights must not alias the trainable policy")
        self.policy = policy
        self.reference = reference.requires_grad_(False).eval()
        self.critic = ValueModel(policy.base_model, value_head)
        self.tokenizer = tokenizer
        self.device = next(policy.parameters()).device
        if next(reference.parameters()).device != self.device:
            raise ValueError("Policy and reference must be on the same device")
        self.policy.eval()
        self.critic.eval()
        if policy_ids & {id(p) for p in self.critic.head.parameters()}:
            raise ValueError("The value head must not alias policy parameters")
        self.optimizer = torch.optim.AdamW([
            {"params": self.policy.parameters(),
             "lr": cfg.actor_lr, "weight_decay": 0.01},
            {"params": self.critic.head.parameters(),
             "lr": cfg.critic_value_head_lr, "weight_decay": 0.0},
        ])
        self._parameters = tuple(p for group in self.optimizer.param_groups for p in group["params"])
        self.policy_version = 0
        self._rng = random.Random(cfg.seed + rank_world()[0])
        self._pending_batch: PPOBatch | None = None
        self._failed = False
        self._updates = 0

    def load_warmup(self, checkpoint):
        """Load both head and shared backbone, never a head over unrelated weights."""
        if self.policy_version or self._pending_batch is not None or self._failed:
            raise RuntimeError("Warm-up can only be loaded before collecting/updating")
        metrics = checkpoint.get("metrics", {})
        warmup_cfg = checkpoint.get("cfg", {})
        if not {"auc_gate", "ev_gate", "sign_gate"} <= warmup_cfg.keys():
            raise ValueError("Warm-up checkpoint must record its calibration thresholds")
        if (metrics.get("gate_scope") != "boundary_calibration"
                or not metrics.get("passed_gate")
                or not passes_calibration_gate(
                    metrics, warmup_cfg["auc_gate"], warmup_cfg["ev_gate"], warmup_cfg["sign_gate"],
                )):
            raise ValueError("Warm-up checkpoint has not passed boundary calibration")
        if warmup_cfg.get("model_name") != self.cfg.actor_model:
            raise ValueError("Warm-up model identity must match actor_model")
        self.critic.backbone.load_state_dict(checkpoint["backbone"], strict=True)
        self.critic.head.load_state_dict(checkpoint["head"], strict=True)

    @torch.no_grad()
    def prepare_batch(self, trajectories: list[Trajectory]) -> PPOBatch:
        """Snapshot actual on-policy inputs before any optimizer update.

        Caller must supply rollouts from this trainer's current policy version.
        The integrated collect method guarantees that ownership. Raw probabilities
        are rescored here, so processed generation scores are never PPO denominators.
        """
        if self._failed:
            raise RuntimeError("Trainer update failed; reload a checkpoint before continuing")
        if self._pending_batch is not None:
            raise RuntimeError("Consume the pending PPO batch before preparing another")
        if not trajectories:
            raise ValueError("PPO requires a nonempty rollout batch")
        self.policy.eval()
        self.critic.eval()
        snapshots = deepcopy(trajectories)
        for trajectory in snapshots:
            if not trajectory.segments or trajectory.reward not in (0, 1):
                raise ValueError("Each trajectory needs segments and a binary terminal reward")
            if trajectory.termination_reason not in ("eos", "protocol_error", "token_limit", "segment_limit", "context_limit"):
                raise ValueError("Cannot train from an unterminated trajectory")
            if trajectory.final_answer_text is None and trajectory.reward != 0:
                raise ValueError("An unfinished/failed trajectory cannot have positive reward")
            values = []
            for segment in trajectory.segments:
                if segment.policy_version != self.policy_version:
                    raise ValueError("Stale or unknown rollout policy_version")
                segment.advantage = 0.0
                segment.log_probs_kind = "policy"
                segment.log_probs = [0.0] * len(segment.generated_ids)
                replay = _to_device(pack_segments([segment], self.tokenizer.pad_token_id), self.device)
                logits = self.policy(
                    replay.sequences, attention_mask=replay.attention_mask,
                    position_ids=replay.position_ids, use_cache=False,
                ).logits
                segment.log_probs = action_log_probs(logits, replay)[0].cpu().tolist()
                ids = torch.tensor([segment.prompt_ids], device=self.device)
                value = float(self.critic(ids, torch.ones_like(ids)).item())
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Boundary critic must produce a finite probability")
                segment.value_estimate = value
                values.append(value)
            advantages = (segment_advantages(values, trajectory.reward) if self.cfg.method == "carl"
                          else [trajectory.reward - values[0]] * len(values))
            for segment, advantage in zip(trajectory.segments, advantages):
                segment.advantage = advantage
        batch = PPOBatch(tuple(snapshots), self.policy_version)
        self._pending_batch = batch
        return batch

    def _segment_actor_loss(self, segment):
        replay = _to_device(pack_segments([segment], self.tokenizer.pad_token_id), self.device)
        logits = self.policy(
            replay.sequences, attention_mask=replay.attention_mask,
            position_ids=replay.position_ids, use_cache=False,
        ).logits
        current = action_log_probs(logits, replay)
        actor_loss = clipped_policy_loss(
            current, replay.old_log_probs, replay.advantages,
            replay.action_mask, self.cfg.eps_clip,
        )
        count = len(segment.generated_ids)
        logp = logits[:, -(count + 1):-1].float().log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1).mean()
        kl = torch.zeros((), device=self.device)
        if self.cfg.kl_coef:
            with torch.no_grad():
                ref_logits = self.reference(
                    replay.sequences, attention_mask=replay.attention_mask,
                    position_ids=replay.position_ids, use_cache=False,
                ).logits[:, -(count + 1):-1]
                ref_logp = ref_logits.float().log_softmax(-1).gather(
                    -1, replay.sequences[:, -count:].unsqueeze(-1)).squeeze(-1)
            delta = ref_logp - current
            kl = (delta.exp() - delta - 1).mean()
        total = actor_loss + self.cfg.kl_coef * kl - self.cfg.ent_coef * entropy
        if not torch.isfinite(total):
            raise ValueError("Nonfinite actor/regularization objective")
        return total, actor_loss.detach(), kl.detach(), entropy.detach()

    def update(self, batch: PPOBatch) -> dict:
        if self._failed:
            raise RuntimeError("Trainer update failed; reload a checkpoint before continuing")
        if batch is not self._pending_batch or batch.policy_version != self.policy_version:
            raise ValueError("PPO update requires this trainer's current unconsumed batch")
        self.policy.eval()
        self.critic.eval()  # eval disables dropout, not autograd
        sums = {"actor_loss": 0.0, "critic_loss": 0.0, "reference_kl": 0.0, "entropy": 0.0}
        groups = 0
        try:
            for _ in range(self.cfg.ppo_epochs):
                order = list(batch.trajectories)
                self._rng.shuffle(order)
                for start in range(0, len(order), self.cfg.ppo_mini_batch):
                    minibatch = order[start:start + self.cfg.ppo_mini_batch]
                    scale = len(minibatch)
                    self.optimizer.zero_grad(set_to_none=True)
                    for trajectory in minibatch:
                        for segment in trajectory.segments:
                            loss, actor, kl, entropy = self._segment_actor_loss(segment)
                            # Paper: sum token-mean segment losses, then mean trajectories.
                            (loss / scale).backward()
                            sums["actor_loss"] += float(actor.item()) / scale
                            sums["reference_kl"] += float(kl.item()) / scale
                            sums["entropy"] += float(entropy.item()) / scale
                    if self.cfg.vf_coef and self.cfg.method != "grpo":
                        for trajectory in minibatch:
                            weight = 1 / (scale * len(trajectory.segments))
                            for segment in trajectory.segments:
                                ids = torch.tensor([segment.prompt_ids], device=self.device)
                                value = self.critic(ids, torch.ones_like(ids))
                                loss = (value - trajectory.reward).square().mean()
                                (self.cfg.vf_coef * weight * loss).backward()
                                sums["critic_loss"] += weight * float(loss.detach().item())
                    parameters = (self._parameters if self.cfg.vf_coef and self.cfg.method != "grpo"
                                  else tuple(self.policy.parameters()))
                    average_gradients(parameters)
                    torch.nn.utils.clip_grad_norm_(
                        parameters, self.cfg.grad_clip, error_if_nonfinite=True,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    groups += 1
        except Exception:
            # A partial optimizer step cannot be retried as the original on-policy batch.
            self._failed = True
            raise
        self._pending_batch = None
        self.policy_version += 1
        self._updates += 1
        return {
            **{key: value / groups for key, value in sums.items()},
            "policy_version": self.policy_version, "optimizer_minibatches": groups,
            "n_trajectories": len(batch.trajectories),
            "mean_reward": sum(t.reward for t in batch.trajectories) / len(batch.trajectories),
        }

    def collect(self, prompts, generator, execute) -> PPOBatch:
        """Use the same policy object for rollout and updates; no external KV survives."""
        from carl.ppo.trainer import collect_segment_rollouts

        if (generator.policy is not self.policy or generator.tokenizer is not self.tokenizer
                or generator.policy_version != self.policy_version):
            raise ValueError("Generator must own this policy/tokenizer and current policy version")
        if self._pending_batch is not None or self._failed:
            raise RuntimeError("Cannot collect with a pending or failed update")
        self.policy.eval()
        rollouts = collect_segment_rollouts(prompts, generator, execute, self.cfg)
        batch = self.prepare_batch([row["trajectory"] for row in rollouts])
        if self.cfg.method == "grpo":
            groups = {}
            for row, trajectory in zip(rollouts, batch.trajectories):
                groups.setdefault(row["sample_id"], []).append(trajectory)
            for group in groups.values():
                rewards = torch.tensor([t.reward for t in group])
                advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
                for trajectory, advantage in zip(group, advantages.tolist()):
                    for segment in trajectory.segments:
                        segment.advantage = advantage
        return batch
