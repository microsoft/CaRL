"""On-policy collection with the same controller used at evaluation."""
from dataclasses import dataclass

from carl.core.rollout import rollout
from carl.eval.scoring import reward


@dataclass
class PPOConfig:
    actor_model: str = ""
    method: str = "carl"
    critic_ckpt: str | None = None
    rollout_batch_prompts: int = 256
    n_samples_per_prompt: int = 5
    ppo_mini_batch: int = 64
    ppo_epochs: int = 4
    eps_clip: float = 0.2
    gae_lambda: float = 0.0
    gamma: float = 1.0
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    kl_coef: float = 0.001
    invoke_max_tokens: int = 2048
    synthesize_max_tokens: int = 2048
    assimilate_max_tokens: int = 256
    max_segments: int = 15
    actor_lr: float = 1e-6
    critic_value_head_lr: float = 5e-6
    grad_clip: float = 1.0
    num_steps: int = 500
    eval_every: int = 25
    seed: int = 1
    warmup_steps: int = 100


def collect_segment_rollouts(prompts, generate, execute, cfg):
    results = []
    for group_index, example in enumerate(prompts):
        for sample in range(cfg.n_samples_per_prompt):
            trajectory = rollout(
                example["question"], generate, execute,
                max_segments=cfg.max_segments, invoke_max_tokens=cfg.invoke_max_tokens,
                assimilate_max_tokens=cfg.assimilate_max_tokens,
                initial_context=generate.render_prompt(example["question"]))
            trajectory.reward = reward(trajectory.final_answer_text, example["gold"], example["dataset"])
            results.append({"trajectory": trajectory, "sample_id": (group_index, example["sample_id"]), "rollout_idx": sample})
    return results
