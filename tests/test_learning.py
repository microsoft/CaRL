from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from carl.core.segment import Segment, Trajectory
from carl.critic.head import ValueHead, ValueModel
from carl.critic.validation import validation_metrics, passes_calibration_gate
from carl.learning import sft_loss, target_tokens
from carl.ppo.advantages import segment_advantages
from carl.ppo.replay import action_log_probs, pack_segments
from carl.ppo.shared import SharedPPOTrainer
from carl.ppo.trainer import PPOConfig
from carl.train import critic_step, split_questions, save


def model():
    torch.manual_seed(7)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
        eos_token_id=2, pad_token_id=0)).eval()


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def save_pretrained(self, path):
        from pathlib import Path
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "fixture.json").write_text("{}")


def trajectory(version=0):
    return Trajectory(segments=[
        Segment("question", "code", prompt_ids=[3, 4], generated_ids=[5],
                segment_type="invoke", termination="tool_call", policy_version=version),
        Segment("question+observation", "extraction", prompt_ids=[3, 4, 5, 6], generated_ids=[7],
                segment_type="assimilate", termination="context_block", policy_version=version),
        Segment("question+extraction", "answer", prompt_ids=[3, 4, 5, 7], generated_ids=[8, 2],
                policy_version=version),
    ], reward=1.0, termination_reason="eos")


def test_opposing_advantages_and_terminal_telescope():
    advantages = segment_advantages([0.2, 0.8, 0.1], 1)
    assert advantages == pytest.approx([0.6, -0.7, 0.9])
    assert sum(advantages) == pytest.approx(0.8)


@pytest.mark.parametrize("values,reward,expected", [
    ([0.7], 1.0, [0.3]),
    ([0.7], 0.0, [-0.7]),
    ([0.2, 0.8, 0.1], 0.0, [0.6, -0.7, -0.1]),
    ([0.2, 0.8, 0.1], 0.5, [0.6, -0.7, 0.4]),
])
def test_segment_advantages_preserve_terminal_credit(values, reward, expected):
    advantages = segment_advantages(values, reward)
    assert advantages == pytest.approx(expected)
    assert sum(advantages) == pytest.approx(reward - values[0])


def test_empty_boundary_values_are_rejected():
    with pytest.raises(ValueError, match="at least one boundary"):
        segment_advantages([], 1.0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_and_rewards_are_rejected(invalid):
    with pytest.raises(ValueError, match="finite"):
        segment_advantages([0.2, invalid, 0.1], 1.0)
    with pytest.raises(ValueError, match="finite"):
        segment_advantages([0.2, 0.8, 0.1], invalid)


@pytest.mark.parametrize("method", ["carl", "trajectory_ppo", "grpo"])
def test_actual_shared_policy_update_and_immutable_reference(method):
    policy = model()
    reference = deepcopy(policy)
    trainer = SharedPPOTrainer(policy, reference, Tokenizer(), PPOConfig(
        method=method, ppo_epochs=1, ppo_mini_batch=1, actor_lr=1e-3,
        critic_value_head_lr=1e-3))
    assert trainer.critic.backbone is policy.base_model
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    frozen = {k: v.clone() for k, v in reference.state_dict().items()}
    batch = trainer.prepare_batch([trajectory()])
    old_probs = [list(s.log_probs) for s in batch.trajectories[0].segments]
    metrics = trainer.update(batch)
    assert metrics["policy_version"] == 1
    assert any(not torch.equal(v, before[k]) for k, v in policy.state_dict().items())
    assert all(torch.equal(v, frozen[k]) for k, v in reference.state_dict().items())
    assert [s.log_probs for s in batch.trajectories[0].segments] == old_probs
    with pytest.raises(ValueError):
        trainer.update(batch)


def test_causal_first_token_and_prompt_masks():
    segment = Segment("Q", "A", prompt_ids=[3, 4], generated_ids=[5, 2],
                      log_probs=[-1.0, -1.0], log_probs_kind="policy", advantage=1)
    replay = pack_segments([segment], 0)
    logits = torch.randn(1, 4, 32)
    actual = action_log_probs(logits, replay)
    expected = torch.stack([logits[0, 1].log_softmax(-1)[5], logits[0, 2].log_softmax(-1)[2]])
    torch.testing.assert_close(actual[0], expected)
    assert replay.labels[0].tolist() == [-100, -100, 5, 2]


def test_sft_full_parameter_gradient_and_reference_kl():
    policy = model()
    reference = deepcopy(policy).requires_grad_(False).eval()
    loss = sft_loss(policy, reference, [3, 4], [5, 2], 0.1)
    loss.backward()
    assert policy.model.embed_tokens.weight.grad is not None
    assert policy.lm_head.weight.grad is not None
    assert all(p.grad is None for p in reference.parameters())
    with pytest.raises(ValueError, match="natural EOS"):
        target_tokens(asdict(Segment("Q", "A", prompt_ids=[3], generated_ids=[4])), Tokenizer())


def test_critic_optimizer_changes_shared_backbone():
    policy = model()
    value = ValueModel(policy.base_model)
    optimizer = torch.optim.AdamW(value.parameters(), lr=1e-2)
    rows = [{"prompt_ids": [3, 4], "V_target": 1.0}]
    before = {k: v.clone() for k, v in policy.base_model.state_dict().items()}
    # The zero-initialized final layer blocks backbone gradients only on step one.
    critic_step(value, rows, None, optimizer, torch.device("cpu"))
    critic_step(value, rows, None, optimizer, torch.device("cpu"))
    assert any(not torch.equal(v, before[k]) for k, v in policy.base_model.state_dict().items())


def test_question_split_and_missing_sign_gate():
    rows = [{"dataset": "gsm8k", "q_idx": str(i//3)} for i in range(12)]
    train, validation = split_questions(rows, 0.5, 1)
    assert {r["q_idx"] for r in train}.isdisjoint({r["q_idx"] for r in validation})
    assert not passes_calibration_gate({"competence_auc": 0.9, "ev": 0.9}, 0.7, 0.45, 0.6)


def test_checkpoint_publishes_last_and_preserves_rng(tmp_path):
    import random
    policy = model()
    optimizer = torch.optim.AdamW(policy.parameters())
    rng = random.Random(7)
    save(tmp_path, 1, policy, Tokenizer(), {"optimizer": optimizer.state_dict()}, rng)
    path = tmp_path / "step-000001"
    assert (path / "complete.json").is_file()
    state = torch.load(path / "state.pt", weights_only=False)
    restored = random.Random()
    restored.setstate(state["rank_rngs"][0])
    assert restored.random() == rng.random()
    fresh = Qwen2ForCausalLM.from_pretrained(path / "model")
    for k, v in fresh.state_dict().items():
        torch.testing.assert_close(v, policy.state_dict()[k])


def test_restored_rng_is_not_changed_by_checkpoint_model_construction(tmp_path):
    import random
    from carl.train import restore_rng
    policy = model()
    rng = random.Random(7)
    save(tmp_path, 1, policy, Tokenizer(), {}, rng)
    state = torch.load(tmp_path / "step-000001" / "state.pt", weights_only=False)
    restore_rng(state, 0)
    expected = torch.rand(5)
    model()  # Loading/constructing auxiliary models may consume/reset the RNG.
    restore_rng(state, 0)
    torch.testing.assert_close(torch.rand(5), expected)


def test_alternate_natural_eos_is_valid_but_control_eos_is_removed():
    segment = asdict(Segment("Q", "A", prompt_ids=[3], generated_ids=[4, 1],
                             decoding_constraints={"eos_token_ids": [1, 2]}))
    assert target_tokens(segment, Tokenizer()) == ([3], [4, 1])
    segment.update(segment_type="invoke", generated_ids=[4, 2], decoding_constraints={"forced_boundary_eos": True})
    assert target_tokens(segment, Tokenizer()) == ([3], [4])


def test_critic_padding_and_calibration_do_not_invent_step_labels():
    policy = model()
    value = ValueModel(policy.base_model).eval()
    with torch.no_grad():
        value.head.fc2.weight.fill_(0.1)
        left = value(torch.tensor([[0, 3, 4]]), torch.tensor([[0, 1, 1]]))
        right = value(torch.tensor([[3, 4, 0]]), torch.tensor([[1, 1, 0]]))
    torch.testing.assert_close(left, right)
    rows = []
    for q, tier, target in (("a", "tier2", 1), ("b", "tier1", 0)):
        for boundary in (0, 1):
            row = {"dataset": "gsm8k", "q_idx": q, "prompt_mode": "no_tool",
                   "rollout_idx": 0, "boundary_index": boundary, "boundary_kind": "pre_segment",
                   "tier": tier, "V_target": target}
            if boundary == 0:
                row.update(expected_value_change=1 if target else -1, value_change_source="fixture comparison")
            rows.append(row)
    metrics = validation_metrics(rows, [0.8, 0.9, 0.2, 0.1])
    assert metrics["competence_auc"] == metrics["transition_sign_accuracy"] == 1
    assert metrics["initial_ece"] == pytest.approx(0.2)
    assert metrics["brier_score"] == pytest.approx(0.025)
    assert passes_calibration_gate(metrics, 0.7, 0.45, 0.6)


def test_value_constructor_shares_backbone_and_preserves_a_supplied_head():
    policy = model()
    head = ValueHead(policy.config.hidden_size, mlp_hidden=4)
    with torch.no_grad():
        head.fc2.bias.fill_(0.25)
    value = ValueModel(policy.base_model, head)
    assert value.backbone is policy.base_model
    assert value.head is head
    assert next(value.head.parameters()).device == next(policy.parameters()).device
    assert next(value.head.parameters()).dtype == torch.float32
    with torch.no_grad():
        tokens = torch.tensor([[3, 4]])
        result = value(tokens, torch.ones_like(tokens))
    torch.testing.assert_close(result, torch.sigmoid(torch.tensor([0.25])))


def make_combined_trainer(**overrides):
    policy = model()
    return SharedPPOTrainer(policy, deepcopy(policy), Tokenizer(), PPOConfig(
        **{"ppo_epochs": 1, "ppo_mini_batch": 1, "actor_lr": 1e-3,
           "critic_value_head_lr": 2e-3, **overrides}))


def test_combined_optimizer_owns_each_shared_parameter_once():
    trainer = make_combined_trainer()
    parameters = [p for group in trainer.optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in trainer.critic.backbone.parameters()} <= {id(p) for p in parameters}
    assert not {id(p) for p in trainer.reference.parameters()} & {id(p) for p in parameters}
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-3
    assert trainer.optimizer.param_groups[1]["lr"] == 2e-3
    assert trainer.optimizer.param_groups[1]["weight_decay"] == 0


def test_combined_update_matches_one_step_on_the_literal_summed_objective():
    actual = make_combined_trainer()
    expected = make_combined_trainer()
    expected.policy.load_state_dict(actual.policy.state_dict())
    expected.critic.head.load_state_dict(actual.critic.head.state_dict())
    batch = actual.prepare_batch([trajectory()])
    manual = expected.prepare_batch([trajectory()])
    expected.optimizer.zero_grad(set_to_none=True)
    tr = manual.trajectories[0]
    loss = sum(expected._segment_actor_loss(s)[0] for s in tr.segments)
    for segment in tr.segments:
        ids = torch.tensor([segment.prompt_ids])
        value = expected.critic(ids, torch.ones_like(ids))
        loss = loss + expected.cfg.vf_coef * (value - tr.reward).square().mean() / len(tr.segments)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(expected._parameters, expected.cfg.grad_clip, error_if_nonfinite=True)
    expected.optimizer.step()
    actual.update(batch)
    for key, tensor in actual.policy.state_dict().items():
        torch.testing.assert_close(tensor, expected.policy.state_dict()[key])
    for key, tensor in actual.critic.head.state_dict().items():
        torch.testing.assert_close(tensor, expected.critic.head.state_dict()[key])
    assert all(int(s["step"]) == 1 for s in actual.optimizer.state.values())


def test_critic_forward_precedes_optimizer_step_and_step_count_matches_minibatches(monkeypatch):
    trainer = make_combined_trainer(ppo_epochs=2)
    batch = trainer.prepare_batch([trajectory(), trajectory()])
    original = {key: value.clone() for key, value in trainer.policy.state_dict().items()}
    steps = []
    reduction_calls = []
    forward_checks = []
    actual_step = trainer.optimizer.step

    def step():
        steps.append(True)
        actual_step()

    def before_critic(module, args):
        if not steps:
            assert all(torch.equal(value, original[key]) for key, value in trainer.policy.state_dict().items())
            forward_checks.append(True)

    def reduce(parameters):
        params = tuple(parameters)
        assert len(params) == len({id(p) for p in params})
        reduction_calls.append(True)

    monkeypatch.setattr(trainer.optimizer, "step", step)
    monkeypatch.setattr("carl.ppo.shared.average_gradients", reduce)
    handle = trainer.critic.register_forward_pre_hook(before_critic)
    try:
        metrics = trainer.update(batch)
    finally:
        handle.remove()
    assert forward_checks
    assert len(steps) == len(reduction_calls) == metrics["optimizer_minibatches"] == 4
    assert all(int(s["step"]) == 4 for s in trainer.optimizer.state.values())


@pytest.mark.parametrize("options", [{"method": "grpo"}, {"vf_coef": 0.0}])
def test_disabled_critic_head_gets_no_optimizer_state_or_update(options):
    trainer = make_combined_trainer(**options)
    before = deepcopy(trainer.critic.head.state_dict())
    trainer.update(trainer.prepare_batch([trajectory()]))
    assert all(torch.equal(v, before[k]) for k, v in trainer.critic.head.state_dict().items())
    assert all(p not in trainer.optimizer.state for p in trainer.critic.head.parameters())


def test_combined_optimizer_resume_reproduces_the_next_update(tmp_path):
    original = make_combined_trainer()
    original.update(original.prepare_batch([trajectory()]))
    save(tmp_path, 1, original.policy, Tokenizer(), {
        "head": original.critic.head.state_dict(), "optimizer": original.optimizer.state_dict(),
        "ppo_update_rule": "combined_adam_v1",
    }, original._rng)
    state = torch.load(tmp_path / "step-000001" / "state.pt", weights_only=False)
    resumed = make_combined_trainer()
    resumed.policy.load_state_dict(Qwen2ForCausalLM.from_pretrained(tmp_path / "step-000001" / "model").state_dict())
    resumed.critic.head.load_state_dict(state["head"])
    resumed.optimizer.load_state_dict(state["optimizer"])
    resumed.policy_version = state["step"]
    resumed._rng.setstate(state["rank_rngs"][0])
    assert state["ppo_update_rule"] == "combined_adam_v1"
    original.update(original.prepare_batch([trajectory(version=1)]))
    resumed.update(resumed.prepare_batch([trajectory(version=1)]))
    for key, value in original.policy.state_dict().items():
        torch.testing.assert_close(value, resumed.policy.state_dict()[key])
    for key, value in original.critic.head.state_dict().items():
        torch.testing.assert_close(value, resumed.critic.head.state_dict()[key])


def test_legacy_dual_optimizer_checkpoint_is_rejected(monkeypatch, tmp_path):
    from carl.train import optimize
    policy = model()
    monkeypatch.setattr("carl.train.load_model", lambda *args: (deepcopy(policy), Tokenizer()))
    args = SimpleNamespace(config=None, model="fixture", revision=None, steps=2)
    with pytest.raises(ValueError, match="dual-optimizer states cannot be converted"):
        optimize(args, policy, Tokenizer(), torch.device("cpu"), tmp_path, 1,
                 {"actor_optimizer": {}, "critic_optimizer": {}})


def test_failed_combined_update_cannot_be_retried(monkeypatch):
    trainer = make_combined_trainer()
    batch = trainer.prepare_batch([trajectory()])

    def failed_step():
        raise RuntimeError("Optimizer failure")

    monkeypatch.setattr(trainer.optimizer, "step", failed_step)
    with pytest.raises(RuntimeError, match="Optimizer failure"):
        trainer.update(batch)
    assert trainer.policy_version == 0
    with pytest.raises(RuntimeError, match="reload a checkpoint"):
        trainer.update(batch)
