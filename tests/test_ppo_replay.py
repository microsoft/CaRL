from dataclasses import replace
import math

import pytest
import torch

from carl.core.segment import Segment
from carl.ppo.replay import action_log_probs, clipped_policy_loss, pack_segments


def segment(prompt=(3, 4), actions=(5, 2), advantage=0.6):
    return Segment("Q", "answer", prompt_ids=list(prompt), generated_ids=list(actions),
                   log_probs=[-1.0] * len(actions), log_probs_kind="policy", advantage=advantage)


def test_mixed_lengths_preserve_inputs_and_right_aligned_action_metadata():
    first = segment(prompt=(3, 4), actions=(5,), advantage=0.6)
    second = segment(prompt=(6,), actions=(7, 8, 2), advantage=-0.7)
    replay = pack_segments([first, second], 0)
    assert replay.sequences.tolist() == [[0, 3, 4, 5], [6, 7, 8, 2]]
    assert replay.attention_mask.tolist() == [[0, 1, 1, 1], [1, 1, 1, 1]]
    assert replay.position_ids.tolist() == [[0, 0, 1, 2], [0, 1, 2, 3]]
    assert replay.labels.tolist() == [[-100, -100, -100, 5], [-100, 7, 8, 2]]
    assert replay.action_mask.tolist() == [[False, False, True], [True, True, True]]
    torch.testing.assert_close(replay.advantages, torch.tensor([[0, 0, 0.6], [-0.7, -0.7, -0.7]]))


def test_recorded_actions_are_scored_at_preceding_positions_including_first_token():
    replay = pack_segments([segment(actions=(5,)), segment(prompt=(6,), actions=(7, 8, 2))], 0)
    logits = torch.randn(2, 4, 16, requires_grad=True)
    scores = action_log_probs(logits, replay)
    expected = torch.stack([
        logits[1, 0].log_softmax(-1)[7],
        logits[1, 1].log_softmax(-1)[8],
        logits[1, 2].log_softmax(-1)[2],
    ])
    torch.testing.assert_close(scores[1], expected)
    torch.testing.assert_close(scores[0, 2], logits[0, 2].log_softmax(-1)[5])
    assert scores[0, :2].tolist() == [0, 0]
    scores.sum().backward()
    assert logits.grad[0, :2].count_nonzero() == 0
    assert logits.grad[:, -1].count_nonzero() == 0


def test_clipped_objective_freezes_old_scores_and_advantages():
    old = torch.tensor([[-2.0], [-2.0]], requires_grad=True)
    delta = torch.tensor([[math.log(1.5)], [math.log(0.5)]])
    current = (old.detach() + delta).requires_grad_()
    advantages = torch.tensor([[1.0], [-1.0]], requires_grad=True)
    mask = torch.ones(2, 1, dtype=torch.bool)
    loss = clipped_policy_loss(current, old, advantages, mask)
    assert loss.item() == pytest.approx((-1.2 + 0.8)/2)
    loss.backward()
    assert old.grad is None and advantages.grad is None


def test_unclipped_ratio_has_policy_gradient_even_when_policies_match():
    current = torch.tensor([[-2.0]], requires_grad=True)
    loss = clipped_policy_loss(current, current.detach(), torch.tensor([[0.6]]),
                               torch.ones(1, 1, dtype=torch.bool))
    loss.backward()
    assert current.grad.item() == pytest.approx(-0.6)


def test_padding_is_ignored_and_segments_receive_equal_token_mean_weight():
    current = torch.tensor([[-2.0, float("nan")], [-2.0, -2.0]], requires_grad=True)
    old = current.detach().clone()
    advantages = torch.tensor([[1.0, float("nan")], [3.0, 3.0]])
    mask = torch.tensor([[True, False], [True, True]])
    loss = clipped_policy_loss(current, old, advantages, mask)
    assert loss.item() == pytest.approx(-2.0)
    loss.backward()
    assert current.grad[0, 1] == 0


@pytest.mark.parametrize("changes", [
    {"prompt_ids": []}, {"generated_ids": []}, {"log_probs": []},
    {"log_probs_kind": "sampling"}, {"log_probs": [float("nan"), -1.0]},
    {"log_probs": [0.1, -1.0]}, {"advantage": None}, {"advantage": float("inf")},
])
def test_invalid_replay_records_are_rejected(changes):
    with pytest.raises(ValueError):
        pack_segments([replace(segment(), **changes)], 0)
