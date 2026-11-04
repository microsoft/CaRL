import pytest

from carl.ppo.advantages import segment_advantages


@pytest.mark.parametrize("values,reward,expected", [
    ([0.2, 0.8, 0.1], 1.0, [0.6, -0.7, 0.9]),
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
