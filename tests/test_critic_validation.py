import pytest

from carl.critic.validation import boundary_key, passes_calibration_gate, validation_metrics


def boundary(question="a", index=0, tier="tier2", target=1):
    return {
        "dataset": "gsm8k", "q_idx": question, "rollout_idx": 0,
        "prompt_mode": "no_tool", "boundary_kind": "pre_segment",
        "boundary_index": index, "tier": tier, "V_target": target,
    }


@pytest.mark.parametrize("kind", [None, "legacy_snapshot", "unknown"])
def test_missing_or_unsupported_boundary_kind_is_rejected(kind):
    row = boundary()
    if kind is None:
        del row["boundary_kind"]
    else:
        row["boundary_kind"] = kind
    with pytest.raises(ValueError, match="boundary_kind='pre_segment'"):
        boundary_key(row)
    with pytest.raises(ValueError, match="boundary_kind='pre_segment'"):
        validation_metrics([row], [0.8])


def test_mixed_records_cannot_silently_exclude_unidentified_states():
    legacy = {"context": "Unidentified snapshot", "V_target": 0}
    with pytest.raises(ValueError, match="boundary_kind='pre_segment'"):
        validation_metrics([boundary(), legacy], [0.8, 0.2])


def test_current_boundaries_keep_metrics_and_gate_unchanged():
    rows = [
        boundary(index=0), boundary(index=1),
        boundary(question="b", index=0, tier="tier1", target=0),
        boundary(question="b", index=1, tier="tier1", target=0),
    ]
    rows[0].update(expected_value_change=1, value_change_source="checked transition")
    rows[2].update(expected_value_change=-1, value_change_source="checked transition")
    assert boundary_key(rows[0]) == ("gsm8k", "a", "no_tool", "0", 0)
    metrics = validation_metrics(rows, [0.8, 0.9, 0.2, 0.1])
    assert metrics["val_mse"] == metrics["brier_score"] == pytest.approx(0.025)
    assert metrics["ev"] == pytest.approx(0.9)
    assert metrics["competence_auc"] == metrics["transition_sign_accuracy"] == 1
    assert metrics["initial_ece"] == pytest.approx(0.2)
    assert metrics["n_boundaries"] == 4 and metrics["n_labeled_transitions"] == 2
    assert passes_calibration_gate(metrics, 0.7, 0.45, 0.6)


def test_missing_sign_evidence_still_fails_gate():
    metrics = validation_metrics(
        [boundary(), boundary(question="b", tier="tier1", target=0)], [0.9, 0.1])
    assert metrics["transition_sign_accuracy"] is None
    assert "transition_sign_accuracy" in metrics["missing_gate_evidence"]
    assert not passes_calibration_gate(metrics, 0.7, 0.45, 0.6)
