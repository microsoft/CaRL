"""Boundary-aware critic diagnostics; terminal outcomes are not step labels."""
from collections import defaultdict
import math

import torch


def explained_variance(targets, predictions):
    variance = targets.var(unbiased=False)
    if variance.item() == 0:
        return None
    return 1.0 - (targets - predictions).var(unbiased=False).item() / variance.item()


def binary_auc(scores, labels):
    """Rank-based AUC, with half credit for ties and no invented single-class AUC."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    ranked = sorted(zip(scores, labels))
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(label for _, label in ranked[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def boundary_key(row):
    """Identify the exact pre-segment state within a question's rollout."""
    if row.get("boundary_kind") != "pre_segment":
        raise ValueError("Validation requires boundary_kind='pre_segment'")
    for field in ("dataset", "prompt_mode"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Pre-segment boundary requires {field}")
    for field in ("q_idx", "rollout_idx"):
        value = row.get(field)
        if (isinstance(value, bool) or not isinstance(value, (str, int))
                or not str(value).strip()):
            raise ValueError(f"Pre-segment boundary requires {field}")
    index = row.get("boundary_index")
    if type(index) is not int or index < 0:
        raise ValueError("Pre-segment boundary_index must be a nonnegative integer")
    if row.get("tier") not in (None, "tier1", "tier2"):
        raise ValueError("Boundary tier must be tier1 or tier2")
    if "expected_value_change" in row:
        direction = row["expected_value_change"]
        if type(direction) is not int or direction not in (-1, 1):
            raise ValueError("expected_value_change must be -1 or 1")
        source = row.get("value_change_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Transition sign labels require a value_change_source")
    return (
        row["dataset"], str(row["q_idx"]), row["prompt_mode"],
        str(row["rollout_idx"]), index,
    )


def validation_metrics(rows, predictions, competence_prompt_mode="no_tool"):
    """Use no-tool s0 tier labels and explicitly annotated adjacent transitions.

    Repeated initial states are averaged per question before competence AUC.
    Sign annotations belong to the transition from this row to the next
    boundary in the same trajectory. They must be supplied independently,
    never inferred from terminal reward by this evaluator.
    """
    if not rows or len(rows) != len(predictions):
        raise ValueError("Validation requires one prediction per non-empty boundary row")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in predictions):
        raise ValueError("Critic predictions must be finite probabilities in [0, 1]")
    if any(row.get("V_target") not in (0, 1) for row in rows):
        raise ValueError("Validation requires binary V_target outcomes")
    if not isinstance(competence_prompt_mode, str) or not competence_prompt_mode.strip():
        raise ValueError("competence_prompt_mode must be non-empty")

    boundaries = {}
    initial_scores = defaultdict(list)
    initial_outcomes = defaultdict(list)
    tiers = {}
    for row, prediction in zip(rows, predictions):
        key = boundary_key(row)
        if key in boundaries:
            raise ValueError(f"Duplicate validation boundary: {key}")
        boundaries[key] = (row, prediction)
        question = key[:2]
        tier = row.get("tier")
        if tier is not None:
            if question in tiers and tiers[question] != tier:
                raise ValueError(f"Conflicting competence tiers for question {question}")
            tiers[question] = tier
        if key[-1] == 0 and row["prompt_mode"] == competence_prompt_mode and tier is not None:
            initial_scores[question].append(prediction)
            initial_outcomes[question].append(float(row["V_target"]))

    competence_scores = [
        sum(scores) / len(scores) for scores in initial_scores.values()
    ]
    competence_labels = [int(tiers[question] == "tier2") for question in initial_scores]
    outcome_means = [sum(initial_outcomes[q]) / len(initial_outcomes[q]) for q in initial_scores]
    ece = None
    if competence_scores:
        ece = 0.0
        for index in range(10):
            selected = [(s, y) for s, y in zip(competence_scores, outcome_means)
                        if min(int(s * 10), 9) == index]
            if selected:
                ece += abs(sum(s-y for s, y in selected)) / len(competence_scores)
    signs = []
    for key, (row, before) in boundaries.items():
        if "expected_value_change" not in row:
            continue
        next_key = (*key[:-1], key[-1] + 1)
        if next_key not in boundaries:
            raise ValueError(f"Sign-labeled boundary is missing its adjacent state: {key}")
        after = boundaries[next_key][1]
        signs.append(int((after - before) * row["expected_value_change"] > 0))

    targets = torch.tensor([float(row["V_target"]) for row in rows], dtype=torch.float64)
    scores = torch.tensor(predictions, dtype=torch.float64)
    mse = (targets - scores).square().mean().item()
    metrics = {
        "val_mse": mse,
        "ev": explained_variance(targets, scores),
        "outcome_auc": binary_auc(predictions, targets.int().tolist()),
        "outcome_accuracy": ((scores >= 0.5) == (targets >= 0.5)).double().mean().item(),
        "brier_score": mse,
        "initial_ece": ece,
        "competence_auc": binary_auc(competence_scores, competence_labels),
        "competence_prompt_mode": competence_prompt_mode,
        "transition_sign_accuracy": sum(signs) / len(signs) if signs else None,
        "n_boundaries": len(rows),
        "n_competence_questions": len(initial_scores),
        "n_labeled_transitions": len(signs),
        "gate_scope": "boundary_calibration",
    }
    metrics["missing_gate_evidence"] = [
        key for key in ("competence_auc", "ev", "transition_sign_accuracy")
        if metrics[key] is None
    ]
    return metrics


def passes_calibration_gate(metrics, auc_gate, ev_gate, sign_gate):
    """Missing or nonfinite measurements cannot pass any configured threshold."""
    checks = (
        ("competence_auc", auc_gate), ("ev", ev_gate),
        ("transition_sign_accuracy", sign_gate),
    )
    for key, threshold in checks:
        if not math.isfinite(threshold):
            raise ValueError(f"Nonfinite threshold for {key}")
    if not 0 <= auc_gate <= 1 or not 0 <= sign_gate <= 1:
        raise ValueError("AUC and sign-accuracy thresholds must be in [0, 1]")
    for key, threshold in checks:
        value = metrics.get(key)
        if value is None or not math.isfinite(value) or value < threshold:
            return False
    return True
