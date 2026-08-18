"""Target semantics for the repaired hierarchical Civil Comments pipeline.

The historical experiment collapsed overlapping toxicity attributes into one
order-dependent multiclass label. The repaired strategy keeps the original
fractional annotation scores and separates the task into two stages:

Stage 1
    ``toxicity`` as the routing signal plus ``severe_toxicity`` as an
    auxiliary output, both trained as soft targets.

Stage 2
    Five independent fine-grained toxicity outputs, also trained as soft
    targets. ``non_toxic`` is not a synthetic output class.
"""

from __future__ import annotations

from typing import Any

import numpy as np


STAGE1_TARGET_COLUMNS = (
    "toxicity",
    "severe_toxicity",
)

STAGE2_TARGET_COLUMNS = (
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
)

ALL_TARGET_COLUMNS = STAGE1_TARGET_COLUMNS + STAGE2_TARGET_COLUMNS

DEFAULT_GATE_THRESHOLD = 0.5
DEFAULT_LABEL_THRESHOLD = 0.5


def _validate_threshold(value: float, *, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def validate_target_frame(frame: Any) -> None:
    """Validate that all required target columns exist and contain scores in [0, 1]."""

    missing = set(ALL_TARGET_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"missing target columns: {sorted(missing)}")

    scores = frame.loc[:, ALL_TARGET_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("target scores must be finite")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("target scores must be between 0 and 1")


def get_stage1_targets(frame: Any) -> np.ndarray:
    """Return Stage 1 fractional targets without binarizing annotator scores."""

    validate_target_frame(frame)
    return frame.loc[:, STAGE1_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)


def get_stage2_targets(frame: Any) -> np.ndarray:
    """Return Stage 2 fractional multilabel targets without collapsing overlaps."""

    validate_target_frame(frame)
    return frame.loc[:, STAGE2_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)


def get_stage2_binary_targets(
    frame: Any,
    *,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
) -> np.ndarray:
    """Binarize Stage 2 scores for stratification and evaluation only."""

    _validate_threshold(label_threshold, name="label_threshold")
    return (get_stage2_targets(frame) >= label_threshold).astype(np.int8)


def get_stage2_gate_mask(
    frame: Any,
    *,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
) -> np.ndarray:
    """Return the ground-truth routing mask used to define Stage 2 training data."""

    _validate_threshold(gate_threshold, name="gate_threshold")
    validate_target_frame(frame)
    toxicity = frame["toxicity"].to_numpy(dtype=float)
    return toxicity >= gate_threshold


def analyze_gate_coverage(
    frame: Any,
    *,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
) -> dict[str, Any]:
    """Quantify subtype positives that a ground-truth toxicity gate would exclude.

    This analysis must be run on the real dataset before the routing threshold is
    treated as empirically safe. It reports missed positives per fine-grained
    label and across samples with at least one positive Stage 2 label.
    """

    _validate_threshold(gate_threshold, name="gate_threshold")
    _validate_threshold(label_threshold, name="label_threshold")
    validate_target_frame(frame)

    gate = get_stage2_gate_mask(frame, gate_threshold=gate_threshold)
    binary = get_stage2_binary_targets(frame, label_threshold=label_threshold)

    per_label: dict[str, dict[str, float | int]] = {}
    for column_index, label in enumerate(STAGE2_TARGET_COLUMNS):
        positives = binary[:, column_index].astype(bool)
        positive_count = int(positives.sum())
        missed_count = int((positives & ~gate).sum())
        routed_count = positive_count - missed_count
        missed_rate = missed_count / positive_count if positive_count else 0.0
        per_label[label] = {
            "positive_count": positive_count,
            "routed_count": routed_count,
            "missed_count": missed_count,
            "missed_rate": missed_rate,
        }

    any_positive = binary.any(axis=1)
    any_positive_count = int(any_positive.sum())
    any_missed_count = int((any_positive & ~gate).sum())

    return {
        "total_samples": int(len(frame)),
        "routed_samples": int(gate.sum()),
        "gate_threshold": gate_threshold,
        "label_threshold": label_threshold,
        "per_label": per_label,
        "any_stage2_positive": {
            "positive_count": any_positive_count,
            "routed_count": any_positive_count - any_missed_count,
            "missed_count": any_missed_count,
            "missed_rate": (
                any_missed_count / any_positive_count if any_positive_count else 0.0
            ),
        },
    }
