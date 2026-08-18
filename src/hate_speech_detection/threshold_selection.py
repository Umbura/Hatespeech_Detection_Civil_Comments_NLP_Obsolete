"""Threshold selection utilities for nested hierarchical evaluation.

Thresholds are selected only on an inner validation partition. The outer
validation fold must remain untouched until final metric computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, recall_score


DEFAULT_THRESHOLD_CANDIDATES = np.round(np.arange(0.01, 1.00, 0.01), 2)


@dataclass(frozen=True)
class RoutingThresholdSelection:
    """Result of routing-threshold selection on inner validation data."""

    threshold: float
    macro_f1: float
    gate_recall: float
    routing_rate: float


@dataclass(frozen=True)
class LabelThresholdSelection:
    """Per-label thresholds and the corresponding inner-validation F1 scores."""

    thresholds: np.ndarray
    f1_scores: np.ndarray


def _as_threshold_candidates(candidates: Iterable[float] | None) -> np.ndarray:
    values = (
        DEFAULT_THRESHOLD_CANDIDATES
        if candidates is None
        else np.asarray(list(candidates), dtype=np.float64)
    )
    if values.ndim != 1 or values.size == 0:
        raise ValueError("threshold candidates must be a non-empty 1D sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("threshold candidates must be finite")
    if np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("threshold candidates must be strictly between 0 and 1")
    return np.unique(values)


def _macro_positive_label_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean positive-class F1 across multilabel output columns.

    Computing the mean explicitly keeps the metric definition stable even for
    synthetic one-column tests, where scikit-learn may otherwise infer a binary
    target and average over positive and negative classes instead of labels.
    """

    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
        raise ValueError("multilabel arrays must be 2D and have identical shapes")
    if y_true.shape[1] == 0:
        raise ValueError("multilabel arrays must contain at least one output column")

    scores = [
        f1_score(y_true[:, index], y_pred[:, index], zero_division=0)
        for index in range(y_true.shape[1])
    ]
    return float(np.mean(scores))


def apply_label_thresholds(
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Binarize multilabel probabilities using one threshold per output."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    if thresholds.ndim != 1 or thresholds.size != probabilities.shape[1]:
        raise ValueError("threshold count must match the number of probability columns")
    return (probabilities >= thresholds.reshape(1, -1)).astype(np.int8)


def select_label_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    candidates: Iterable[float] | None = None,
    reference_threshold: float = 0.5,
) -> LabelThresholdSelection:
    """Select each label threshold independently by inner-validation F1.

    Ties are resolved in favor of the threshold closest to
    ``reference_threshold`` so threshold movement is no larger than necessary.
    """

    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("y_true and probabilities must both be 2D arrays")
    if y_true.shape != probabilities.shape:
        raise ValueError("y_true and probabilities must have identical shapes")

    threshold_candidates = _as_threshold_candidates(candidates)
    selected = np.empty(y_true.shape[1], dtype=np.float64)
    selected_scores = np.empty(y_true.shape[1], dtype=np.float64)

    for label_index in range(y_true.shape[1]):
        best_threshold = float(reference_threshold)
        best_score = -1.0
        best_distance = float("inf")

        for threshold in threshold_candidates:
            predicted = probabilities[:, label_index] >= threshold
            score = f1_score(
                y_true[:, label_index],
                predicted,
                zero_division=0,
            )
            distance = abs(float(threshold) - reference_threshold)
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and distance < best_distance - 1e-12
            ):
                best_score = float(score)
                best_threshold = float(threshold)
                best_distance = float(distance)

        selected[label_index] = best_threshold
        selected_scores[label_index] = best_score

    return LabelThresholdSelection(
        thresholds=selected,
        f1_scores=selected_scores,
    )


def select_routing_threshold(
    gate_true: np.ndarray,
    stage1_probabilities: np.ndarray,
    stage2_true: np.ndarray,
    stage2_probabilities: np.ndarray,
    stage2_thresholds: np.ndarray,
    *,
    candidates: Iterable[float] | None = None,
    reference_threshold: float = 0.4,
) -> RoutingThresholdSelection:
    """Select the Stage 1 threshold by inner-validation end-to-end Macro F1.

    Stage 2 predictions are first thresholded with the already-selected
    per-label thresholds. For each Stage 1 candidate, subtype predictions are
    zeroed for samples not routed by that candidate. Ties in Macro F1 prefer
    higher routing recall, then a threshold closer to the reference gate.
    """

    gate_true = np.asarray(gate_true, dtype=np.int8).reshape(-1)
    stage1_probabilities = np.asarray(stage1_probabilities, dtype=np.float64).reshape(-1)
    stage2_true = np.asarray(stage2_true, dtype=np.int8)
    stage2_probabilities = np.asarray(stage2_probabilities, dtype=np.float64)

    if stage2_true.ndim != 2 or stage2_probabilities.ndim != 2:
        raise ValueError("stage2_true and stage2_probabilities must be 2D arrays")
    if stage2_true.shape != stage2_probabilities.shape:
        raise ValueError("stage2_true and stage2_probabilities must have identical shapes")
    if gate_true.size != stage2_true.shape[0] or stage1_probabilities.size != gate_true.size:
        raise ValueError("all routing inputs must describe the same number of samples")

    label_predictions = apply_label_thresholds(
        stage2_probabilities,
        stage2_thresholds,
    )
    threshold_candidates = _as_threshold_candidates(candidates)

    best_threshold = float(reference_threshold)
    best_macro_f1 = -1.0
    best_recall = -1.0
    best_distance = float("inf")
    best_routing_rate = 0.0

    for threshold in threshold_candidates:
        routed = stage1_probabilities >= threshold
        cascade_predictions = label_predictions.copy()
        cascade_predictions[~routed] = 0

        macro_f1 = _macro_positive_label_f1(
            stage2_true,
            cascade_predictions,
        )
        gate_recall = recall_score(
            gate_true,
            routed.astype(np.int8),
            zero_division=0,
        )
        distance = abs(float(threshold) - reference_threshold)

        better = macro_f1 > best_macro_f1 + 1e-12
        tied_f1 = abs(macro_f1 - best_macro_f1) <= 1e-12
        better_recall = gate_recall > best_recall + 1e-12
        tied_recall = abs(gate_recall - best_recall) <= 1e-12
        closer = distance < best_distance - 1e-12

        if better or (tied_f1 and (better_recall or (tied_recall and closer))):
            best_threshold = float(threshold)
            best_macro_f1 = float(macro_f1)
            best_recall = float(gate_recall)
            best_distance = float(distance)
            best_routing_rate = float(np.mean(routed))

    return RoutingThresholdSelection(
        threshold=best_threshold,
        macro_f1=best_macro_f1,
        gate_recall=best_recall,
        routing_rate=best_routing_rate,
    )
