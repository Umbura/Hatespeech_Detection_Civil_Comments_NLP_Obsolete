"""Cross-validation folds for the hierarchical toxicity strategy.

Toxic samples are stratified with iterative multilabel stratification using
binarized fine-grained targets. Non-toxic samples are distributed separately
with K-fold splitting, then both partitions are recombined into full train and
validation folds.

The target scores themselves remain fractional for training; binarization here
is used only to construct evaluation folds and internal early-stopping splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hate_speech_detection.target_strategy import (
    DEFAULT_GATE_THRESHOLD,
    DEFAULT_LABEL_THRESHOLD,
    get_stage2_binary_targets,
    get_stage2_gate_mask,
)


@dataclass(frozen=True)
class HierarchicalSplit:
    """One full-dataset split with toxic and non-toxic partitions identified."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    stage2_train_indices: np.ndarray
    stage2_validation_indices: np.ndarray


@dataclass(frozen=True)
class TrainValidationSplit:
    """A training-only split used for early stopping without touching outer validation."""

    train_indices: np.ndarray
    validation_indices: np.ndarray


def make_hierarchical_splits(
    frame: Any,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
) -> list[HierarchicalSplit]:
    """Create deterministic outer folds that preserve fine-grained multilabel balance.

    Stage 2 toxic examples are split with ``MultilabelStratifiedKFold``.
    Non-toxic examples are split independently with ``KFold``. Corresponding
    partitions are recombined so Stage 1 can still train and validate on the
    full dataset while Stage 2 uses only toxic rows from the same outer fold.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    from sklearn.model_selection import KFold

    gate_mask = get_stage2_gate_mask(frame, gate_threshold=gate_threshold)
    toxic_indices = np.flatnonzero(gate_mask)
    non_toxic_indices = np.flatnonzero(~gate_mask)

    if len(toxic_indices) < n_splits:
        raise ValueError("not enough routed/toxic samples for the requested folds")
    if len(non_toxic_indices) < n_splits:
        raise ValueError("not enough non-toxic samples for the requested folds")

    toxic_targets = get_stage2_binary_targets(
        frame.iloc[toxic_indices],
        label_threshold=label_threshold,
    )

    toxic_splitter = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    toxic_folds = list(
        toxic_splitter.split(
            np.zeros((len(toxic_indices), 1), dtype=np.int8),
            toxic_targets,
        )
    )

    non_toxic_splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    non_toxic_folds = list(non_toxic_splitter.split(non_toxic_indices))

    splits: list[HierarchicalSplit] = []
    for (toxic_train, toxic_validation), (
        non_toxic_train,
        non_toxic_validation,
    ) in zip(toxic_folds, non_toxic_folds):
        stage2_train = np.sort(toxic_indices[toxic_train])
        stage2_validation = np.sort(toxic_indices[toxic_validation])

        train_indices = np.sort(
            np.concatenate(
                [stage2_train, non_toxic_indices[non_toxic_train]],
            )
        )
        validation_indices = np.sort(
            np.concatenate(
                [stage2_validation, non_toxic_indices[non_toxic_validation]],
            )
        )

        if np.intersect1d(train_indices, validation_indices).size:
            raise RuntimeError("generated train/validation overlap")

        splits.append(
            HierarchicalSplit(
                train_indices=train_indices,
                validation_indices=validation_indices,
                stage2_train_indices=stage2_train,
                stage2_validation_indices=stage2_validation,
            )
        )

    return splits


def make_stage1_inner_split(
    frame: Any,
    train_indices: np.ndarray,
    *,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
) -> TrainValidationSplit:
    """Split one outer training fold for Stage 1 fitting and early stopping.

    The split is stratified on the binary toxicity routing target so the outer
    validation fold remains untouched until final metric computation.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    from sklearn.model_selection import StratifiedShuffleSplit

    outer_train = np.asarray(train_indices, dtype=np.int64)
    if outer_train.size < 4:
        raise ValueError("not enough samples to create an internal validation split")

    local_frame = frame.iloc[outer_train]
    gate_targets = get_stage2_gate_mask(
        local_frame,
        gate_threshold=gate_threshold,
    ).astype(np.int8)
    if np.unique(gate_targets).size < 2:
        raise ValueError("Stage 1 internal split requires both routed and non-routed samples")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )
    fit_positions, stop_positions = next(
        splitter.split(
            np.zeros((outer_train.size, 1), dtype=np.int8),
            gate_targets,
        )
    )

    return TrainValidationSplit(
        train_indices=np.sort(outer_train[fit_positions]),
        validation_indices=np.sort(outer_train[stop_positions]),
    )


def make_stage2_inner_split(
    frame: Any,
    stage2_train_indices: np.ndarray,
    *,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
) -> TrainValidationSplit:
    """Split routed outer-training rows for Stage 2 fitting and early stopping.

    Iterative multilabel stratification is reused here so rare fine-grained
    labels remain represented while the outer Stage 2 validation partition is
    reserved exclusively for final evaluation.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    outer_train = np.asarray(stage2_train_indices, dtype=np.int64)
    if outer_train.size < 4:
        raise ValueError("not enough routed samples to create an internal validation split")

    targets = get_stage2_binary_targets(
        frame.iloc[outer_train],
        label_threshold=label_threshold,
    )
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )
    fit_positions, stop_positions = next(
        splitter.split(
            np.zeros((outer_train.size, 1), dtype=np.int8),
            targets,
        )
    )

    return TrainValidationSplit(
        train_indices=np.sort(outer_train[fit_positions]),
        validation_indices=np.sort(outer_train[stop_positions]),
    )
