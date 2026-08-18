"""Cross-validation folds for the hierarchical toxicity strategy.

Toxic samples are stratified with iterative multilabel stratification using
binarized fine-grained targets. Non-toxic samples are distributed separately
with K-fold or shuffled splitting, then both partitions are recombined into
full train and validation folds.

The target scores themselves remain fractional for training; binarization here
is used only to construct evaluation folds and internal model-selection splits.
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
    """One full-dataset split with routed and non-routed partitions identified."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    stage2_train_indices: np.ndarray
    stage2_validation_indices: np.ndarray


@dataclass(frozen=True)
class TrainValidationSplit:
    """A training-only split used for early stopping without touching outer validation."""

    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class HierarchicalInnerSplit:
    """Common inner split used by both stages for model and threshold selection."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    stage2_train_indices: np.ndarray
    stage2_validation_indices: np.ndarray


def make_hierarchical_splits(
    frame: Any,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
) -> list[HierarchicalSplit]:
    """Create deterministic outer folds that preserve fine-grained multilabel balance.

    Stage 2 routed examples are split with ``MultilabelStratifiedKFold``.
    Non-routed examples are split independently with ``KFold``. Corresponding
    partitions are recombined so Stage 1 can still train and validate on the
    full dataset while Stage 2 uses only routed rows from the same outer fold.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    from sklearn.model_selection import KFold

    gate_mask = get_stage2_gate_mask(frame, gate_threshold=gate_threshold)
    routed_indices = np.flatnonzero(gate_mask)
    non_routed_indices = np.flatnonzero(~gate_mask)

    if len(routed_indices) < n_splits:
        raise ValueError("not enough routed samples for the requested folds")
    if len(non_routed_indices) < n_splits:
        raise ValueError("not enough non-routed samples for the requested folds")

    routed_targets = get_stage2_binary_targets(
        frame.iloc[routed_indices],
        label_threshold=label_threshold,
    )

    routed_splitter = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    routed_folds = list(
        routed_splitter.split(
            np.zeros((len(routed_indices), 1), dtype=np.int8),
            routed_targets,
        )
    )

    non_routed_splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    non_routed_folds = list(non_routed_splitter.split(non_routed_indices))

    splits: list[HierarchicalSplit] = []
    for (routed_train, routed_validation), (
        non_routed_train,
        non_routed_validation,
    ) in zip(routed_folds, non_routed_folds):
        stage2_train = np.sort(routed_indices[routed_train])
        stage2_validation = np.sort(routed_indices[routed_validation])

        train_indices = np.sort(
            np.concatenate(
                [stage2_train, non_routed_indices[non_routed_train]],
            )
        )
        validation_indices = np.sort(
            np.concatenate(
                [stage2_validation, non_routed_indices[non_routed_validation]],
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


def make_hierarchical_inner_split(
    frame: Any,
    outer_train_indices: np.ndarray,
    *,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
) -> HierarchicalInnerSplit:
    """Create one common inner split for both stages.

    Routed rows are split with iterative multilabel stratification and
    non-routed rows are split independently. The two partitions are recombined
    into a common full-population inner validation set. Stage 2 fitting uses
    only the routed subset of the common inner training set, and Stage 2 early
    stopping/threshold selection uses only the routed subset of the common
    inner validation set.

    This alignment is required for leakage-free end-to-end threshold selection:
    every row used to choose a routing threshold is excluded from both Stage 1
    and Stage 2 fitting.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    from sklearn.model_selection import ShuffleSplit

    outer_train = np.asarray(outer_train_indices, dtype=np.int64)
    if outer_train.size < 8:
        raise ValueError("not enough samples to create a hierarchical inner split")

    local_frame = frame.iloc[outer_train]
    local_gate = get_stage2_gate_mask(
        local_frame,
        gate_threshold=gate_threshold,
    )
    routed_indices = outer_train[np.flatnonzero(local_gate)]
    non_routed_indices = outer_train[np.flatnonzero(~local_gate)]

    if routed_indices.size < 4:
        raise ValueError("not enough routed samples for an internal validation split")
    if non_routed_indices.size < 4:
        raise ValueError("not enough non-routed samples for an internal validation split")

    routed_targets = get_stage2_binary_targets(
        frame.iloc[routed_indices],
        label_threshold=label_threshold,
    )
    routed_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )
    routed_fit_positions, routed_stop_positions = next(
        routed_splitter.split(
            np.zeros((routed_indices.size, 1), dtype=np.int8),
            routed_targets,
        )
    )

    non_routed_splitter = ShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )
    non_routed_fit_positions, non_routed_stop_positions = next(
        non_routed_splitter.split(non_routed_indices)
    )

    stage2_train = np.sort(routed_indices[routed_fit_positions])
    stage2_validation = np.sort(routed_indices[routed_stop_positions])
    train_indices = np.sort(
        np.concatenate(
            [stage2_train, non_routed_indices[non_routed_fit_positions]],
        )
    )
    validation_indices = np.sort(
        np.concatenate(
            [stage2_validation, non_routed_indices[non_routed_stop_positions]],
        )
    )

    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("generated inner train/validation overlap")
    if np.intersect1d(stage2_train, stage2_validation).size:
        raise RuntimeError("generated Stage 2 inner train/validation overlap")
    if not np.array_equal(
        np.sort(np.concatenate([train_indices, validation_indices])),
        np.sort(outer_train),
    ):
        raise RuntimeError("inner split does not cover the complete outer training fold")

    return HierarchicalInnerSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        stage2_train_indices=stage2_train,
        stage2_validation_indices=stage2_validation,
    )


def make_stage1_inner_split(
    frame: Any,
    train_indices: np.ndarray,
    *,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
) -> TrainValidationSplit:
    """Split one outer training fold for Stage 1 fitting and early stopping.

    This compatibility helper stratifies on the binary routing target. New
    hierarchical experiments should prefer ``make_hierarchical_inner_split``
    when Stage 1 and Stage 2 need a shared model-selection partition.
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

    This compatibility helper preserves iterative multilabel stratification.
    New hierarchical experiments should prefer ``make_hierarchical_inner_split``
    when Stage 1 and Stage 2 need a shared model-selection partition.
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
