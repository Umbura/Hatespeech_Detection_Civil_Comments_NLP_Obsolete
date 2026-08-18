"""Cross-validation folds for the hierarchical toxicity strategy.

Toxic samples are stratified with iterative multilabel stratification using
binarized fine-grained targets. Non-toxic samples are distributed separately
with K-fold splitting, then both partitions are recombined into full train and
validation folds.

The target scores themselves remain fractional for training; binarization here
is used only to construct evaluation folds.
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


def make_hierarchical_splits(
    frame: Any,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
) -> list[HierarchicalSplit]:
    """Create deterministic folds that preserve fine-grained multilabel balance.

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
