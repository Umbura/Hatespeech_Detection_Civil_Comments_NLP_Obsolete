"""Leakage-safe cross-validation preparation utilities.

The historical notebook balances the full dataset and fits the tokenizer before
creating validation folds. This module defines the repaired preparation order:

1. split the raw labeled dataset;
2. balance only the training partition;
3. fit the tokenizer only on training text;
4. keep validation rows untouched.

Target-label semantics are intentionally not changed here. They are a separate
review item documented in ``docs/KNOWN_ISSUES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class PreparedFold:
    """Training and validation material prepared for one CV fold."""

    train_frame: Any
    validation_frame: Any
    tokenizer: Any


def make_stratified_splits(
    labels: Sequence[Any],
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[Any, Any]]:
    """Create reproducible stratified indices from the unbalanced labels."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    samples = list(range(len(labels)))
    return list(splitter.split(samples, labels))


def balance_training_frame(
    train_frame: Any,
    *,
    target_samples: int,
    label_col: str = "label",
    text_col: str = "text",
    random_state: int = 42,
) -> Any:
    """Balance one training partition while preserving source-row identity.

    Majority classes are undersampled without replacement and minority classes
    are oversampled with ``RandomOverSampler``. The function accepts only a
    training partition so validation rows cannot enter the resampling stage.
    """

    if target_samples <= 0:
        raise ValueError("target_samples must be positive")

    import pandas as pd
    from imblearn.over_sampling import RandomOverSampler

    missing = {label_col, text_col}.difference(train_frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    columns = [text_col, label_col]
    if "_source_id" in train_frame.columns:
        columns.append("_source_id")

    working = train_frame[columns].copy()
    if "_source_id" not in working.columns:
        working["_source_id"] = train_frame.index.to_numpy(copy=True)

    balanced_parts = []
    minority_parts = []
    minority_labels = []

    for label in working[label_col].unique():
        subset = working[working[label_col] == label]
        if len(subset) >= target_samples:
            balanced_parts.append(
                subset.sample(n=target_samples, random_state=random_state)
            )
        else:
            minority_parts.append(subset)
            minority_labels.append(label)

    if minority_parts:
        minority = pd.concat(minority_parts, ignore_index=True)
        sampler = RandomOverSampler(
            sampling_strategy={label: target_samples for label in minority_labels},
            random_state=random_state,
        )
        features, labels = sampler.fit_resample(
            minority[[text_col, "_source_id"]], minority[label_col]
        )
        oversampled = features.copy()
        oversampled[label_col] = labels.to_numpy()
        balanced_parts.append(oversampled)

    if not balanced_parts:
        raise ValueError("training partition contains no classes to balance")

    return (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def prepare_fold(
    frame: Any,
    train_indices: Iterable[int],
    validation_indices: Iterable[int],
    *,
    target_samples: int,
    tokenizer_factory: Callable[..., Any],
    num_words: int,
    label_col: str = "label",
    text_col: str = "text",
    random_state: int = 42,
) -> PreparedFold:
    """Prepare one fold without exposing validation data to learned preprocessing."""

    train_indices = tuple(int(index) for index in train_indices)
    validation_indices = tuple(int(index) for index in validation_indices)

    if set(train_indices).intersection(validation_indices):
        raise ValueError("training and validation indices must be disjoint")

    train_raw = frame.iloc[list(train_indices)].copy()
    validation = frame.iloc[list(validation_indices)].copy()

    train_balanced = balance_training_frame(
        train_raw,
        target_samples=target_samples,
        label_col=label_col,
        text_col=text_col,
        random_state=random_state,
    )

    tokenizer = tokenizer_factory(num_words=num_words)
    tokenizer.fit_on_texts(train_balanced[text_col].tolist())

    return PreparedFold(
        train_frame=train_balanced,
        validation_frame=validation,
        tokenizer=tokenizer,
    )
