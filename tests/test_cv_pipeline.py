from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from hate_speech_detection.cv_pipeline import (
    balance_training_frame,
    make_stratified_splits,
    prepare_fold,
)


class RecordingTokenizer:
    def __init__(self, *, num_words):
        self.num_words = num_words
        self.fitted_texts = []

    def fit_on_texts(self, texts):
        self.fitted_texts = list(texts)


class CrossValidationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "text": [
                    "train alpha one",
                    "train alpha two",
                    "train beta one",
                    "train beta two",
                    "validation alpha sentinel",
                    "validation beta sentinel",
                ],
                "label": ["alpha", "alpha", "beta", "beta", "alpha", "beta"],
            }
        )

    def test_prepare_fold_keeps_validation_out_of_resampling_and_tokenizer_fit(self):
        prepared = prepare_fold(
            self.frame,
            train_indices=[0, 1, 2, 3],
            validation_indices=[4, 5],
            target_samples=3,
            tokenizer_factory=RecordingTokenizer,
            num_words=100,
            random_state=42,
        )

        self.assertEqual(
            prepared.validation_frame["text"].tolist(),
            ["validation alpha sentinel", "validation beta sentinel"],
        )
        self.assertEqual(prepared.validation_frame.index.tolist(), [4, 5])

        validation_texts = set(prepared.validation_frame["text"])
        self.assertTrue(validation_texts.isdisjoint(prepared.tokenizer.fitted_texts))
        self.assertEqual(
            prepared.tokenizer.fitted_texts,
            prepared.train_frame["text"].tolist(),
        )

        train_source_ids = set(prepared.train_frame["_source_id"])
        validation_source_ids = set(prepared.validation_frame.index)
        self.assertTrue(train_source_ids.isdisjoint(validation_source_ids))
        self.assertTrue(train_source_ids.issubset({0, 1, 2, 3}))

    def test_balance_training_frame_reaches_target_using_training_rows_only(self):
        train = self.frame.iloc[[0, 1, 2]].copy()
        balanced = balance_training_frame(
            train,
            target_samples=4,
            random_state=42,
        )

        counts = balanced["label"].value_counts().to_dict()
        self.assertEqual(counts, {"alpha": 4, "beta": 4})
        self.assertTrue(set(balanced["_source_id"]).issubset({0, 1, 2}))
        self.assertNotIn(4, set(balanced["_source_id"]))
        self.assertNotIn(5, set(balanced["_source_id"]))

    def test_prepare_fold_rejects_overlapping_indices(self):
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            prepare_fold(
                self.frame,
                train_indices=[0, 1, 2],
                validation_indices=[2, 3],
                target_samples=2,
                tokenizer_factory=RecordingTokenizer,
                num_words=100,
            )

    def test_stratified_splits_are_disjoint_and_cover_each_sample_once_in_validation(self):
        labels = ["alpha"] * 6 + ["beta"] * 6
        splits = make_stratified_splits(labels, n_splits=3, random_state=42)

        validation_indices = []
        for train_indices, fold_validation_indices in splits:
            self.assertTrue(set(train_indices).isdisjoint(fold_validation_indices))
            validation_indices.extend(int(index) for index in fold_validation_indices)

        self.assertEqual(sorted(validation_indices), list(range(len(labels))))


if __name__ == "__main__":
    unittest.main()
