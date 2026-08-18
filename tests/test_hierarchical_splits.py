from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from hate_speech_detection.hierarchical_splits import (
    make_hierarchical_inner_split,
    make_hierarchical_splits,
    make_stage1_inner_split,
    make_stage2_inner_split,
)
from hate_speech_detection.target_strategy import STAGE2_TARGET_COLUMNS


class HierarchicalSplitTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for index in range(30):
            row = {
                "toxicity": 0.9,
                "severe_toxicity": 0.6 if index % 6 == 0 else 0.1,
                "obscene": 0.9 if index % 3 == 0 else 0.1,
                "threat": 0.9 if index % 3 == 1 else 0.1,
                "insult": 0.9 if index % 3 == 2 else 0.1,
                "identity_attack": 0.9 if index % 5 in (0, 1) else 0.1,
                "sexual_explicit": 0.9 if index % 5 in (2, 3) else 0.1,
            }
            rows.append(row)

        for _ in range(30):
            rows.append(
                {
                    "toxicity": 0.1,
                    "severe_toxicity": 0.0,
                    "obscene": 0.0,
                    "threat": 0.0,
                    "insult": 0.0,
                    "identity_attack": 0.0,
                    "sexual_explicit": 0.0,
                }
            )

        self.frame = pd.DataFrame(rows)

    def test_splits_are_disjoint_and_cover_each_sample_once_in_validation(self):
        splits = make_hierarchical_splits(
            self.frame,
            n_splits=5,
            random_state=42,
        )

        validation_indices = []
        for split in splits:
            self.assertTrue(
                set(split.train_indices).isdisjoint(split.validation_indices)
            )
            self.assertTrue(
                set(split.stage2_train_indices).issubset(split.train_indices)
            )
            self.assertTrue(
                set(split.stage2_validation_indices).issubset(
                    split.validation_indices
                )
            )
            validation_indices.extend(int(index) for index in split.validation_indices)

        self.assertEqual(sorted(validation_indices), list(range(len(self.frame))))

    def test_stage2_partitions_contain_only_routed_samples(self):
        splits = make_hierarchical_splits(self.frame, n_splits=5, random_state=42)

        for split in splits:
            self.assertTrue(
                (self.frame.iloc[split.stage2_train_indices]["toxicity"] >= 0.5).all()
            )
            self.assertTrue(
                (
                    self.frame.iloc[split.stage2_validation_indices]["toxicity"]
                    >= 0.5
                ).all()
            )

    def test_iterative_stratification_distributes_each_fine_grained_label(self):
        splits = make_hierarchical_splits(self.frame, n_splits=5, random_state=42)

        for split in splits:
            validation = self.frame.iloc[split.stage2_validation_indices]
            for label in STAGE2_TARGET_COLUMNS:
                positives = int((validation[label] >= 0.5).sum())
                self.assertGreater(
                    positives,
                    0,
                    f"validation fold has no positive samples for {label}",
                )

    def test_stage2_is_not_oversampled(self):
        splits = make_hierarchical_splits(self.frame, n_splits=5, random_state=42)
        original_toxic_indices = set(range(30))

        for split in splits:
            self.assertEqual(
                len(split.stage2_train_indices),
                len(set(int(index) for index in split.stage2_train_indices)),
            )
            self.assertTrue(
                set(int(index) for index in split.stage2_train_indices).issubset(
                    original_toxic_indices
                )
            )

    def test_common_inner_split_is_shared_by_both_stages(self):
        outer = make_hierarchical_splits(
            self.frame,
            n_splits=2,
            random_state=42,
        )[0]
        inner = make_hierarchical_inner_split(
            self.frame,
            outer.train_indices,
            validation_fraction=0.25,
            random_state=43,
            gate_threshold=0.4,
            label_threshold=0.5,
        )

        inner_train = set(int(index) for index in inner.train_indices)
        inner_validation = set(int(index) for index in inner.validation_indices)
        outer_train = set(int(index) for index in outer.train_indices)
        outer_validation = set(int(index) for index in outer.validation_indices)
        stage2_train = set(int(index) for index in inner.stage2_train_indices)
        stage2_validation = set(
            int(index) for index in inner.stage2_validation_indices
        )

        self.assertTrue(inner_train.isdisjoint(inner_validation))
        self.assertEqual(inner_train | inner_validation, outer_train)
        self.assertTrue((inner_train | inner_validation).isdisjoint(outer_validation))
        self.assertTrue(stage2_train.issubset(inner_train))
        self.assertTrue(stage2_validation.issubset(inner_validation))
        self.assertTrue(stage2_train.isdisjoint(stage2_validation))

        routed_inner_train = {
            index
            for index in inner_train
            if self.frame.iloc[index]["toxicity"] >= 0.4
        }
        routed_inner_validation = {
            index
            for index in inner_validation
            if self.frame.iloc[index]["toxicity"] >= 0.4
        }
        self.assertEqual(stage2_train, routed_inner_train)
        self.assertEqual(stage2_validation, routed_inner_validation)

    def test_stage1_inner_validation_never_uses_outer_validation(self):
        outer = make_hierarchical_splits(
            self.frame,
            n_splits=2,
            random_state=42,
        )[0]
        inner = make_stage1_inner_split(
            self.frame,
            outer.train_indices,
            validation_fraction=0.25,
            random_state=43,
        )

        fit_indices = set(int(index) for index in inner.train_indices)
        stop_indices = set(int(index) for index in inner.validation_indices)
        outer_train = set(int(index) for index in outer.train_indices)
        outer_validation = set(int(index) for index in outer.validation_indices)

        self.assertTrue(fit_indices.isdisjoint(stop_indices))
        self.assertEqual(fit_indices | stop_indices, outer_train)
        self.assertTrue((fit_indices | stop_indices).isdisjoint(outer_validation))

        fit_gate = self.frame.iloc[inner.train_indices]["toxicity"] >= 0.4
        stop_gate = self.frame.iloc[inner.validation_indices]["toxicity"] >= 0.4
        self.assertTrue(fit_gate.any())
        self.assertTrue((~fit_gate).any())
        self.assertTrue(stop_gate.any())
        self.assertTrue((~stop_gate).any())

    def test_stage2_inner_validation_never_uses_outer_validation(self):
        outer = make_hierarchical_splits(
            self.frame,
            n_splits=2,
            random_state=42,
        )[0]
        inner = make_stage2_inner_split(
            self.frame,
            outer.stage2_train_indices,
            validation_fraction=0.25,
            random_state=43,
        )

        fit_indices = set(int(index) for index in inner.train_indices)
        stop_indices = set(int(index) for index in inner.validation_indices)
        outer_train = set(int(index) for index in outer.stage2_train_indices)
        outer_validation = set(
            int(index) for index in outer.stage2_validation_indices
        )

        self.assertTrue(fit_indices.isdisjoint(stop_indices))
        self.assertEqual(fit_indices | stop_indices, outer_train)
        self.assertTrue((fit_indices | stop_indices).isdisjoint(outer_validation))
        self.assertTrue(
            (self.frame.iloc[inner.train_indices]["toxicity"] >= 0.4).all()
        )
        self.assertTrue(
            (self.frame.iloc[inner.validation_indices]["toxicity"] >= 0.4).all()
        )


if __name__ == "__main__":
    unittest.main()
