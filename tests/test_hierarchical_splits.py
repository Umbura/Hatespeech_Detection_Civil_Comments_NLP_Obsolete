from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from hate_speech_detection.hierarchical_splits import make_hierarchical_splits
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


if __name__ == "__main__":
    unittest.main()
