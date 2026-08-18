from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from hate_speech_detection.target_strategy import (
    DEFAULT_GATE_THRESHOLD,
    STAGE1_TARGET_COLUMNS,
    STAGE2_TARGET_COLUMNS,
    analyze_gate_coverage,
    get_stage1_targets,
    get_stage2_binary_targets,
    get_stage2_gate_mask,
    get_stage2_targets,
)


class TargetStrategyTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "toxicity": [0.90, 0.40, 0.70, 0.10],
                "severe_toxicity": [0.20, 0.00, 0.60, 0.00],
                "obscene": [0.80, 0.00, 0.20, 0.00],
                "threat": [0.10, 0.70, 0.60, 0.00],
                "insult": [0.90, 0.80, 0.10, 0.00],
                "identity_attack": [0.00, 0.00, 0.70, 0.00],
                "sexual_explicit": [0.00, 0.00, 0.00, 0.00],
            }
        )

    def test_soft_targets_preserve_fractional_scores_and_overlaps(self):
        stage1 = get_stage1_targets(self.frame)
        stage2 = get_stage2_targets(self.frame)

        self.assertEqual(stage1.shape, (4, len(STAGE1_TARGET_COLUMNS)))
        self.assertEqual(stage2.shape, (4, len(STAGE2_TARGET_COLUMNS)))
        self.assertAlmostEqual(float(stage1[0, 0]), 0.90, places=6)

        insult_index = STAGE2_TARGET_COLUMNS.index("insult")
        obscene_index = STAGE2_TARGET_COLUMNS.index("obscene")
        self.assertAlmostEqual(float(stage2[0, insult_index]), 0.90, places=6)
        self.assertAlmostEqual(float(stage2[0, obscene_index]), 0.80, places=6)

        binary = get_stage2_binary_targets(self.frame, label_threshold=0.5)
        self.assertEqual(int(binary[0, insult_index]), 1)
        self.assertEqual(int(binary[0, obscene_index]), 1)

    def test_default_gate_uses_selected_threshold(self):
        self.assertEqual(DEFAULT_GATE_THRESHOLD, 0.4)
        mask = get_stage2_gate_mask(self.frame)
        np.testing.assert_array_equal(mask, np.array([True, True, True, False]))

    def test_gate_uses_toxicity_only_for_routing(self):
        mask = get_stage2_gate_mask(self.frame, gate_threshold=0.5)
        np.testing.assert_array_equal(mask, np.array([True, False, True, False]))

    def test_gate_coverage_reports_subtype_positives_below_gate(self):
        report = analyze_gate_coverage(
            self.frame,
            gate_threshold=0.5,
            label_threshold=0.5,
        )

        self.assertEqual(report["per_label"]["threat"]["positive_count"], 2)
        self.assertEqual(report["per_label"]["threat"]["missed_count"], 1)
        self.assertEqual(report["per_label"]["insult"]["positive_count"], 2)
        self.assertEqual(report["per_label"]["insult"]["missed_count"], 1)
        self.assertEqual(report["any_stage2_positive"]["positive_count"], 3)
        self.assertEqual(report["any_stage2_positive"]["missed_count"], 1)

    def test_non_toxic_is_not_a_model_target(self):
        self.assertNotIn("non_toxic", STAGE1_TARGET_COLUMNS)
        self.assertNotIn("non_toxic", STAGE2_TARGET_COLUMNS)

    def test_invalid_scores_are_rejected(self):
        invalid = self.frame.copy()
        invalid.loc[0, "toxicity"] = 1.2
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            get_stage1_targets(invalid)


if __name__ == "__main__":
    unittest.main()
