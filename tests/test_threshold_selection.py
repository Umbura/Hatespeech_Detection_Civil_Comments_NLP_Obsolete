from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hate_speech_detection.threshold_selection import (
    apply_label_thresholds,
    select_label_thresholds,
    select_routing_threshold,
)


class ThresholdSelectionTests(unittest.TestCase):
    def test_apply_label_thresholds_uses_one_threshold_per_output(self):
        probabilities = np.array(
            [
                [0.30, 0.70],
                [0.60, 0.40],
            ]
        )
        thresholds = np.array([0.50, 0.60])

        predicted = apply_label_thresholds(probabilities, thresholds)

        np.testing.assert_array_equal(
            predicted,
            np.array([[0, 1], [1, 0]], dtype=np.int8),
        )

    def test_label_threshold_selection_maximizes_inner_f1(self):
        y_true = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.int8,
        )
        probabilities = np.array(
            [
                [0.10, 0.20],
                [0.20, 0.55],
                [0.40, 0.30],
                [0.45, 0.60],
            ]
        )

        selection = select_label_thresholds(
            y_true,
            probabilities,
            candidates=[0.30, 0.40, 0.50, 0.60],
            reference_threshold=0.50,
        )

        np.testing.assert_allclose(selection.thresholds, [0.40, 0.50])
        np.testing.assert_allclose(selection.f1_scores, [1.0, 1.0])

    def test_routing_threshold_selection_uses_end_to_end_macro_f1(self):
        gate_true = np.array([0, 0, 1, 1], dtype=np.int8)
        stage1_probabilities = np.array([0.10, 0.30, 0.35, 0.45])
        stage2_true = np.array([[0], [0], [1], [1]], dtype=np.int8)
        stage2_probabilities = np.array([[0.10], [0.90], [0.80], [0.90]])

        selection = select_routing_threshold(
            gate_true,
            stage1_probabilities,
            stage2_true,
            stage2_probabilities,
            np.array([0.50]),
            candidates=[0.30, 0.40, 0.50],
            reference_threshold=0.40,
        )

        self.assertAlmostEqual(selection.threshold, 0.30)
        self.assertAlmostEqual(selection.macro_f1, 0.80)
        self.assertAlmostEqual(selection.gate_recall, 1.0)
        self.assertAlmostEqual(selection.routing_rate, 0.75)

    def test_threshold_candidates_reject_invalid_values(self):
        y_true = np.array([[0], [1]], dtype=np.int8)
        probabilities = np.array([[0.2], [0.8]])

        with self.assertRaises(ValueError):
            select_label_thresholds(
                y_true,
                probabilities,
                candidates=[0.0, 0.5],
            )


if __name__ == "__main__":
    unittest.main()
