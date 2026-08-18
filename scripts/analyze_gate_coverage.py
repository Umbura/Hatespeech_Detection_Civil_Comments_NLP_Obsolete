"""Measure fine-grained label coverage under a toxicity routing threshold.

This script does not train a model. It answers the prerequisite question for
Issue #5: how many Stage 2 positives would be excluded if Stage 2 training and
oracle routing use ``toxicity >= gate_threshold``?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasets import load_dataset

from hate_speech_detection.target_strategy import (
    ALL_TARGET_COLUMNS,
    analyze_gate_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.5,
        help="toxicity score used to route samples to Stage 2 (default: 0.5)",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.5,
        help="fine-grained score used only to count positive labels (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_dataset("google/civil_comments", split="train").to_pandas()
    frame = frame.loc[:, ALL_TARGET_COLUMNS].reset_index(drop=True)

    report = analyze_gate_coverage(
        frame,
        gate_threshold=args.gate_threshold,
        label_threshold=args.label_threshold,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
