from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Hatespeech_Detection_LSTM_CNN.ipynb"


class RepositoryIntegrityTests(unittest.TestCase):
    def test_expected_project_paths_exist(self) -> None:
        expected_paths = [
            ROOT / ".python-version",
            ROOT / "AGENTS.md",
            ROOT / "README.md",
            ROOT / "README_PT.md",
            ROOT / "requirements.txt",
            ROOT / "docs" / "KNOWN_ISSUES.md",
            ROOT / "docs" / "EXPERIMENT_HISTORY.md",
            ROOT / "docs" / "REPRODUCIBILITY.md",
            ROOT / "docs" / "TARGET_STRATEGY.md",
            ROOT / "src" / "hate_speech_detection" / "__init__.py",
            ROOT / "src" / "hate_speech_detection" / "cv_pipeline.py",
            ROOT / "src" / "hate_speech_detection" / "target_strategy.py",
            ROOT / "src" / "hate_speech_detection" / "hierarchical_splits.py",
            ROOT / "scripts" / "run_leakage_safe_cv.py",
            ROOT / "scripts" / "run_hierarchical_cv.py",
            ROOT / "scripts" / "analyze_gate_coverage.py",
            NOTEBOOK,
        ]

        missing = [str(path.relative_to(ROOT)) for path in expected_paths if not path.exists()]
        self.assertEqual(missing, [], f"Missing expected repository paths: {missing}")

    def test_historical_notebook_is_valid_nbformat_4_json(self) -> None:
        with NOTEBOOK.open("r", encoding="utf-8") as handle:
            notebook = json.load(handle)

        self.assertEqual(notebook.get("nbformat"), 4)
        self.assertIsInstance(notebook.get("cells"), list)
        self.assertGreater(len(notebook["cells"]), 0)

        for index, cell in enumerate(notebook["cells"]):
            self.assertIn(cell.get("cell_type"), {"code", "markdown", "raw"}, f"Invalid cell {index}")
            self.assertIn("source", cell, f"Cell {index} has no source")

    def test_historical_notebook_is_not_duplicated_at_repository_root(self) -> None:
        root_notebook = ROOT / "Hatespeech_Detection_LSTM_CNN.ipynb"
        self.assertFalse(root_notebook.exists())

    def test_readmes_reference_current_notebook_path(self) -> None:
        expected_path = "notebooks/Hatespeech_Detection_LSTM_CNN.ipynb"

        for readme_name in ("README.md", "README_PT.md"):
            content = (ROOT / readme_name).read_text(encoding="utf-8")
            self.assertIn(expected_path, content, f"{readme_name} does not reference the current notebook path")

    def test_runtime_scripts_use_namespaced_civil_comments_id(self) -> None:
        script_names = (
            "analyze_gate_coverage.py",
            "run_hierarchical_cv.py",
            "run_leakage_safe_cv.py",
        )

        for script_name in script_names:
            content = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
            self.assertIn('load_dataset("google/civil_comments"', content)
            self.assertNotIn('load_dataset("civil_comments"', content)


if __name__ == "__main__":
    unittest.main()
