from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_NOTEBOOK = ROOT / "notebooks" / "Hatespeech_Detection_LSTM_CNN.ipynb"
FINAL_NOTEBOOK = ROOT / "notebooks" / "final" / "HateSpeech_Final_Hierarchical.ipynb"


class RepositoryIntegrityTests(unittest.TestCase):
    def test_expected_project_paths_exist(self) -> None:
        expected_paths = [
            ROOT / ".python-version",
            ROOT / "README.md",
            ROOT / "README_PT.md",
            ROOT / "requirements.txt",
            ROOT / "docs" / "KNOWN_ISSUES.md",
            ROOT / "docs" / "EXPERIMENT_HISTORY.md",
            ROOT / "docs" / "REPRODUCIBILITY.md",
            ROOT / "docs" / "TARGET_STRATEGY.md",
            ROOT / "results" / "FINAL_RESULTS.md",
            ROOT / "results" / "final_metrics.json",
            ROOT / "notebooks" / "README.md",
            ROOT / "src" / "hate_speech_detection" / "__init__.py",
            ROOT / "src" / "hate_speech_detection" / "cv_pipeline.py",
            ROOT / "src" / "hate_speech_detection" / "target_strategy.py",
            ROOT / "src" / "hate_speech_detection" / "hierarchical_splits.py",
            ROOT / "src" / "hate_speech_detection" / "threshold_selection.py",
            ROOT / "scripts" / "run_leakage_safe_cv.py",
            ROOT / "scripts" / "run_hierarchical_cv.py",
            ROOT / "scripts" / "run_route_head_cv.py",
            ROOT / "scripts" / "analyze_gate_coverage.py",
            HISTORICAL_NOTEBOOK,
            FINAL_NOTEBOOK,
        ]

        missing = [str(path.relative_to(ROOT)) for path in expected_paths if not path.exists()]
        self.assertEqual(missing, [], f"Missing expected repository paths: {missing}")

    def _assert_valid_notebook(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            notebook = json.load(handle)

        self.assertEqual(notebook.get("nbformat"), 4)
        self.assertIsInstance(notebook.get("cells"), list)
        self.assertGreater(len(notebook["cells"]), 0)

        for index, cell in enumerate(notebook["cells"]):
            self.assertIn(cell.get("cell_type"), {"code", "markdown", "raw"}, f"Invalid cell {index} in {path.name}")
            self.assertIn("source", cell, f"Cell {index} has no source in {path.name}")

    def test_research_notebooks_are_valid_nbformat_4_json(self) -> None:
        for notebook_path in (HISTORICAL_NOTEBOOK, FINAL_NOTEBOOK):
            self._assert_valid_notebook(notebook_path)

    def test_historical_notebook_is_not_duplicated_at_repository_root(self) -> None:
        root_notebook = ROOT / "Hatespeech_Detection_LSTM_CNN.ipynb"
        self.assertFalse(root_notebook.exists())

    def test_readmes_reference_historical_and_final_notebooks(self) -> None:
        expected_paths = (
            "notebooks/Hatespeech_Detection_LSTM_CNN.ipynb",
            "notebooks/final/HateSpeech_Final_Hierarchical.ipynb",
        )

        for readme_name in ("README.md", "README_PT.md"):
            content = (ROOT / readme_name).read_text(encoding="utf-8")
            for expected_path in expected_paths:
                self.assertIn(expected_path, content, f"{readme_name} does not reference {expected_path}")

    def test_final_metrics_match_reporting_contract(self) -> None:
        metrics = json.loads((ROOT / "results" / "final_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["project_status"], "final_research_benchmark")
        self.assertAlmostEqual(metrics["primary_run"]["end_to_end_nested_macro_f1"], 0.4412, places=4)
        self.assertAlmostEqual(metrics["replication_run"]["end_to_end_nested_macro_f1"], 0.4427, places=4)

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
