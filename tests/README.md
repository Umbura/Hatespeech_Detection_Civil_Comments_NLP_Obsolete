# Tests

The current automated tests validate repository integrity only. They intentionally do not claim model-quality, training, or evaluation coverage while the historical experimental pipeline remains under repair.

Run them from the repository root with:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Current checks cover the expected repository layout, notebook JSON/nbformat integrity, preservation of the notebook under `notebooks/`, and README references to the current notebook path.

Scientific regression tests will be added when the training and evaluation logic is extracted from the historical notebook into testable modules.
