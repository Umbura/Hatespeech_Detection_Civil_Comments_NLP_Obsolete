# Reproducibility Baseline

## Current status

This repository preserves a historical CNN + Bi-LSTM experiment whose exact runtime package versions were not recorded. The existing `requirements.txt` therefore describes the runtime dependency set but must not be interpreted as an exact environment lock.

The saved notebook remains the source of truth for the historical experiment. Its scientific limitations and evaluation concerns are documented separately in `KNOWN_ISSUES.md` and `EXPERIMENT_HISTORY.md`.

A repaired cross-validation path now exists outside the historical notebook:

- `src/hate_speech_detection/cv_pipeline.py` defines fold isolation, training-only resampling, and training-only tokenizer fitting;
- `scripts/run_leakage_safe_cv.py` applies that order while preserving the historical model architecture and target rule.

The repaired runner has not yet produced replacement benchmark metrics.

## Repository-level validation

Run the structural and leakage-regression checks from the repository root:

```bash
python -m pip install pandas numpy scikit-learn imbalanced-learn
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

These checks verify repository layout, source syntax, notebook JSON/nbformat integrity, notebook placement, documentation references, stratified fold disjointness, training-only balancing, and exclusion of validation text from tokenizer fitting.

GitHub Actions runs the same validation on pull requests and pushes to `main`. The CI job uses Python 3.12 and installs only the data/validation dependencies required by the regression tests; it does not install TensorFlow or execute model training.

## Leakage-safe experiment command

The repaired experiment runner can be started from the repository root after installing the full runtime dependencies from `requirements.txt`:

```bash
python scripts/run_leakage_safe_cv.py
```

This command downloads Civil Comments through the Hugging Face `datasets` library and performs five-fold training. It is computationally expensive and is intentionally not part of CI.

## What this baseline does not validate

The current CI does not:

- install TensorFlow or the complete training runtime;
- download or validate a specific Civil Comments dataset revision;
- execute notebook or runner training;
- reproduce historical metrics;
- prove that the corrected pipeline resolves the confirmed overfitting;
- validate the final target-label semantics;
- validate fairness, robustness, or production suitability.

## Requirements for the repaired experiment

Before replacement metrics are promoted as a validated benchmark, reproducibility evidence should include at least:

1. a supported Python version;
2. exact dependency versions or a generated lock/environment file;
3. deterministic seeds where supported;
4. the dataset source and revision/configuration used;
5. preprocessing, balancing, split, and label-generation configuration;
6. hardware/runtime notes when they materially affect execution;
7. commands required to train and evaluate from a clean environment;
8. generated evaluation artifacts tied to the corresponding code revision.

No historical metric should be promoted to a validated benchmark until that environment and evaluation procedure have been executed successfully.
