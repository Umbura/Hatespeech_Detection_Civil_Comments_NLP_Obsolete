# Reproducibility Baseline

## Current status

This repository preserves a historical CNN + Bi-LSTM experiment whose exact runtime package versions were not recorded. The existing `requirements.txt` therefore describes the runtime dependency set but must not be interpreted as an exact environment lock.

The saved notebook remains the source of truth for the historical experiment. Its scientific limitations and evaluation concerns are documented separately in `KNOWN_ISSUES.md` and `EXPERIMENT_HISTORY.md`.

Two repaired code paths now exist outside the historical notebook:

- `scripts/run_leakage_safe_cv.py` preserves the historical multiclass target rule while correcting cross-validation leakage; it remains an intermediate comparison baseline;
- `scripts/run_hierarchical_cv.py` implements the Issue #5 hierarchical target strategy documented in `TARGET_STRATEGY.md`.

Neither runner has produced replacement benchmark metrics that are approved for publication.

## Repository-level validation

Run the structural and regression checks from the repository root:

```bash
python -m pip install pandas numpy scikit-learn imbalanced-learn 'iterative-stratification>=0.1.9'
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

These checks verify repository layout, source syntax, notebook JSON/nbformat integrity, historical leakage protections, hierarchical target extraction, preservation of overlapping labels, gate-coverage accounting, fold disjointness, multilabel-aware toxic-sample stratification, and absence of Stage 2 oversampling in the new strategy.

GitHub Actions runs the same validation on pull requests and pushes to `main`. The CI job uses Python 3.12 and installs only the data/validation dependencies required by regression tests; it does not install TensorFlow or execute model training.

## Gate-coverage analysis

Before the initial `toxicity >= 0.5` routing threshold is described as empirically safe, run:

```bash
python scripts/analyze_gate_coverage.py
```

The command downloads Civil Comments through Hugging Face `datasets` and reports how many fine-grained positives would be excluded by the ground-truth gate. The measured output should be preserved with the experiment evidence.

## Hierarchical experiment command

After installing the full runtime dependencies from `requirements.txt`, the hierarchical runner can be started with:

```bash
python scripts/run_hierarchical_cv.py
```

It performs five-fold training and separately reports Stage 1 routing, Stage 2 oracle, and end-to-end subtype metrics. It is computationally expensive and is intentionally not part of CI.

The earlier leakage-safe multiclass comparison path remains available as:

```bash
python scripts/run_leakage_safe_cv.py
```

## What this baseline does not validate

The current CI does not:

- install TensorFlow or the complete training runtime;
- download or pin a specific Civil Comments dataset revision;
- execute either training runner;
- reproduce historical metrics;
- quantify real-dataset gate coverage;
- prove that the corrected pipeline resolves the confirmed overfitting;
- optimize routing or per-label thresholds;
- validate fairness, robustness, or production suitability.

## Requirements for replacement metrics

Before new metrics are promoted as a validated benchmark, reproducibility evidence should include at least:

1. a supported Python version;
2. exact dependency versions or a generated lock/environment file;
3. deterministic seeds where supported;
4. the dataset source and revision/configuration used;
5. preprocessing, split, routing, and target configuration;
6. the measured gate-coverage report;
7. hardware/runtime notes when they materially affect execution;
8. commands required to train and evaluate from a clean environment;
9. Stage 1, Stage 2 oracle, and end-to-end metrics tied to the corresponding code revision;
10. generated evaluation artifacts tied to that same revision.

No historical or replacement metric should be promoted to a validated benchmark until the corresponding environment and evaluation procedure have been executed successfully.
