# Reproducibility Baseline

## Current status

This repository preserves a historical CNN + Bi-LSTM experiment whose exact original runtime package versions were not recorded. The saved notebook remains the source of truth for that historical experiment, and its scientific limitations are documented in `KNOWN_ISSUES.md` and `EXPERIMENT_HISTORY.md`.

The repaired code path now has a separate validated runtime baseline:

- Python `3.12` is declared in `.python-version`;
- `requirements.txt` pins the dependency versions used by the repaired pipeline;
- TensorFlow `2.20.0` was validated with GPU detection under WSL2 after TensorFlow `2.21.0` failed to register the same NVIDIA GPU in that environment;
- the pinned environment passed dependency compatibility checks and imported the repaired ML/data stack successfully;
- `scripts/analyze_gate_coverage.py` executed against the full Civil Comments training split using the current `google/civil_comments` dataset source.

This does **not** mean the full hierarchical training run has been validated yet. The dependency pins and gate analysis are established; replacement model metrics still require a new training execution.

Two repaired code paths exist outside the historical notebook:

- `scripts/run_leakage_safe_cv.py` preserves the historical multiclass target rule while correcting cross-validation leakage; it remains an intermediate comparison baseline;
- `scripts/run_hierarchical_cv.py` implements the Issue #5 hierarchical target strategy documented in `TARGET_STRATEGY.md`.

Neither runner has produced replacement benchmark metrics that are approved for publication.

## Validated runtime

The repaired runtime currently pins:

```text
Python                  3.12
TensorFlow              2.20.0
NumPy                   2.5.2
pandas                  3.0.5
matplotlib              3.11.1
scikit-learn            1.9.0
imbalanced-learn        0.14.2
datasets                5.0.0
iterative-stratification 0.1.9
```

Create the local environment from the repository root with `uv`:

```bash
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

On POSIX/Linux shells, use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

The historical notebook predates these pins. They describe the repaired runtime, not the unrecorded original notebook environment.

## Repository-level validation

Run the structural and regression checks from the repository root:

```bash
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

These checks verify repository layout, source syntax, notebook JSON/nbformat integrity, historical leakage protections, hierarchical target extraction, preservation of overlapping labels, gate behavior, outer-fold disjointness, internal early-stopping isolation, multilabel-aware toxic-sample stratification, the namespaced Civil Comments dataset identifier, and absence of Stage 2 oversampling in the new strategy.

GitHub Actions runs the same validation on pull requests and pushes to `main`. The CI job uses Python 3.12 with the same pinned lightweight data/validation dependencies required by regression tests; it intentionally does not install TensorFlow or execute model training.

## Gate-coverage evidence

The gate-coverage analysis was executed on the full Civil Comments training split (`1,804,874` samples), with Stage 2 positives counted at `score >= 0.5`.

The selected initial routing threshold is:

```text
toxicity >= 0.4
```

At this gate, `201,476` samples are routed to Stage 2 (11.16% of train), and `533` of `126,250` samples with at least one positive Stage 2 label are missed, corresponding to 99.578% any-positive coverage.

The measured comparison across `0.50`, `0.40`, `0.35`, and `0.30`, including the `sexual_explicit` limitation, is recorded in `TARGET_STRATEGY.md` and Issue #5.

The analysis can be reproduced with:

```bash
python scripts/analyze_gate_coverage.py
```

The script defaults to the centrally defined selected gate but still accepts `--gate-threshold` for controlled comparisons.

## Hierarchical experiment command

The hierarchical runner now uses two outer folds by default for the initial experimental cycle. Within each outer training fold, 10% of the training rows are reserved as an internal early-stopping partition. The outer validation fold is not passed to Keras during fitting and is used only for final Stage 1, Stage 2 oracle, and end-to-end metrics.

Run the current full-dataset initial experiment with:

```bash
python scripts/run_hierarchical_cv.py
```

The equivalent explicit command is:

```bash
python scripts/run_hierarchical_cv.py --n-splits 2 --epochs 5
```

For smoke and debugging runs, a deterministic sample limit and smaller epoch count can be used:

```bash
python scripts/run_hierarchical_cv.py --max-samples 100000 --n-splits 2 --epochs 2
```

Metrics produced with `--max-samples` are diagnostic only and must not be reported as the full-dataset benchmark.

The number of outer folds can be increased later (for example to 3) when additional stability evidence is worth the computational cost. The exact fold count used for any reported result must be recorded with that result.

The earlier leakage-safe multiclass comparison path remains available as:

```bash
python scripts/run_leakage_safe_cv.py
```

## What this baseline does not validate

The current CI and completed gate analysis do not:

- execute either model-training runner;
- reproduce historical metrics;
- prove that the repaired pipeline resolves the confirmed overfitting;
- pin a specific immutable Civil Comments dataset revision;
- optimize routing or per-label prediction thresholds beyond the selected initial gate;
- validate fairness, robustness, or production suitability.

## Requirements for replacement metrics

Before new metrics are promoted as a validated benchmark, reproducibility evidence should include at least:

1. the declared Python and pinned dependency versions;
2. deterministic seeds where supported;
3. the dataset source and immutable revision/configuration used for the metric run;
4. preprocessing, outer split, internal early-stopping split, routing, and target configuration;
5. the measured gate-coverage report;
6. hardware/runtime notes when they materially affect execution;
7. commands required to train and evaluate from a clean environment;
8. Stage 1, Stage 2 oracle, and end-to-end metrics tied to the corresponding code revision;
9. generated evaluation artifacts tied to that same revision;
10. train/internal-validation behavior sufficient to assess the known overfitting risk.

No historical or replacement metric should be promoted to a validated benchmark until the corresponding training and evaluation procedure has been executed successfully.
