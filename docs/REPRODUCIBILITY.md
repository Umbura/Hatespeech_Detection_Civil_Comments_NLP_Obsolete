# Reproducibility Baseline

## Current status

This repository preserves a historical CNN + Bi-LSTM experiment whose exact original runtime package versions were not recorded. The saved notebook remains the source of truth for that historical experiment, and its scientific limitations are documented in `KNOWN_ISSUES.md` and `EXPERIMENT_HISTORY.md`.

The repaired code path now has a separate reproducibility baseline:

- Python `3.12` is declared in `.python-version`;
- `requirements.txt` pins the dependency versions used by the repaired local pipeline;
- TensorFlow `2.20.0` was validated with GPU detection under WSL2 after TensorFlow `2.21.0` failed to register the same NVIDIA GPU in that environment;
- the pinned local environment passed dependency compatibility checks;
- `scripts/analyze_gate_coverage.py` executed against the full Civil Comments training split using the current `google/civil_comments` dataset source;
- a full two-fold hierarchical fixed-threshold baseline was executed on Google Colab on 2026-08-18;
- the current runner adds nested prediction-threshold selection without changing the architecture or training loss.

Two repaired code paths exist outside the historical notebook:

- `scripts/run_leakage_safe_cv.py` preserves the historical multiclass target rule while correcting cross-validation leakage; it remains an intermediate comparison baseline;
- `scripts/run_hierarchical_cv.py` implements the hierarchical target strategy documented in `TARGET_STRATEGY.md` and is the canonical current experimental runner.

## Validated local runtime

The repaired local runtime currently pins:

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

## Recorded Colab baseline runtime

The completed two-fold fixed-threshold hierarchical baseline was executed in a Google Colab Tesla T4 session. The notebook reported:

```text
TensorFlow              2.20.0
NumPy                   2.0.2
pandas                  2.2.3
scikit-learn            1.6.1
GPU                     Tesla T4
```

This Colab runtime is recorded because it produced the fixed-threshold baseline in `EXPERIMENT_HISTORY.md`. It is not claimed to be byte-for-byte identical to the pinned local environment. Any final reported result should record the exact runtime that generated it.

## Repository-level validation

Run the structural and regression checks from the repository root:

```bash
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

These checks verify repository layout, source syntax, notebook JSON/nbformat integrity, historical leakage protections, hierarchical target extraction, preservation of overlapping labels, gate behavior, outer-fold disjointness, common inner-validation isolation, multilabel-aware routed-sample stratification, threshold-selection behavior, the namespaced Civil Comments dataset identifier, and absence of Stage 2 oversampling in the hierarchical strategy.

GitHub Actions runs the same validation on pull requests and pushes to `main`. The CI job uses Python 3.12 with the lightweight data/validation dependencies required by regression tests; it intentionally does not install TensorFlow or execute model training.

## Gate-coverage evidence

The gate-coverage analysis was executed on the full Civil Comments training split (`1,804,874` samples), with Stage 2 positives counted at `score >= 0.5`.

The selected ground-truth routing definition is:

```text
toxicity >= 0.4
```

At this gate, `201,476` samples are routed to Stage 2 (11.16% of train), and `533` of `126,250` samples with at least one positive Stage 2 label are missed, corresponding to 99.578% any-positive coverage.

The measured comparison across `0.50`, `0.40`, `0.35`, and `0.30`, including the `sexual_explicit` limitation, is recorded in `TARGET_STRATEGY.md`.

The analysis can be reproduced with:

```bash
python scripts/analyze_gate_coverage.py
```

The ground-truth gate is a target/data definition. The predicted Stage 1 routing threshold is now treated separately and selected from model outputs inside nested validation.

## Hierarchical experiment protocol

The hierarchical runner uses two outer folds by default for the current experimental cycle. Within each outer training fold, 10% of the rows are reserved as one common inner validation partition shared by both stages.

The common inner split is built hierarchically:

- routed rows use iterative multilabel stratification;
- non-routed rows are split separately;
- Stage 2 fit/validation rows are exactly the routed subsets of the same common Stage 1 inner partitions.

The inner validation partition is used for:

1. EarlyStopping for Stage 1;
2. EarlyStopping for Stage 2;
3. per-label Stage 2 prediction-threshold selection by F1;
4. Stage 1 predicted-routing-threshold selection by **end-to-end Macro F1**.

The threshold grid is `0.01` through `0.99` in steps of `0.01`. Stage 2 ties prefer the threshold closest to the fixed `0.5` reference. Stage 1 ties in end-to-end Macro F1 prefer higher gate recall, then the threshold closest to the ground-truth `0.4` reference.

The outer validation fold is not used for fitting, EarlyStopping, or threshold selection.

## Fixed versus nested-tuned evaluation

The same trained models produce both references in one run:

### Fixed reference

```text
Stage 1 predicted routing threshold = 0.4
Stage 2 predicted label thresholds  = 0.5 for all five labels
```

### Nested tuned reference

```text
Stage 2 thresholds = selected per label on routed inner validation
Stage 1 threshold  = selected on full inner validation by end-to-end Macro F1
```

This lets threshold effects be measured without training a second architecture or changing the loss.

The runner additionally reports:

- Stage 1 precision, recall, F1, routing rate, PR-AUC/average precision, ROC-AUC, and auxiliary `severe_toxicity` MAE;
- Stage 2 oracle Macro F1, per-label F1, and per-label PR-AUC/average precision;
- end-to-end Macro F1 and per-label F1;
- the threshold selected in each outer fold and its inner-validation score.

## Commands

Full current two-fold experiment:

```bash
python scripts/run_hierarchical_cv.py --n-splits 2 --epochs 5
```

Smoke/debug run:

```bash
python scripts/run_hierarchical_cv.py --max-samples 50000 --n-splits 2 --epochs 1
```

Metrics produced with `--max-samples` are diagnostic only and must not be reported as the full-dataset benchmark.

The number of outer folds can be increased later when additional stability evidence is worth the computational cost. The exact fold count used for any reported result must be recorded with that result.

The earlier leakage-safe multiclass comparison path remains available as:

```bash
python scripts/run_leakage_safe_cv.py
```

## Current fixed-threshold evidence

The completed full-data two-fold fixed-threshold baseline is recorded in `EXPERIMENT_HISTORY.md`. Its main results were:

```text
Stage 1 recall                  0.4614
Stage 1 F1                      0.5840
Stage 2 oracle Macro F1         0.4674
End-to-end Macro F1             0.3484
```

Those values motivate the nested threshold diagnostic. They do not establish that threshold tuning will improve the outer-fold result.

## What remains unvalidated

The current evidence does not yet:

- provide outer-fold results from the new nested threshold-selection path;
- prove that threshold selection closes the Stage 2 oracle/end-to-end gap;
- resolve the confirmed early overfitting behavior;
- pin a specific immutable Civil Comments dataset revision;
- validate fairness, robustness, or production suitability;
- provide a final evaluation on the untouched official Civil Comments test split.

## Requirements for final replacement metrics

Before new metrics are promoted as the final project benchmark, reproducibility evidence should include at least:

1. the exact Python and dependency versions of the execution environment;
2. deterministic seeds where supported;
3. the dataset source and immutable revision/configuration used for the metric run;
4. preprocessing, outer split, common inner split, routing, target, and threshold-selection configuration;
5. the measured gate-coverage report;
6. hardware/runtime notes;
7. commands required to train and evaluate;
8. Stage 1, Stage 2 oracle, and end-to-end metrics tied to the corresponding code revision;
9. selected thresholds and diagnostic PR-AUC/average-precision values;
10. train/inner-validation behavior sufficient to assess overfitting;
11. a final official-test evaluation performed only after the development configuration is frozen.
