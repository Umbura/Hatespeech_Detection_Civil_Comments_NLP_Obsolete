# Reproducibility Baseline

## Final project status

The canonical research pipeline is implemented by:

```text
scripts/run_hierarchical_cv.py
```

The canonical final notebook is:

```text
notebooks/final/HateSpeech_Final_Hierarchical.ipynb
```

The official repository benchmark is:

```text
End-to-end Macro F1 = 0.4412
```

A repeated full run produced `0.4427`, which is reported as replication evidence.

The historical notebook remains preserved separately and is not the source of the final benchmark.

## Runtime definitions

### Pinned local runtime

`requirements.txt` currently defines the repaired local environment:

```text
Python                   3.12
TensorFlow               2.20.0
NumPy                    2.5.2
pandas                   3.0.5
matplotlib               3.11.1
scikit-learn             1.9.0
imbalanced-learn         0.14.2
datasets                 5.0.0
iterative-stratification 0.1.9
```

Create it with:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

### Recorded Colab full-run runtime

The completed full-data hierarchical experiments were executed in a Google Colab Tesla T4 session with:

```text
TensorFlow       2.20.0
NumPy            2.0.2
pandas           2.2.3
scikit-learn     1.6.1
GPU              Tesla T4
```

The local pinned environment and the recorded Colab environment are not claimed to be byte-for-byte identical. Runtime differences must be recorded when reproducing results.

## Dataset

The experiment uses:

```text
google/civil_comments
```

The training split used for the hierarchical cross-validation contains:

```text
1,804,874 rows
```

The repository does not currently pin an immutable dataset revision, which remains a reproducibility limitation.

## Ground-truth hierarchy

Stage 2 membership is defined by:

```text
toxicity >= 0.4
```

Binary evaluation of subtype targets uses:

```text
score >= 0.5
```

Full-train gate analysis at the selected definition produced:

```text
Routed rows                         201,476
Any-positive Stage 2 rows          126,250
Any-positive rows missed               533
Any-positive Stage 2 coverage       99.578%
```

Reproduce the analysis with:

```bash
python scripts/analyze_gate_coverage.py
```

## Hierarchical validation protocol

The final runner uses two outer folds.

Within each outer-training partition, one common inner validation partition is created for both stages.

The inner partition is used for:

1. Stage 1 EarlyStopping;
2. Stage 2 EarlyStopping;
3. Stage 2 per-label threshold selection by F1;
4. Stage 1 routing-threshold selection by end-to-end Macro F1.

Routed inner rows use iterative multilabel stratification. Non-routed rows are split separately and recombined into the common Stage 1 inner partition.

The outer fold is evaluation-only and is not used for model fitting, EarlyStopping, or threshold selection.

## Threshold selection

The candidate grid is:

```text
0.01 ... 0.99
```

in steps of `0.01`.

Stage 2 ties prefer the threshold closest to `0.5`.

Stage 1 ties prefer:

1. higher gate recall;
2. then the threshold closest to the `0.4` ground-truth routing reference.

Selected Stage 1 routing thresholds in the primary full run were:

```text
Fold 1: 0.27
Fold 2: 0.30
```

## Primary full-data evidence

### Stage 1

```text
Fixed F1                 0.5567
Nested F1                0.6319
Nested recall            0.5653
PR-AUC / AP              0.7095
ROC-AUC                  0.9195
```

### Stage 2 oracle

```text
Fixed Macro F1           0.4766
Nested Macro F1          0.5959
```

### End-to-end

```text
Fixed Macro F1           0.3496
Nested Macro F1          0.4412
```

The fixed-to-nested improvement is `+0.0916` absolute, approximately `+26.2%` relative.

## Replication evidence

A second full execution of the same protocol produced:

```text
Stage 1 nested F1        0.6351
Stage 1 nested recall    0.5780
Stage 1 PR-AUC / AP      0.7086
Stage 1 ROC-AUC          0.9195
Stage 2 oracle Macro F1  0.5967
End-to-end Macro F1      0.4427
```

The close end-to-end values (`0.4412` and `0.4427`) are reported as evidence that the observed performance region is stable across the two recorded full runs.

The primary benchmark remains `0.4412` to avoid choosing the larger result after observing both runs.

## Repository validation

From the repository root:

```bash
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

GitHub Actions executes the repository's lightweight structural/regression validation. CI does not perform full TensorFlow model training.

## Smoke run

Use only for execution-path diagnostics:

```bash
python scripts/run_hierarchical_cv.py \
  --max-samples 50000 \
  --n-splits 2 \
  --epochs 1
```

Metrics from sampled smoke runs must not be reported as full-data scientific results.

## Full reproduction command

```bash
python scripts/run_hierarchical_cv.py \
  --n-splits 2 \
  --epochs 5
```

GPU execution is recommended because CPU training is substantially slower.

## Final notebook workflow

`notebooks/final/HateSpeech_Final_Hierarchical.ipynb` provides:

- the research question;
- target strategy;
- architecture summary;
- validated result tables;
- primary vs replication distinction;
- interpretation and limitations;
- gate-analysis command;
- smoke command;
- full reproduction command.

## What is not claimed

The final project evidence does not establish:

- an official frozen test-set benchmark;
- state-of-the-art performance;
- fairness across identity subgroups;
- adversarial or cross-domain robustness;
- production latency, scalability, or reliability;
- resolved overfitting;
- byte-identical determinism across CPU/GPU and dependency environments.

Those items are explicitly outside the completed academic scope and may be addressed only in future work.
