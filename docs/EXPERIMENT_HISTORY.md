# Experiment History

This document preserves the scientific evolution of the repository and separates historical evidence, repaired baselines, diagnostic experiments, and the final research benchmark.

## 1. Historical notebook

The original parallel CNN + Bi-LSTM experiment remains preserved in:

`notebooks/Hatespeech_Detection_LSTM_CNN.ipynb`

Its saved outputs report approximately:

| Metric | Historical saved result |
|---|---:|
| Accuracy | 0.9033 |
| Macro F1 | 0.9012 |
| Macro Recall | 0.9033 |

These numbers are retained only for traceability. They are not the current benchmark because the historical target construction and validation procedure contain documented methodological limitations.

## 2. Methodology repair

The repaired path was introduced incrementally:

- **PR #4** — fold-local preprocessing/resampling protections for leakage-safe evaluation;
- **PR #6** — hierarchical two-stage target strategy preserving overlapping Civil Comments labels;
- **PR #9** — inner-validation EarlyStopping separated from the outer evaluation fold;
- **PR #10** — nested prediction-threshold selection using only inner validation;
- **PR #11** — optional explicit binary routing-head experiment, retained as exploratory follow-up code.

The canonical final runner is:

`scripts/run_hierarchical_cv.py`

## 3. Ground-truth routing definition

The full `1,804,874`-row Civil Comments training split was analyzed before training the hierarchy.

The selected Stage 2 membership definition is:

```text
toxicity >= 0.4
```

At this threshold:

- 201,476 rows are routed (11.16%);
- 126,250 rows contain at least one positive Stage 2 label;
- 533 of those positives fall below the gate;
- any-positive Stage 2 coverage is **99.578%**.

This is a data/target definition. It is distinct from the learned Stage 1 prediction threshold.

## 4. Repaired fixed-threshold baseline

A full two-fold run was completed on the entire Civil Comments training split using the repaired hierarchical pipeline.

### Stage 1

| Metric | Result |
|---|---:|
| Accuracy | 0.9266 |
| F1 | 0.5840 |
| Recall | 0.4614 |
| `severe_toxicity` auxiliary MAE | 0.0073 |

### Stage 2 oracle

| Metric | Result |
|---|---:|
| Macro F1 | 0.4674 |

### End-to-end

| Metric | Result |
|---|---:|
| Macro F1 | 0.3484 |

The gap between Stage 2 oracle and end-to-end performance identified Stage 1 routing as a major source of propagation error.

## 5. Nested-threshold experiment — primary final benchmark

PR #10 kept the same architecture, data, BCE losses, two outer folds, and common inner-validation design. It changed only prediction-time threshold selection:

1. each Stage 2 label threshold was selected by inner-validation F1;
2. the Stage 1 routing threshold was selected by inner end-to-end Macro F1;
3. all thresholds were frozen before outer evaluation;
4. fixed and tuned predictions came from the same trained model fits.

### Selected thresholds

| Output | Fold 1 | Fold 2 |
|---|---:|---:|
| Stage 1 routing | 0.27 | 0.30 |
| `obscene` | 0.35 | 0.41 |
| `threat` | 0.38 | 0.40 |
| `insult` | 0.43 | 0.46 |
| `identity_attack` | 0.33 | 0.37 |
| `sexual_explicit` | 0.34 | 0.37 |

### Stage 1

| Metric | Fixed `0.40` | Nested |
|---|---:|---:|
| Accuracy | 0.9257 | 0.9265 |
| Precision | 0.8340 | 0.7163 |
| Recall | 0.4178 | 0.5653 |
| F1 | 0.5567 | 0.6319 |
| Routing rate | 0.0559 | 0.0881 |
| PR-AUC / AP | 0.7095 | 0.7095 |
| ROC-AUC | 0.9195 | 0.9195 |

The unchanged ranking metrics and improved routing F1/recall show that fixed-threshold mismatch was a meaningful contributor to Stage 1 error.

### Stage 2 oracle

| Label | Fixed F1 | Nested F1 |
|---|---:|---:|
| `obscene` | 0.5294 | 0.6177 |
| `threat` | 0.4572 | 0.5112 |
| `insult` | 0.7486 | 0.7639 |
| `identity_attack` | 0.2841 | 0.5595 |
| `sexual_explicit` | 0.3635 | 0.5272 |
| **Macro F1** | **0.4766** | **0.5959** |

### End-to-end

| Label | Fixed F1 | Nested F1 |
|---|---:|---:|
| `obscene` | 0.4608 | 0.5115 |
| `threat` | 0.1991 | 0.2951 |
| `insult` | 0.6327 | 0.6343 |
| `identity_attack` | 0.1758 | 0.3761 |
| `sexual_explicit` | 0.2797 | 0.3892 |
| **Macro F1** | **0.3496** | **0.4412** |

The nested strategy improved end-to-end Macro F1 by `+0.0916` absolute, approximately **+26.2% relative** within the same model fits.

## 6. Full-run replication

A second full execution of the same hierarchical nested-threshold protocol produced:

| Metric | Replication |
|---|---:|
| Stage 1 tuned F1 | 0.6351 |
| Stage 1 tuned recall | 0.5780 |
| Stage 1 PR-AUC / AP | 0.7086 |
| Stage 1 ROC-AUC | 0.9195 |
| Stage 2 oracle tuned Macro F1 | 0.5967 |
| **End-to-end tuned Macro F1** | **0.4427** |

The primary and repeated end-to-end values (`0.4412` and `0.4427`) support a stable performance region around `0.44`.

The project intentionally keeps `0.4412` as the official benchmark rather than selecting the slightly larger replication result after the fact.

## 7. Explicit routing-head experiment

PR #11 added an optional binary Stage 1 routing output trained directly on the `toxicity >= 0.4` decision while retaining the soft `toxicity` and `severe_toxicity` outputs.

A 50k-row, one-epoch smoke run completed successfully and showed the experimental path functioning. Those sampled metrics are diagnostic only and are not used as final scientific evidence.

Because the nested-threshold baseline already provides a stable, defensible academic result, a further full-data route-head optimization cycle is outside the finalized project scope.

The script remains available as:

`scripts/run_route_head_cv.py`

## 8. Final project decision

For the completed academic delivery:

- **official end-to-end Macro F1:** `0.4412`;
- **replication end-to-end Macro F1:** `0.4427`;
- **Stage 2 oracle Macro F1:** `0.5959`;
- **canonical runner:** `scripts/run_hierarchical_cv.py`;
- **canonical notebook:** `notebooks/final/HateSpeech_Final_Hierarchical.ipynb`.

The project does not claim state-of-the-art or production readiness. The scientific contribution is the repaired hierarchical evaluation and the demonstrated impact of validation-only threshold selection on a highly imbalanced cascade.

Machine-readable and presentation-ready metrics are stored under `results/`.
