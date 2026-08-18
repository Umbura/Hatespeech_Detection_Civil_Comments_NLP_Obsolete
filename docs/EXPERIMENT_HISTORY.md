# Experiment History

## Historical baseline

The repository preserves the original parallel CNN + Bi-LSTM Civil Comments experiment in `notebooks/Hatespeech_Detection_LSTM_CNN.ipynb`.

The notebook contains the previously executed training and cross-validation outputs. The saved run reports approximately:

| Metric | Historical saved result |
| :--- | :--- |
| Accuracy | 0.9033 |
| Macro F1 | 0.9012 |
| Macro Recall | 0.9033 |

These values are retained for traceability only. They are not considered the validated benchmark for the repaired model because the historical target construction and validation procedure are not comparable to the repaired hierarchical path.

## Methodology repair

Two major methodological issues were repaired without rewriting the historical notebook:

1. PR #4 introduced fold-local preprocessing/resampling guarantees for the leakage-safe comparison path;
2. Issue #5 / PR #6 introduced the hierarchical two-stage target strategy that preserves overlapping fractional Civil Comments targets.

The full-train gate-coverage prerequisite was then executed. The ground-truth Stage 2 routing definition was selected at `toxicity >= 0.4`, with the measured evidence recorded in `TARGET_STRATEGY.md`.

PR #9 subsequently isolated EarlyStopping from the outer validation fold and reduced the initial outer CV cycle to two folds. Both stages use inner validation for model selection while the outer fold remains evaluation-only.

## Repaired two-fold hierarchical baseline — 2026-08-18

A full two-fold run was executed on the `1,804,874`-row Civil Comments training split using a Google Colab Tesla T4 and TensorFlow `2.20.0`. The architecture and losses remained the historical parallel CNN + Bi-LSTM formulation adapted to the hierarchical targets.

The fixed prediction thresholds were:

- Stage 1 predicted routing: `0.4`;
- all Stage 2 predicted labels: `0.5`.

The aggregated out-of-fold result was:

### Stage 1 toxicity gate

| Metric | Result |
| :--- | ---: |
| Accuracy | 0.9266 |
| F1 | 0.5840 |
| Recall | 0.4614 |
| `severe_toxicity` auxiliary MAE | 0.0073 |

### Stage 2 oracle — ground-truth routing

| Metric | Result |
| :--- | ---: |
| Macro F1 | 0.4674 |
| `obscene` F1 | 0.4735 |
| `threat` F1 | 0.4251 |
| `insult` F1 | 0.7451 |
| `identity_attack` F1 | 0.3359 |
| `sexual_explicit` F1 | 0.3573 |

### End-to-end — predicted Stage 1 routing

| Metric | Result |
| :--- | ---: |
| Macro F1 | 0.3484 |
| `obscene` F1 | 0.4200 |
| `threat` F1 | 0.2075 |
| `insult` F1 | 0.6360 |
| `identity_attack` F1 | 0.2114 |
| `sexual_explicit` F1 | 0.2672 |

EarlyStopping selected very early epochs: Stage 1 selected epoch 2 in fold 1 and epoch 1 in fold 2; Stage 2 selected epochs 2 and 3. This confirms that overfitting pressure appears early even after the evaluation-leakage repair.

The gap between Stage 2 oracle Macro F1 (`0.4674`) and end-to-end Macro F1 (`0.3484`), together with Stage 1 recall (`0.4614`), identifies predicted routing as a major current bottleneck. High Stage 1 accuracy is therefore not treated as sufficient evidence of good routing performance.

These results are the repaired fixed-threshold baseline for comparison. They are not yet the final project benchmark or the official-test result.

## Current threshold diagnostic

The next experiment keeps the same architecture, data, soft targets, loss functions, and outer-fold protocol. It changes only prediction-time model selection inside the inner validation partition:

1. each Stage 2 label threshold is selected by inner-validation F1;
2. the Stage 1 predicted routing threshold is selected by inner-validation **end-to-end Macro F1** using those Stage 2 thresholds;
3. fixed `0.4/0.5` predictions are still evaluated from the same trained models as a reference;
4. the selected thresholds are frozen before outer-fold evaluation;
5. PR-AUC/average precision is recorded to distinguish threshold mismatch from weak class separation.

This design deliberately avoids introducing weighted losses, focal/asymmetric loss, oversampling, architecture changes, or transformer baselines before the cheaper threshold diagnostic establishes whether those additional experiments are necessary.

## Preservation policy

The historical notebook is intentionally kept unchanged as the reference point for future work. Repair commits should make experimental changes explicit and reviewable so that improvements and regressions can be attributed to known modifications rather than to silent changes in data handling, labeling, or evaluation.

## Next experimental stage

After the nested threshold run:

1. compare fixed and nested-tuned outer-fold results from the same model fits;
2. inspect Stage 1 and per-label PR-AUC/average precision;
3. introduce a training change only if discrimination remains weak after threshold selection;
4. keep any later imbalance, regularization, architecture, or transformer experiment as a separate controlled comparison;
5. reserve the official Civil Comments test split for a final configuration rather than repeated development feedback.
