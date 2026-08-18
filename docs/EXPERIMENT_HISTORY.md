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

The gap between Stage 2 oracle Macro F1 (`0.4674`) and end-to-end Macro F1 (`0.3484`), together with Stage 1 recall (`0.4614`), identified predicted routing as a major bottleneck. High Stage 1 accuracy is therefore not treated as sufficient evidence of good routing performance.

These results remain the repaired fixed-threshold baseline for comparison. They are not the final official-test result.

## Nested threshold diagnostic — 2026-08-18

PR #10 kept the same CNN + Bi-LSTM architecture, data, soft targets, BCE losses, two outer folds, and common inner-validation protocol. It changed only prediction-time model selection inside the inner validation partition:

1. each Stage 2 label threshold was selected by inner-validation F1;
2. the Stage 1 predicted routing threshold was selected by inner-validation end-to-end Macro F1 using those Stage 2 thresholds;
3. fixed `0.4/0.5` predictions were evaluated from the same model fits for direct comparison;
4. all selected thresholds were frozen before outer-fold evaluation;
5. PR-AUC/average precision was recorded to distinguish threshold mismatch from weak discrimination.

The full-data two-fold run completed successfully in the recorded Colab Tesla T4 runtime.

### Selected thresholds

| Output | Fold 1 | Fold 2 |
| :--- | ---: | ---: |
| Stage 1 routing | 0.27 | 0.30 |
| `obscene` | 0.35 | 0.41 |
| `threat` | 0.38 | 0.40 |
| `insult` | 0.43 | 0.46 |
| `identity_attack` | 0.33 | 0.37 |
| `sexual_explicit` | 0.34 | 0.37 |

The similar operating regions across folds are evidence that the full-data thresholds are materially more stable than the one-epoch 50k smoke-run thresholds, which are diagnostic only.

### Stage 1 — same scores, different routing thresholds

| Metric | Fixed `0.40` | Nested tuned |
| :--- | ---: | ---: |
| Accuracy | 0.9257 | 0.9265 |
| Precision | 0.8340 | 0.7163 |
| Recall | 0.4178 | 0.5653 |
| F1 | 0.5567 | 0.6319 |
| Routing rate | 0.0559 | 0.0881 |
| PR-AUC / average precision | 0.7095 | 0.7095 |
| ROC-AUC | 0.9195 | 0.9195 |
| `severe_toxicity` auxiliary MAE | 0.0069 | 0.0069 |

The unchanged PR-AUC/ROC-AUC and improved F1/recall show that a substantial part of the fixed-threshold Stage 1 error was an operating-point mismatch rather than complete failure to rank routed examples.

### Stage 2 oracle

| Metric | Fixed `0.50` | Nested tuned |
| :--- | ---: | ---: |
| Macro F1 | 0.4766 | 0.5959 |
| `obscene` F1 | 0.5294 | 0.6177 |
| `threat` F1 | 0.4572 | 0.5112 |
| `insult` F1 | 0.7486 | 0.7639 |
| `identity_attack` F1 | 0.2841 | 0.5595 |
| `sexual_explicit` F1 | 0.3635 | 0.5272 |

Per-label PR-AUC/AP from the same outer predictions was `0.6262` (`obscene`), `0.5007` (`threat`), `0.8434` (`insult`), `0.5301` (`identity_attack`), and `0.5321` (`sexual_explicit`).

### End-to-end

| Metric | Fixed `0.40/0.50` | Nested tuned |
| :--- | ---: | ---: |
| Macro F1 | 0.3496 | 0.4412 |
| `obscene` F1 | 0.4608 | 0.5115 |
| `threat` F1 | 0.1991 | 0.2951 |
| `insult` F1 | 0.6327 | 0.6343 |
| `identity_attack` F1 | 0.1758 | 0.3761 |
| `sexual_explicit` F1 | 0.2797 | 0.3892 |

Nested threshold selection improved end-to-end Macro F1 from `0.3496` to `0.4412` in the same model fits, a relative increase of about 26%. The Stage 2 oracle score remains higher (`0.5959`), so propagation error is reduced but not eliminated.

The evidence therefore does **not** justify immediately replacing the architecture or adding multiple imbalance techniques. The next controlled experiment is narrower: align Stage 1 training with its operational role by adding an explicit binary routing output while preserving the existing soft `toxicity` and `severe_toxicity` outputs.

## Preservation policy

The historical notebook is intentionally kept unchanged as the reference point for future work. Repair commits should make experimental changes explicit and reviewable so that improvements and regressions can be attributed to known modifications rather than to silent changes in data handling, labeling, or evaluation.

## Next experimental stage

The next experiment should answer one question only:

> Does an explicit binary Stage 1 routing head, trained on `toxicity >= 0.4`, reduce end-to-end propagation error beyond the nested-threshold baseline of `0.4412` Macro F1?

Weighted losses, focal/asymmetric loss, oversampling, architecture replacement, transformer baselines, and fairness expansion remain deferred until this narrower hypothesis is tested.