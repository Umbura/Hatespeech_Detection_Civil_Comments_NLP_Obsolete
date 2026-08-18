# Experiment History

## Historical baseline

The repository preserves the original parallel CNN + Bi-LSTM Civil Comments experiment in `notebooks/Hatespeech_Detection_LSTM_CNN.ipynb`.

The notebook contains the previously executed training and cross-validation outputs. The saved run reports approximately:

| Metric | Historical saved result |
| :--- | :--- |
| Accuracy | 0.9033 |
| Macro F1 | 0.9012 |
| Macro Recall | 0.9033 |

These values are retained for traceability only. They are not considered the validated benchmark for the repaired model.

## Current interpretation

The historical experiment has confirmed overfitting and historical evaluation concerns documented in `KNOWN_ISSUES.md`.

Two major methodological issues have now been repaired in separate code paths without rewriting the historical notebook:

1. PR #4 introduced fold-local preprocessing/resampling guarantees for the leakage-safe comparison path;
2. Issue #5 / PR #6 introduced the hierarchical two-stage target strategy that preserves overlapping fractional Civil Comments targets.

The full-train gate-coverage prerequisite for Issue #5 was subsequently executed. The initial Stage 2 routing gate was selected at `toxicity >= 0.4`, with the measured evidence recorded in `TARGET_STRATEGY.md`.

These repairs improve the experimental protocol, but they do not retroactively validate the historical metrics and they do not resolve the confirmed model overfitting.

## Preservation policy

The historical notebook is intentionally kept unchanged as the reference point for future work. Repair commits should make experimental changes explicit and reviewable so that improvements and regressions can be attributed to known modifications rather than to silent changes in data handling, labeling, or evaluation.

## Current repaired baseline state

The repaired runtime is now declared with Python 3.12 and pinned dependencies. Repository tests cover leakage protections, hierarchical target semantics, gate behavior, multilabel-aware split construction, and repository integrity.

The hierarchical runner is implemented but has not yet produced replacement benchmark metrics. Therefore the project is currently between **methodology repair** and **new baseline execution**.

## Next experimental stage

The next reviewable stages are:

1. execute the hierarchical baseline and capture Stage 1, Stage 2 oracle, and end-to-end behavior;
2. inspect train/validation behavior against the already confirmed overfitting problem;
3. repair model regularization/architecture only with evidence from the new baseline;
4. evaluate threshold calibration or imbalance strategies only as controlled follow-up experiments;
5. retrain and publish replacement metrics only after the repaired environment, data revision, and evaluation artifacts are tied to the corresponding code revision.
