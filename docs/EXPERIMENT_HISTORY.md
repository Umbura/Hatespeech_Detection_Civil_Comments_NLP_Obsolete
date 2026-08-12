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

The historical experiment has confirmed overfitting and unresolved evaluation concerns documented in `KNOWN_ISSUES.md`.

The current preprocessing and balancing order occurs before fold creation, and the target-label construction also requires explicit review. Because these issues affect confidence in the evaluation protocol, the saved metrics must remain labeled as historical results until the pipeline is repaired and re-evaluated.

## Preservation policy

The historical notebook is intentionally kept unchanged as the reference point for future work. Repair commits should make experimental changes explicit and reviewable so that improvements and regressions can be attributed to known modifications rather than to silent changes in data handling, labeling, or evaluation.

## Next experimental stage

The planned repair stage is expected to address, in separate reviewable changes:

1. cross-validation preprocessing and resampling order;
2. target-label formulation;
3. confirmed model overfitting;
4. retraining and final evaluation;
5. publication of a new validated benchmark only after those checks are complete.
