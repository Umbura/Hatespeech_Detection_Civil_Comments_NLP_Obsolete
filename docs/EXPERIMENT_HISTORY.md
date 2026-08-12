# Experiment History

## Historical baseline

The repository currently preserves the original CNN + Bi-LSTM Civil Comments experiment in `notebooks/Hatespeech_Detection_LSTM_CNN.ipynb`.

The notebook contains the previously executed training and cross-validation outputs, including a historical macro F1 around 0.90. These outputs are retained for traceability only.

## Interpretation

The historical experiment has confirmed overfitting and unresolved evaluation concerns documented in `KNOWN_ISSUES.md`. Its saved metrics must not be interpreted as the validated benchmark for the repaired model.

## Preservation policy

During repository reorganization, the notebook is moved without changing its scientific content. Future experimental changes should be reviewable against this baseline so that improvements and regressions can be attributed to explicit changes.
