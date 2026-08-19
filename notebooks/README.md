# Notebooks

## Canonical research notebook

- `final/HateSpeech_Final_Hierarchical.ipynb` is the final executable reproduction notebook.
- Run it top-to-bottom in a clean Google Colab runtime, preferably with GPU.
- The notebook records the Git revision, software/hardware environment, a SHA-256 fingerprint of the scientific code, repository validation, the full two-fold experiment, selected thresholds, observed metrics, and result figures.
- For the public research artifact, keep the executed outputs in the committed notebook so readers can inspect the recorded run without retraining first.

## Historical notebook

- `Hatespeech_Detection_LSTM_CNN.ipynb` preserves the original experiment and its historical outputs for traceability.
- It contains known methodological limitations and must not be used as the current benchmark.

The implementation used by the canonical notebook lives in `src/` and `scripts/`; the notebook is the reproducible analysis and reporting interface rather than a second copy of the model implementation.
