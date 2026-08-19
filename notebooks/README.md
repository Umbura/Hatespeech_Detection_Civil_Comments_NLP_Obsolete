# Notebooks

## Canonical research notebook

- `final/HateSpeech_Final_Hierarchical.ipynb` is the final scientific-initiation artifact.
- It preserves the completed full-data reproduction, including the executed commit, gate coverage, fold-level thresholds, Stage 1 metrics, Stage 2 oracle metrics, and end-to-end results.
- The published reproduction obtained **0.4423 end-to-end Macro F1**, consistent with the documented primary benchmark (`0.4412`) and previous replication (`0.4427`).
- The implementation used by the notebook remains in `src/` and `scripts/`; the notebook is the readable computational narrative and reproducibility artifact.

## Historical notebook

- `Hatespeech_Detection_LSTM_CNN.ipynb` preserves the original experiment and historical outputs for traceability.
- It contains known methodological limitations and must not be used as the current benchmark.

For academic reporting, prefer the canonical notebook together with `results/FINAL_RESULTS.md` and `results/final_metrics.json`.
