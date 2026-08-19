<div align="center">

# Hybrid Hate Speech Detection

### Final academic CNN + Bi-LSTM hierarchical benchmark on Civil Comments

[![Read in Portuguese](https://img.shields.io/badge/Read%20in-Portuguese-2ea44f?style=for-the-badge&logo=google-translate&logoColor=white)](README_PT.md)

<p>
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TensorFlow-2.20-orange" alt="TensorFlow 2.20">
  <img src="https://img.shields.io/badge/Status-Research_Complete-brightgreen" alt="Status: Research Complete">
  <a href="https://colab.research.google.com/github/Umbura/Hatespeech_Detection_Civil_Comments_NLP_Obsolete/blob/main/notebooks/final/HateSpeech_Final_Hierarchical.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Final Notebook In Colab">
  </a>
</p>

</div>

---

## Overview

This repository contains an academic NLP experiment for toxicity and hate-speech-related classification using the **Civil Comments** dataset and a parallel **CNN + Bi-LSTM** encoder.

The original notebook is preserved as historical evidence, but the final research result comes from a repaired **hierarchical two-stage pipeline** with fold-local preprocessing, inner-validation EarlyStopping, and nested prediction-threshold selection.

The project is considered complete for its current academic scope. It does **not** claim state-of-the-art performance or production readiness.

---

## Final Research Benchmark

The official repository benchmark is the controlled full-data PR #10 nested-threshold experiment:

| Evaluation | Result |
|---|---:|
| Stage 1 tuned F1 | 0.6319 |
| Stage 1 tuned recall | 0.5653 |
| Stage 1 PR-AUC / AP | 0.7095 |
| Stage 1 ROC-AUC | 0.9195 |
| Stage 2 oracle tuned Macro F1 | 0.5959 |
| **End-to-end tuned Macro F1** | **0.4412** |

A second full run of the same protocol produced **0.4427 end-to-end Macro F1**. The replication is kept as stability evidence; `0.4412` remains the primary benchmark rather than retroactively selecting the slightly higher run.

Detailed metrics: [results/FINAL_RESULTS.md](results/FINAL_RESULTS.md) and [results/final_metrics.json](results/final_metrics.json).

---

## Main Finding

The same trained models were evaluated with fixed and nested-selected thresholds:

| End-to-end evaluation | Macro F1 |
|---|---:|
| Fixed routing `0.40` / labels `0.50` | 0.3496 |
| **Nested-selected thresholds** | **0.4412** |

This is an absolute gain of `+0.0916`, approximately **+26.2% relative**.

The main scientific finding is therefore not simply the final F1 value. The experiment shows that fixed probability thresholds were materially mismatched to the imbalanced hierarchical task. Selecting thresholds only on inner validation recovered substantial performance without replacing the core CNN + Bi-LSTM architecture.

---

## Hierarchical Formulation

### Stage 1 — routing

Stage 1 learns the fractional Civil Comments outputs:

- `toxicity`;
- auxiliary `severe_toxicity`.

The ground-truth routing definition is:

```text
toxicity >= 0.4
```

On the full `1,804,874`-row training split this routes 201,476 rows and covers **99.578%** of samples containing at least one positive Stage 2 subtype.

### Stage 2 — multilabel classification

Routed comments are classified independently for:

- `obscene`;
- `threat`;
- `insult`;
- `identity_attack`;
- `sexual_explicit`.

Overlapping labels are preserved instead of being collapsed into one order-dependent class.

---

## Validation Protocol

The final experiment uses two outer folds. For each fold:

1. the outer fold is evaluation-only;
2. the remaining data is divided into fit and common inner validation partitions;
3. learned preprocessing is fit only on training data;
4. Stage 1 and Stage 2 EarlyStopping use only inner validation;
5. Stage 2 label thresholds are selected only on inner validation;
6. the Stage 1 routing threshold is selected by inner end-to-end Macro F1;
7. thresholds are frozen before outer evaluation.

This prevents the outer evaluation fold from influencing training, EarlyStopping, or threshold selection.

---

## End-to-End Per-Label Results

| Label | F1 |
|---|---:|
| obscene | 0.5115 |
| threat | 0.2951 |
| insult | 0.6343 |
| identity_attack | 0.3761 |
| sexual_explicit | 0.3892 |
| **Macro F1** | **0.4412** |

Stage 2 oracle Macro F1 is `0.5959`; it must not be presented as system-level performance because it assumes perfect routing.

---

## Historical Experiment

`notebooks/Hatespeech_Detection_LSTM_CNN.ipynb` preserves the original experiment and its saved metrics around Macro F1 `~0.90`.

Those historical metrics are **not** the current benchmark because the original target construction and validation procedure contain documented methodological limitations. The notebook remains unchanged for traceability.

The canonical academic notebook is:

`notebooks/final/HateSpeech_Final_Hierarchical.ipynb`

---

## Repository Structure

```text
.
├── README.md
├── README_PT.md
├── requirements.txt
├── notebooks/
│   ├── README.md
│   ├── Hatespeech_Detection_LSTM_CNN.ipynb     # historical
│   └── final/
│       └── HateSpeech_Final_Hierarchical.ipynb # canonical research notebook
├── results/
│   ├── FINAL_RESULTS.md
│   └── final_metrics.json
├── scripts/
│   ├── analyze_gate_coverage.py
│   ├── run_hierarchical_cv.py
│   ├── run_leakage_safe_cv.py
│   └── run_route_head_cv.py
├── src/
│   └── hate_speech_detection/
├── tests/
├── docs/
│   ├── EXPERIMENT_HISTORY.md
│   ├── KNOWN_ISSUES.md
│   ├── REPRODUCIBILITY.md
│   └── TARGET_STRATEGY.md
└── assets/
```

`run_hierarchical_cv.py` is the canonical final experimental runner. `run_route_head_cv.py` is retained as an exploratory follow-up and is not the source of the final benchmark.

---

## Reproduce the Final Experiment

Create the local environment from the repository root:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

Run repository validation:

```bash
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

Check full-dataset gate coverage:

```bash
python scripts/analyze_gate_coverage.py
```

Diagnostic smoke run:

```bash
python scripts/run_hierarchical_cv.py --max-samples 50000 --n-splits 2 --epochs 1
```

Full experiment:

```bash
python scripts/run_hierarchical_cv.py --n-splits 2 --epochs 5
```

Metrics from `--max-samples` are diagnostic only and must not be reported as the full-data benchmark.

---

## Limitations

- two outer folds were used because full training is computationally expensive;
- early overfitting pressure remains visible;
- routing errors propagate permanently into Stage 2;
- `sexual_explicit` is the subtype most affected by the `toxicity >= 0.4` ground-truth gate;
- fairness and subgroup robustness were not validated;
- the final repository benchmark is a cross-validation estimate, not a frozen evaluation on the official test split;
- the project does not claim state-of-the-art or production readiness.

See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) for the complete interpretation.

---

## References

1. **PITSILIS, G. K.** *Improved two-stage hate speech classification for twitter based on Deep Neural Networks*. arXiv:2206.04162, 2022.
2. **ZHOU, C. et al.** *A C-LSTM Neural Network for Text Classification*. COLING 2016.
3. **SCHUSTER, M.; PALIWAL, K. K.** *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing, 1997.
4. **JIGSAW / GOOGLE.** *Civil Comments / Jigsaw Unintended Bias in Toxicity Classification*.

---

## License

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE).
