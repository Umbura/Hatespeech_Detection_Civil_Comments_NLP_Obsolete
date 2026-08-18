<div align="center">

# Hybrid Hate Speech Detection

### Historical CNN + Bi-LSTM Research Baseline on Civil Comments

[![Read in Portuguese](https://img.shields.io/badge/Read%20in-Portuguese-2ea44f?style=for-the-badge&logo=google-translate&logoColor=white)](README_PT.md)

<p>
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TensorFlow-2.21-orange" alt="TensorFlow 2.21">
  <img src="https://img.shields.io/badge/Status-Historical_Baseline-yellow" alt="Status: Historical Baseline">
  <a href="https://colab.research.google.com/github/Umbura/Hatespeech_Detection_Civil_Comments_NLP_Obsolete/blob/main/notebooks/Hatespeech_Detection_LSTM_CNN.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

<img src="assets/resultado_neos_v4.png" alt="Historical Civil Comments experiment results" width="100%">

*Historical output retained for traceability. It is not the validated benchmark for a repaired model.*

</div>

---

## Overview

This repository preserves a research experiment for multiclass toxicity / hate-speech-related text classification using the **Civil Comments** dataset and a parallel **CNN + Bi-LSTM** neural architecture.

The experiment evolved through multiple approaches before reaching the parallel hybrid architecture currently stored in the historical notebook. Embeddings are trained from scratch rather than initialized with pre-trained vectors such as GloVe or Word2Vec.

The repository is now in a controlled repair and re-evaluation cycle. The historical notebook is intentionally preserved so the repaired experimental path can be compared against it without rewriting past evidence.

---

## Current Project Status

The saved experiment is **not considered a final or production-ready model**.

| Item | Current status |
| :--- | :--- |
| Architecture | Parallel CNN + Bi-LSTM historical baseline |
| Dataset | Civil Comments |
| Saved macro F1 | ~0.90 (historical run) |
| Overfitting | **Confirmed** |
| Evaluation pipeline | Leakage-safe and hierarchical runners implemented; replacement benchmark pending |
| Target-label strategy | Hierarchical two-stage strategy defined; initial routing gate selected at `0.4` |
| Model repair | In progress |
| Final validated benchmark | **Not available yet** |

The saved notebook reports an overall macro F1 around **0.90**, but this value must be interpreted only as a **historical experimental result**. Confirmed overfitting and unresolved model-quality questions mean it should not be used as evidence of final model quality.

See [Known Issues](docs/KNOWN_ISSUES.md), [Experiment History](docs/EXPERIMENT_HISTORY.md), [Target Strategy](docs/TARGET_STRATEGY.md), and [Reproducibility](docs/REPRODUCIBILITY.md) for the current technical interpretation.

---

## Project Evolution

### 1. Initial Phase — One-vs-Rest Prototype

- **Approach:** Two-stage One-vs-Rest classification inspired by Pitsilis et al. (2022).
- **Architecture:** Multiple independent Bi-LSTM classifiers.
- **Historical result:** approximately 71% accuracy.
- **Observed limitation:** model-management complexity and weak probability integration.

### 2. Intermediate Phase — Sequential Hybrid

- **Approach:** CNN feature extraction followed by recurrent processing (CNN → RNN).
- **Historical result:** approximately 86% accuracy.
- **Observed limitation:** the sequential design appeared to discard contextual information needed by the recurrent stage.

### 3. Historical Baseline — Parallel Hybrid

- **Approach:** the same embedding input is processed by independent contextual and local-pattern branches and fused before classification.
- **Architecture:** Bi-LSTM branch + multi-kernel CNN branch.
- **Saved experiment:** macro F1 around 0.90.
- **Current interpretation:** useful as an experimental baseline, but not yet a validated benchmark because the run has confirmed overfitting and the repaired pipeline has not yet produced replacement metrics.

<div align="center">
  <img src="assets/resultado_neos_v3.png" alt="Historical architecture evolution result" width="80%">
</div>

---

## Architecture

The historical model processes the tokenized input through two branches in parallel:

1. **Embedding** — trainable embedding initialized from scratch.
2. **Contextual branch (Bi-LSTM)** — models sequential context and token order.
3. **Pattern branch (Multi-Kernel CNN)** — Conv1D kernels of different sizes capture local patterns.
4. **Fusion and classification** — branch outputs are concatenated, followed by dense layers and a softmax classifier.

A GRU variant was also explored historically, but the preserved experiment uses Bi-LSTM.

---

## Data and Historical Training Procedure

- **Dataset:** [Google / Civil Comments](https://huggingface.co/datasets/google/civil_comments)
- **Historical balancing:** undersampling of larger classes and oversampling of smaller classes.
- **Historical validation:** 5-fold stratified cross-validation.
- **Training control:** Early Stopping.
- **Embedding:** learned from scratch.

These describe what the preserved notebook does; they do **not** imply that the historical protocol is the final methodology. The repaired path now isolates fold preprocessing and uses the hierarchical target strategy documented in `docs/TARGET_STRATEGY.md`.

---

## Historical Results

The preserved notebook contains the previously executed outputs and reports approximately:

| Metric | Historical saved result |
| :--- | :--- |
| Accuracy | ~0.90 |
| Macro F1 | ~0.90 |
| Macro Recall | ~0.90 |

These metrics are retained for traceability. They will be replaced as the project benchmark only after the repaired experimental pipeline is retrained and independently re-evaluated.

---

## Known Limitations

The current project still has the following documented limitations:

- confirmed model overfitting;
- the hierarchical runner has not yet produced replacement benchmark metrics;
- Stage 1 routing errors may propagate into Stage 2 and must be measured end-to-end;
- the selected `0.4` routing gate still misses a small fraction of subtype-positive rows, especially `sexual_explicit`;
- per-label threshold calibration, fairness, robustness, and production suitability remain unvalidated;
- historical metrics are not considered the final benchmark.

Details: [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) and [docs/TARGET_STRATEGY.md](docs/TARGET_STRATEGY.md).

---

## Repository Structure

```text
.
├── .python-version
├── AGENTS.md
├── README.md
├── README_PT.md
├── requirements.txt
├── notebooks/
│   └── Hatespeech_Detection_LSTM_CNN.ipynb
├── scripts/
│   ├── analyze_gate_coverage.py
│   ├── run_hierarchical_cv.py
│   └── run_leakage_safe_cv.py
├── src/
│   └── hate_speech_detection/
├── tests/
├── docs/
│   ├── KNOWN_ISSUES.md
│   ├── EXPERIMENT_HISTORY.md
│   ├── REPRODUCIBILITY.md
│   └── TARGET_STRATEGY.md
└── assets/
```

The notebook remains the historical experiment. The `src/`, `scripts/`, and `tests/` areas contain the repaired implementation and validation path.

---

## Repaired Runtime Setup

The repaired runtime is pinned to **Python 3.12** and the exact dependency versions in `requirements.txt`.

Using `uv`:

```bash
git clone https://github.com/Umbura/Hatespeech_Detection_Civil_Comments_NLP_Obsolete.git
cd Hatespeech_Detection_Civil_Comments_NLP_Obsolete
uv venv --python 3.12
uv pip install -r requirements.txt
```

Run repository validation:

```bash
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

Reproduce the measured gate analysis:

```bash
python scripts/analyze_gate_coverage.py
```

Start the hierarchical experiment only when a full training run is intended:

```bash
python scripts/run_hierarchical_cv.py
```

The historical notebook predates the pinned repaired runtime. Its exact original package versions were not recorded, so the current pins should not be described as the original historical environment.

---

## Historical Notebook

Open `notebooks/Hatespeech_Detection_LSTM_CNN.ipynb` in a Jupyter-compatible environment or use the Colab button at the top of this README.

> The notebook is preserved as historical evidence and contains known methodological limitations. It is intentionally not modified to represent the repaired pipeline.

---

## References

1. **PITSILIS, G. K.** *Improved two-stage hate speech classification for twitter based on Deep Neural Networks*. arXiv:2206.04162, 2022.
2. **ZHOU, C. et al.** *A C-LSTM Neural Network for Text Classification*. COLING 2016.
3. **SCHUSTER, M.; PALIWAL, K. K.** *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing, 1997.
4. **JIGSAW / GOOGLE.** *Jigsaw Unintended Bias in Toxicity Classification*. Kaggle, 2019.

---

## License

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
