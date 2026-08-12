# Reproducibility Baseline

## Current status

This repository preserves a historical CNN + Bi-LSTM experiment whose exact runtime package versions were not recorded. The existing `requirements.txt` therefore describes the runtime dependency set but must not be interpreted as an exact environment lock.

The saved notebook remains the source of truth for the historical experiment. Its scientific limitations and evaluation concerns are documented separately in `KNOWN_ISSUES.md` and `EXPERIMENT_HISTORY.md`.

## Repository-level validation

The repository now provides deterministic structural checks that do not require training the model or downloading the Civil Comments dataset.

Run them from the repository root:

```bash
python -m compileall -q src
python -m unittest discover -s tests -p "test_*.py" -v
```

These checks verify repository layout, source syntax, notebook JSON/nbformat integrity, notebook placement, and documentation references.

GitHub Actions runs the same validation on pull requests and pushes to `main`. The CI job uses Python 3.12 for repository validation only; this is not a claim that Python 3.12 reproduces the original training environment.

## What this baseline does not validate

The current CI does not:

- install the full ML runtime dependency set;
- download or validate the Civil Comments dataset;
- execute notebook training cells;
- retrain the model;
- reproduce historical metrics;
- validate model quality, leakage, overfitting, label semantics, or fairness.

Those checks would be misleading before the experimental pipeline is repaired.

## Requirements for the repaired experiment

When the repaired pipeline is ready for evaluation, reproducibility evidence should include at least:

1. a supported Python version;
2. exact dependency versions or a generated lock/environment file;
3. deterministic seeds where supported;
4. the dataset source and revision/configuration used;
5. preprocessing, balancing, split, and label-generation configuration;
6. hardware/runtime notes when they materially affect execution;
7. commands required to train and evaluate from a clean environment;
8. generated evaluation artifacts tied to the corresponding code revision.

No historical metric should be promoted to a validated benchmark until that environment and evaluation procedure have been executed successfully.
