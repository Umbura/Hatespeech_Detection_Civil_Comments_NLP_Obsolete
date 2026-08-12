# Known Issues

This document records known limitations of the historical experiment preserved in this repository. It is not a list of resolved defects.

## Confirmed

### Model overfitting

Overfitting was confirmed in the current notebook experiment. The existing model and saved metrics must therefore be treated as a historical baseline rather than as a final or production-ready result.

## Evaluation concerns to address

### Cross-validation preprocessing

The current notebook performs dataset balancing and fits the tokenizer before `StratifiedKFold` creates the train/validation folds. This evaluation pipeline must be reviewed before the historical metrics are treated as a reliable benchmark.

### Target-label definition

The current notebook defines several toxicity columns and then assigns a single label using the first score above a threshold. The target formulation and the handling of `severe_toxicity` need explicit review before retraining.

## Current scope

The repository-reorganization work intentionally preserves the historical notebook without modifying its training logic, architecture, dataset processing, or reported outputs. Model repair and re-evaluation will be handled separately.
