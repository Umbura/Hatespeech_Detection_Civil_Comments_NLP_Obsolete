# Known Issues

This document records known limitations and unresolved evaluation concerns in the historical experiment preserved in this repository. These items are documented for traceability and are not presented as resolved defects.

## Confirmed issue

### Model overfitting

Overfitting was confirmed in the current notebook experiment. The preserved model outputs and metrics must therefore be treated as a historical baseline rather than as evidence of final or production-ready performance.

The repair phase must re-evaluate training/validation behavior after the experimental pipeline is corrected.

## Evaluation concerns requiring correction or explicit review

### Cross-validation preprocessing and balancing order

The historical notebook balances the dataset and fits the tokenizer before `StratifiedKFold` creates the train/validation folds.

This ordering creates a risk of train/validation contamination and prevents the current cross-validation outputs from being treated as a clean benchmark. Oversampling performed before fold creation is particularly important to correct because duplicated samples may be distributed across training and validation subsets.

The magnitude of the effect on the saved metrics has not yet been measured. A repaired evaluation must perform fold-specific preprocessing and keep validation data untouched by training-only resampling.

### Target-label definition

The notebook constructs a single multiclass target by scanning toxicity score columns and selecting the first label whose score exceeds the configured threshold.

The historical configuration includes `severe_toxicity`, while the saved balanced output contains seven final labels without `severe_toxicity`. The target formulation, precedence rule, threshold behavior, and intended handling of overlapping toxicity categories require explicit review before retraining.

This repository does not yet claim that multiclass or multilabel formulation is the correct final design.

## Unsupported claims intentionally avoided

Until the repair and re-evaluation work is complete, the project should not claim:

- production readiness;
- validated F1 around 0.90 as the final benchmark;
- demonstrated cross-language generalization;
- resolved overfitting;
- validated fairness, robustness, or deployment suitability.

## Current scope

The historical notebook remains preserved so future experimental changes can be compared against a traceable baseline. Model repair, target reformulation, leakage prevention, retraining, and final evaluation are separate follow-up work.
