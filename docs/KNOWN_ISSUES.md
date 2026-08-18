# Known Issues

This document records known limitations and unresolved evaluation concerns in the historical experiment preserved in this repository.

## Confirmed issue

### Model overfitting

Overfitting was confirmed in the historical notebook experiment. The preserved model outputs and metrics must therefore be treated as a historical baseline rather than as evidence of final or production-ready performance.

The repair phase must re-evaluate training/validation behavior after the experimental pipeline and target formulation are corrected.

## Evaluation concerns

### Cross-validation preprocessing and balancing order

The historical notebook balances the dataset and fits the tokenizer before `StratifiedKFold` creates the train/validation folds. That ordering can contaminate validation because resampled examples and learned preprocessing are created before fold isolation.

A leakage-safe replacement path is now implemented in:

- `src/hate_speech_detection/cv_pipeline.py`;
- `scripts/run_leakage_safe_cv.py`.

The repaired order is:

1. create stratified folds from the raw labeled dataset;
2. isolate training and validation rows;
3. balance only the training partition;
4. fit a new tokenizer only on that fold's balanced training text;
5. transform the untouched validation text with the training-fitted tokenizer.

Regression tests verify that validation rows cannot enter training resampling, validation text is not exposed during tokenizer fitting, training and validation source IDs remain disjoint, and each sample appears in validation exactly once across the configured stratified folds.

The historical notebook remains unchanged and still contains the old evaluation procedure. The new runner has not yet been used to publish replacement model metrics, so the effect of this correction on model performance remains **unmeasured**.

### Target-label definition

The historical target rule is intentionally preserved in the leakage-safe runner so this change does not mix two scientific variables.

The code constructs a single multiclass target by scanning toxicity score columns and selecting the first label whose score exceeds the configured threshold. The configuration includes `severe_toxicity`, while the saved historical balanced output contains seven final labels without `severe_toxicity`.

The target formulation, precedence rule, threshold behavior, and handling of overlapping toxicity categories still require explicit review before final retraining. This repository does not yet claim that multiclass or multilabel formulation is the correct final design.

## Unsupported claims intentionally avoided

Until the remaining repair and re-evaluation work is complete, the project should not claim:

- production readiness;
- validated F1 around 0.90 as the final benchmark;
- demonstrated cross-language generalization;
- resolved overfitting;
- validated fairness, robustness, or deployment suitability.

## Current scope

The historical notebook is preserved as an immutable experimental reference. Leakage-safe fold preparation is implemented separately and covered by regression tests. Target reformulation, overfitting repair, full retraining, and publication of replacement metrics remain separate follow-up work.
