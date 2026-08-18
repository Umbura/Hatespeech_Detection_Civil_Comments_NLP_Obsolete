# Known Issues

This document records known limitations and unresolved evaluation concerns in the historical experiment and repaired experimental paths.

## Confirmed issue

### Model overfitting

Overfitting was confirmed in the historical notebook experiment. The preserved outputs and metrics must therefore be treated as a historical baseline rather than as evidence of final or production-ready performance.

The hierarchical target reformulation does not attempt to repair overfitting. Training/validation behavior must be re-evaluated separately after the target and evaluation pipeline changes are merged.

## Historical evaluation issues

### Cross-validation preprocessing and balancing order

The historical notebook balances the dataset and fits the tokenizer before `StratifiedKFold` creates train/validation folds. That ordering can contaminate validation because resampled examples and learned preprocessing are created before fold isolation.

A leakage-safe replacement path was introduced in PR #4. Regression tests cover train/validation isolation and fold-local learned preprocessing. The historical notebook remains unchanged and still contains the old evaluation procedure.

`scripts/run_leakage_safe_cv.py` remains as an intermediate repaired multiclass baseline. It is not the canonical target formulation after Issue #5.

### Historical target-label definition

The historical target rule scans toxicity score columns in a fixed order and selects the first label above the configured threshold. This makes class assignment order-dependent and discards overlapping toxicity attributes. `severe_toxicity` is configured historically but is absent from the saved final balanced labels.

Issue #5 replaces this rule in the new experimental path with the hierarchical strategy documented in `TARGET_STRATEGY.md`:

- Stage 1: fractional `toxicity` plus auxiliary `severe_toxicity`;
- Stage 2: fractional multilabel outputs for `obscene`, `threat`, `insult`, `identity_attack`, and `sexual_explicit`;
- no synthetic `non_toxic` output class;
- no Stage 2 oversampling in the initial implementation.

The historical notebook remains preserved and is not rewritten.

## Current unresolved evaluation concerns

### Toxicity gate coverage

The initial hierarchical routing definition uses `toxicity >= 0.5` to define the Stage 2 training/oracle subset. This has precedent, but the repository has not yet measured how many fine-grained positives occur with `toxicity < 0.5` in the full Civil Comments training data.

`scripts/analyze_gate_coverage.py` provides the required measurement. The threshold must not be described as empirically safe until that analysis is executed and recorded.

### Hierarchical propagation error

A two-stage system can lose subtype positives when Stage 1 fails to route a sample. For this reason, Stage 2 oracle metrics are insufficient on their own.

The hierarchical runner separates:

1. Stage 1 toxicity-gate metrics;
2. Stage 2 oracle metrics using ground-truth routing;
3. end-to-end subtype metrics using predicted Stage 1 routing.

No replacement benchmark metrics have been generated yet.

## Unsupported claims intentionally avoided

Until the remaining repair and re-evaluation work is complete, the project should not claim:

- production readiness;
- validated F1 around 0.90 as the final benchmark;
- demonstrated cross-language generalization;
- resolved overfitting;
- an empirically optimal routing threshold;
- validated fairness, robustness, or deployment suitability.

## Current scope

The historical notebook is preserved as an immutable experimental reference. Leakage-safe evaluation and a hierarchical target strategy now exist as separate, testable code paths. Gate analysis, overfitting repair, full retraining, threshold calibration, and publication of replacement metrics remain follow-up work.
