# Known Issues

This document records known limitations and unresolved evaluation concerns in the historical experiment and repaired experimental paths.

## Confirmed issue

### Model overfitting

Overfitting was confirmed in the historical notebook experiment. The preserved outputs and metrics must therefore be treated as a historical baseline rather than as evidence of final or production-ready performance.

The repaired two-fold hierarchical baseline also showed very early best inner-validation epochs: Stage 1 selected epochs 2 and 1 across the two folds, while Stage 2 selected epochs 2 and 3. EarlyStopping limits the damage, but the architecture still begins to overfit quickly and this remains unresolved.

## Historical evaluation issues

### Cross-validation preprocessing and balancing order

The historical notebook balances the dataset and fits the tokenizer before `StratifiedKFold` creates train/validation folds. That ordering can contaminate validation because resampled examples and learned preprocessing are created before fold isolation.

A leakage-safe replacement path was introduced in PR #4. Regression tests cover train/validation isolation and fold-local learned preprocessing. The historical notebook remains unchanged and still contains the old evaluation procedure.

`scripts/run_leakage_safe_cv.py` remains as an intermediate repaired multiclass baseline. It is not the canonical target formulation after Issue #5.

### Historical target-label definition

The historical target rule scans toxicity score columns in a fixed order and selects the first label above the configured threshold. This makes class assignment order-dependent and discards overlapping toxicity attributes. `severe_toxicity` is configured historically but is absent from the saved final balanced labels.

Issue #5 replaced this rule in the new experimental path with the hierarchical strategy documented in `TARGET_STRATEGY.md`:

- Stage 1: fractional `toxicity` plus auxiliary `severe_toxicity`;
- Stage 2: fractional multilabel outputs for `obscene`, `threat`, `insult`, `identity_attack`, and `sexual_explicit`;
- no synthetic `non_toxic` output class;
- no Stage 2 oversampling in the initial implementation.

The historical notebook remains preserved and is not rewritten.

## Current unresolved evaluation concerns

### Ground-truth toxicity-gate limitation

Full-train gate coverage was measured on all `1,804,874` Civil Comments training samples. The repaired hierarchy uses `toxicity >= 0.4` as its **ground-truth routing definition**.

At this threshold:

- 201,476 samples (11.16% of train) are routed to Stage 2;
- 533 of 126,250 samples with at least one positive Stage 2 label fall below the gate;
- any-positive coverage is 99.578%;
- `sexual_explicit` remains the most affected subtype, with 226 of 4,686 positives (4.823%) excluded by the ground-truth gate.

This definition determines Stage 2 membership and oracle evaluation. It does not prove that a trained Stage 1 model should use `0.4` as its prediction threshold.

### Hierarchical propagation error

The repaired two-fold fixed-threshold baseline quantified a substantial propagation gap:

```text
Stage 1 recall                  0.4614
Stage 2 oracle Macro F1         0.4674
End-to-end Macro F1             0.3484
```

The Stage 1 accuracy was 0.9266, but the much lower recall shows that accuracy is dominated by the non-routed majority and is not sufficient evidence of a useful routing gate.

A two-stage system can lose subtype positives permanently when Stage 1 fails to route a sample. For this reason, Stage 2 oracle metrics are insufficient on their own.

The current runner therefore separates:

1. Stage 1 toxicity-gate metrics;
2. Stage 2 oracle metrics using ground-truth routing;
3. end-to-end subtype metrics using predicted Stage 1 routing.

### Prediction-threshold mismatch

The fixed baseline used `0.4` for predicted Stage 1 routing and `0.5` for all Stage 2 labels. Those values are convenient references but are not guaranteed to be optimal operating points for model outputs.

The current experimental path now selects prediction thresholds only inside the common inner validation partition:

- Stage 2 thresholds are selected per label by F1;
- the Stage 1 routing threshold is selected by end-to-end Macro F1;
- fixed-threshold metrics are retained from the same trained models for direct comparison;
- the outer validation fold remains untouched until final evaluation.

This is a diagnostic before introducing more expensive training changes. If PR-AUC/average precision is strong while tuned F1 improves materially, threshold mismatch explains part of the error. If discrimination remains weak, loss weighting, regularization, or architecture changes become better-supported follow-up experiments.

### Runtime reproducibility across local and Colab environments

The repaired local environment is pinned in `requirements.txt`, while the completed fixed-threshold baseline was executed in a Colab Tesla T4 runtime whose NumPy/pandas/scikit-learn versions differed from the local pins. The exact Colab versions are recorded in `REPRODUCIBILITY.md`.

Any final reported metric must record the exact execution environment that generated it. A final result should not silently combine metrics from different unrecorded runtimes.

## Unsupported claims intentionally avoided

Until the remaining re-evaluation work is complete, the project should not claim:

- production readiness;
- validated F1 around 0.90 as the final benchmark;
- demonstrated cross-language generalization;
- resolved overfitting;
- an empirically optimal predicted routing threshold before outer validation confirms it;
- validated fairness, robustness, or deployment suitability;
- a final official-test result before the development configuration is frozen.

## Current scope

The historical notebook is preserved as an immutable experimental reference. Leakage-safe evaluation, the hierarchical target strategy, full-train gate analysis, a repaired two-fold fixed-threshold baseline, and nested threshold selection now exist as separate, reviewable steps.

The immediate next step is to run the nested threshold diagnostic and compare fixed versus tuned outer-fold metrics from the same model fits. More invasive imbalance or architecture experiments should be added only if that evidence shows they are needed.
