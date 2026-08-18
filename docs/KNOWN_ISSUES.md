# Known Issues

This document records known limitations and unresolved evaluation concerns in the historical experiment and repaired experimental paths.

## Confirmed issue

### Model overfitting

Overfitting was confirmed in the historical notebook experiment. The preserved outputs and metrics must therefore be treated as a historical baseline rather than as evidence of final or production-ready performance.

The repaired hierarchical experiments also select very early best inner-validation epochs. EarlyStopping limits the damage, but the architecture still begins to overfit quickly and this remains unresolved.

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

This definition determines Stage 2 membership and oracle evaluation. It does not imply that a trained Stage 1 score should also use `0.4` as its prediction threshold.

### Hierarchical propagation error

The first repaired fixed-threshold baseline quantified a substantial propagation gap:

```text
Stage 1 recall                  0.4614
Stage 2 oracle Macro F1         0.4674
End-to-end Macro F1             0.3484
```

The nested threshold diagnostic then reduced part of this error without changing the architecture or loss:

```text
Stage 1 tuned recall            0.5653
Stage 1 tuned F1                0.6319
Stage 2 oracle tuned Macro F1   0.5959
End-to-end tuned Macro F1       0.4412
```

Stage 1 PR-AUC/average precision was `0.7095` and ROC-AUC was `0.9195`. This supports the conclusion that fixed-threshold mismatch explained a meaningful part of the original routing problem.

However, the remaining oracle-to-end-to-end gap is still `0.1547` Macro F1 (`0.5959 - 0.4412`). A two-stage system permanently loses subtype predictions when Stage 1 does not route a sample, so propagation error remains an active limitation.

The runner therefore keeps three distinct evaluation views:

1. Stage 1 routing metrics;
2. Stage 2 oracle metrics using ground-truth routing;
3. end-to-end subtype metrics using predicted Stage 1 routing.

Stage 2 oracle metrics must not be presented as end-to-end system performance.

### Prediction-threshold mismatch — partially resolved

The fixed baseline used `0.4` for predicted Stage 1 routing and `0.5` for all Stage 2 labels. PR #10 tested nested threshold selection using only the common inner validation partition and freezing thresholds before outer evaluation.

The full-data two-fold result improved:

- Stage 1 F1: `0.5567 -> 0.6319` within the PR #10 same-fit comparison;
- Stage 2 oracle Macro F1: `0.4766 -> 0.5959`;
- end-to-end Macro F1: `0.3496 -> 0.4412`.

Selected thresholds were also reasonably consistent across the two full-data folds: Stage 1 routing selected `0.27` and `0.30`, while Stage 2 label thresholds stayed between `0.33` and `0.46`.

Threshold mismatch is therefore no longer an untested hypothesis; it is a confirmed contributor to the earlier error. It is not, by itself, a complete solution because the remaining propagation gap is still material.

### Stage 1 training-objective alignment

Stage 1 is currently trained only on fractional `toxicity` and `severe_toxicity` scores, while its operational role is a binary routing decision defined by `toxicity >= 0.4`.

The next controlled experiment will add an explicit binary routing output while preserving both existing soft outputs. The experiment should change no other major training factor. Its purpose is to test whether direct supervision of the operational decision improves end-to-end Macro F1 beyond the nested-threshold baseline of `0.4412`.

Weighted BCE, focal/asymmetric loss, oversampling, architecture replacement, and transformer baselines remain deferred until this narrower hypothesis is tested.

### Runtime reproducibility across local and Colab environments

The repaired local environment is pinned in `requirements.txt`, while the completed Colab runs used a Tesla T4 runtime whose NumPy/pandas/scikit-learn versions differed from the local pins. The exact Colab versions are recorded in `REPRODUCIBILITY.md`.

Any final reported metric must record the exact execution environment that generated it. A final result should not silently combine metrics from different unrecorded runtimes.

## Unsupported claims intentionally avoided

Until the remaining re-evaluation work is complete, the project should not claim:

- production readiness;
- validated F1 around 0.90 as the final benchmark;
- demonstrated cross-language generalization;
- resolved overfitting;
- that nested threshold selection eliminates hierarchical propagation error;
- that the next routing-head experiment is superior before outer validation is executed;
- validated fairness, robustness, or deployment suitability;
- a final official-test result before the development configuration is frozen.

## Current scope

The historical notebook is preserved as an immutable experimental reference. Leakage-safe evaluation, the hierarchical target strategy, full-train gate analysis, a repaired fixed-threshold baseline, and a completed nested threshold diagnostic now exist as separate, reviewable evidence.

The immediate next experiment is the explicit binary Stage 1 routing head. More invasive imbalance or architecture experiments should be added only if that evidence shows they are needed.