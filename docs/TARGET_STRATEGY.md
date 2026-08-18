# Hierarchical Target Strategy

## Status

This document defines the repaired target formulation introduced for Issue #5. It replaces the historical order-dependent multiclass rule in the new experimental path, but it does not replace the historical notebook or its saved outputs.

No replacement benchmark metrics are claimed until the hierarchical runner is executed and evaluated.

## Why the historical rule is being replaced

The historical notebook scans Civil Comments toxicity attributes in a fixed order and returns the first score above a threshold. That makes the final class depend on column order and discards valid overlaps between toxicity attributes.

The repaired strategy preserves the original fractional annotation scores and separates broad toxicity detection from fine-grained subtype prediction.

## Stage 1 — toxicity routing

Stage 1 trains on the full Civil Comments dataset with two sigmoid outputs:

- `toxicity` — primary routing signal;
- `severe_toxicity` — auxiliary output.

Both outputs use the original fractional scores as soft training targets. The routing threshold is applied only when a binary routing decision is required.

Initial routing threshold:

```text
toxicity >= 0.5
```

The threshold is configurable and is not assumed to be optimal.

## Stage 2 — fine-grained multilabel classification

Stage 2 trains only on rows selected by the ground-truth Stage 1 routing definition for the current experiment. It predicts five independent sigmoid outputs:

- `obscene`;
- `threat`;
- `insult`;
- `identity_attack`;
- `sexual_explicit`.

The five original fractional scores are preserved as soft training targets. A sample may therefore contribute to multiple labels simultaneously.

`non_toxic` is not modeled as a synthetic output class.

## Balancing policy

The initial hierarchical implementation does not oversample Stage 2 labels.

Multilabel oversampling can distort joint label distributions because one duplicated example may carry multiple labels. Class weighting, focal loss, threshold calibration, or other imbalance strategies may be evaluated later as controlled experiments, but they are not part of Issue #5.

## Cross-validation

The outer folds are constructed in two coordinated parts:

1. samples routed to Stage 2 are split with iterative multilabel stratification using binarized subtype scores only for fold construction;
2. non-routed samples are split independently with shuffled K-fold;
3. corresponding partitions are recombined into full train/validation folds.

The fractional targets remain unchanged for training.

This allows Stage 1 to train and validate on the full dataset while Stage 2 uses the toxic portion of the same outer fold without introducing resampling duplicates.

## Gate-coverage prerequisite

Before treating `toxicity >= 0.5` as an empirically safe gate, run:

```bash
python scripts/analyze_gate_coverage.py
```

The report counts, per Stage 2 label and overall, how many subtype-positive examples (`score >= 0.5`) have `toxicity < 0.5` and would therefore be excluded by the ground-truth gate.

The repository must report the measured result rather than assume the missed-positive rate is negligible.

## Required evaluation views

The hierarchical runner reports three distinct evaluation levels:

### Stage 1

Measures the toxicity routing decision on the full validation population.

### Stage 2 oracle

Measures fine-grained subtype predictions only for validation rows selected by the ground-truth toxicity gate. This isolates Stage 2 quality from Stage 1 routing errors.

### End-to-end

Uses the predicted Stage 1 gate to decide which validation samples receive Stage 2 predictions. This captures propagation errors from the first stage and is the relevant full-system view.

Stage 2 oracle metrics must not be presented as end-to-end system performance.

## Deferred work

Issue #5 does not attempt to solve:

- confirmed model overfitting;
- per-label threshold calibration;
- focal loss or weighted BCE experiments;
- architecture search;
- fairness or bias evaluation;
- production readiness.
