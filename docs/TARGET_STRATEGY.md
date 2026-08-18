# Hierarchical Target Strategy

## Status

This document defines the repaired target formulation introduced for Issue #5. It replaces the historical order-dependent multiclass rule in the new experimental path, but it does not replace the historical notebook or its saved outputs.

The repaired strategy preserves the original fractional annotation scores and separates broad toxicity routing from fine-grained subtype prediction.

## Why the historical rule is being replaced

The historical notebook scans Civil Comments toxicity attributes in a fixed order and returns the first score above a threshold. That makes the final class depend on column order and discards valid overlaps between toxicity attributes.

The repaired strategy keeps overlapping targets and separates the ground-truth definition used to construct the hierarchy from prediction thresholds chosen for an imperfect trained model.

## Stage 1 — toxicity routing

Stage 1 trains on the full Civil Comments dataset with two sigmoid outputs:

- `toxicity` — primary routing signal;
- `severe_toxicity` — auxiliary output.

Both outputs use the original fractional scores as soft training targets.

### Ground-truth routing definition

The hierarchy defines a sample as routed when its original Civil Comments toxicity annotation satisfies:

```text
toxicity >= 0.4
```

This value was selected from measured full-train gate coverage. It defines which training rows belong to Stage 2 and which validation rows are used for the Stage 2 oracle view.

It is **not** assumed to be the optimal threshold for the predicted Stage 1 probability. A trained model is imperfect and its score distribution need not be calibrated so that a predicted value of `0.4` has the same operating meaning as an annotation score of `0.4`.

### Predicted routing threshold

The hierarchical runner therefore reports two end-to-end references from the same trained models:

1. a fixed prediction threshold of `0.4`, retained for direct comparison with the original repaired baseline;
2. a nested prediction threshold selected only on the common inner validation partition to maximize end-to-end Macro F1.

The selected prediction threshold is frozen before the outer validation fold is evaluated. The outer fold never participates in threshold selection.

## Stage 2 — fine-grained multilabel classification

Stage 2 trains only on rows selected by the ground-truth Stage 1 routing definition for the current experiment. It predicts five independent sigmoid outputs:

- `obscene`;
- `threat`;
- `insult`;
- `identity_attack`;
- `sexual_explicit`.

The five original fractional scores are preserved as soft training targets. A sample may therefore contribute to multiple labels simultaneously.

`non_toxic` is not modeled as a synthetic output class.

For binary evaluation, the ground-truth fine-grained labels continue to use `score >= 0.5`. Prediction thresholds are treated separately:

1. `0.5` remains the fixed-threshold reference;
2. one threshold per Stage 2 output is selected on routed inner-validation rows by per-label F1;
3. those thresholds are frozen before the outer Stage 2 oracle and end-to-end metrics are computed.

## Balancing policy

The hierarchical implementation does not oversample Stage 2 labels.

Multilabel oversampling can distort joint label distributions because one duplicated example may carry multiple labels. Class weighting, focal/asymmetric losses, or other imbalance strategies remain possible controlled follow-up experiments, but they are not introduced before the threshold diagnostic establishes whether the current model already separates the classes adequately.

## Cross-validation and model selection

The outer folds are constructed in two coordinated parts:

1. samples routed to Stage 2 are split with iterative multilabel stratification using binarized subtype scores only for fold construction;
2. non-routed samples are split independently with shuffled K-fold;
3. corresponding partitions are recombined into full outer train/validation folds.

Inside each outer training fold, one **common inner validation partition** is constructed in the same hierarchical manner:

- routed rows are multilabel-stratified;
- non-routed rows are split separately;
- both are recombined into a full inner train/validation partition;
- Stage 2 uses exactly the routed subset of those same inner partitions.

This common split is important for end-to-end threshold selection: every row used to choose a Stage 1 routing threshold is excluded from both Stage 1 and Stage 2 fitting.

The inner validation partition is used for:

- EarlyStopping for Stage 1 and Stage 2;
- Stage 2 per-label prediction-threshold selection;
- Stage 1 predicted-routing-threshold selection using end-to-end Macro F1.

The outer validation fold is evaluation-only.

The fractional targets remain unchanged for training.

## Gate-coverage evidence

Gate coverage was executed on the full Civil Comments training split (`1,804,874` samples). Fine-grained labels were counted as positive at `score >= 0.5`.

| Ground-truth gate threshold | Routed samples | Share of train | Any Stage 2 positives missed | Any-positive coverage |
| --- | ---: | ---: | ---: | ---: |
| `0.50` | 144,334 | 8.00% | 4,213 / 126,250 (3.337%) | 96.663% |
| `0.40` | 201,476 | 11.16% | 533 / 126,250 (0.422%) | 99.578% |
| `0.35` | 204,460 | 11.33% | 529 / 126,250 (0.419%) | 99.581% |
| `0.30` | 266,089 | 14.74% | 79 / 126,250 (0.063%) | 99.937% |

`0.40` remains the selected **ground-truth routing definition**. Moving from `0.40` to `0.35` routed 2,984 additional samples while recovering only 4 additional Stage 2-positive samples, so the extra routing load did not justify the negligible coverage gain.

`sexual_explicit` remains the most affected subtype at the selected ground-truth gate: 226 of 4,686 positives (4.823%) fall below `toxicity = 0.4`.

The separate predicted-routing threshold is now selected from model outputs inside nested validation rather than inferred from this gate-coverage table.

## Required evaluation views

The hierarchical runner reports three distinct evaluation levels and two prediction-threshold regimes.

### Stage 1

Measures the toxicity routing decision on the full outer-validation population. Precision, recall, F1, routing rate, PR-AUC/average precision, and ROC-AUC are reported so a high accuracy caused by class imbalance cannot hide poor routing recall.

### Stage 2 oracle

Measures fine-grained subtype predictions only for outer-validation rows selected by the ground-truth toxicity gate. This isolates Stage 2 quality from Stage 1 routing errors. Per-label F1 and PR-AUC/average precision are reported.

### End-to-end

Uses the predicted Stage 1 gate to decide which outer-validation samples receive Stage 2 predictions. This captures propagation errors from the first stage and is the relevant full-system view.

For all three views, the fixed thresholds are retained as a reference while nested tuned thresholds quantify whether decision-threshold mismatch explains part of the observed error without retraining a different architecture.

Stage 2 oracle metrics must not be presented as end-to-end system performance.

## Deferred work

Threshold selection is now part of the repaired experimental path, but the following remain separate controlled questions:

- confirmed model overfitting beyond EarlyStopping;
- class-weighted, focal, or asymmetric loss experiments;
- architecture search or model replacement;
- fairness or identity-bias evaluation;
- final untouched official-test evaluation;
- production readiness.
