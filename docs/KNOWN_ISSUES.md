# Known Issues and Final Limitations

This document records the limitations that remain after the project was finalized for academic delivery.

## Resolved methodological issues

### Cross-validation leakage

The historical notebook balanced data and fit learned preprocessing before fold isolation. That procedure can contaminate validation.

The repaired pipeline now performs fold-local preprocessing and keeps the outer fold evaluation-only.

### Order-dependent historical targets

The historical notebook collapsed overlapping Civil Comments labels into one order-dependent class.

The repaired hierarchical strategy preserves overlapping Stage 2 labels and separates routing from subtype classification.

### Fixed-threshold mismatch

The original repaired baseline used fixed prediction thresholds (`0.4` for Stage 1 routing and `0.5` for all Stage 2 labels).

Nested threshold selection, using only inner validation, improved:

```text
Stage 1 F1                     0.5567 -> 0.6319
Stage 2 oracle Macro F1        0.4766 -> 0.5959
End-to-end Macro F1            0.3496 -> 0.4412
```

This issue is therefore considered **partially resolved**: threshold mismatch was confirmed as an important source of error, but the hierarchical system still has a remaining propagation gap.

## Final known limitations

### 1. Hierarchical propagation error

The final primary benchmark is:

```text
Stage 2 oracle Macro F1   0.5959
End-to-end Macro F1       0.4412
```

The difference is `0.1547` Macro F1.

This gap exists because Stage 2 cannot recover subtype predictions for comments that Stage 1 fails to route.

The end-to-end score, not the oracle score, is the correct system-level metric.

### 2. Ground-truth gate limitation

The final hierarchy defines Stage 2 membership as:

```text
toxicity >= 0.4
```

On all `1,804,874` training examples:

- 201,476 rows are routed;
- 126,250 rows contain at least one positive Stage 2 label;
- 533 any-positive rows fall below the gate;
- overall any-positive coverage is 99.578%.

`sexual_explicit` is the most affected subtype, with 4.823% of its positive examples falling below the ground-truth gate.

### 3. Early overfitting pressure

The historical experiment showed clear overfitting, and the repaired hierarchy still tends to select early best epochs.

Inner-validation EarlyStopping reduces the effect, but the architecture is not claimed to have solved overfitting.

### 4. Two outer folds

The final experiment uses two outer folds because full Civil Comments training is computationally expensive.

This is an explicit research-design limitation. The project does not claim that two-fold validation provides the same uncertainty estimate as a larger repeated cross-validation design.

### 5. No frozen official-test benchmark

The final repository benchmark is a cross-validation estimate over the Civil Comments training split.

The project did not freeze a development configuration and then perform one untouched final evaluation on the official test split.

Therefore the repository should describe `0.4412` as its **final research cross-validation benchmark**, not as an official test-set score.

### 6. Fairness and subgroup robustness not validated

Civil Comments contains identity-related fields and is commonly used to study unintended bias.

This project does not provide a completed subgroup fairness analysis, BPSN/BNSP evaluation, adversarial robustness study, or cross-domain generalization benchmark.

No fairness or robustness claim should be made from the current evidence.

### 7. Not state of the art or production-ready

The system is an academic experiment.

The repository does not validate:

- production latency or throughput;
- model monitoring;
- drift handling;
- security hardening;
- serving reliability;
- calibrated real-world moderation policies;
- state-of-the-art performance against modern transformer systems under an identical protocol.

## Explicit routing-head follow-up

`run_route_head_cv.py` implements an exploratory binary Stage 1 routing-head experiment.

A 50k-row, one-epoch smoke run completed successfully, proving the path executes, but sampled smoke metrics are not sufficient to replace the final benchmark.

The project is finalized without requiring a full-data route-head run. Future work may revisit it if a new research cycle is opened.

## Historical notebook status

`notebooks/Hatespeech_Detection_LSTM_CNN.ipynb` is preserved unchanged.

Its saved `~0.90` metrics must not be used as the current project benchmark.

## Final reporting rules

For academic presentation:

- report **0.4412** as the primary end-to-end Macro F1;
- report **0.4427** as replication evidence;
- report `0.5959` only as Stage 2 oracle performance;
- state that the benchmark is cross-validation based;
- state that fairness, robustness, official-test performance, SOTA status, and production readiness are not established.
