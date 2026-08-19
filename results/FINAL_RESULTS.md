# Final Research Results

## Official benchmark

The repository's final research benchmark is the **nested-threshold hierarchical CNN + Bi-LSTM** experiment on the Civil Comments training split.

The official value retained for reporting is:

```text
End-to-end Macro F1 = 0.4412
```

Two later full-data reproductions support the same performance region:

```text
Previous replication:      0.4427
Publication reproduction:  0.4423
```

The primary benchmark remains `0.4412` to avoid retroactively selecting the largest run.

## Primary full-data run

### Stage 1

| Metric | Fixed | Nested |
|---|---:|---:|
| Accuracy | 0.9257 | 0.9265 |
| Precision | 0.8340 | 0.7163 |
| Recall | 0.4178 | 0.5653 |
| F1 | 0.5567 | 0.6319 |
| Routing rate | 0.0559 | 0.0881 |
| PR-AUC / AP | 0.7095 | 0.7095 |
| ROC-AUC | 0.9195 | 0.9195 |

### Stage 2 oracle

| Metric | Fixed | Nested |
|---|---:|---:|
| Macro F1 | 0.4766 | 0.5959 |

Nested per-label F1:

| Label | F1 |
|---|---:|
| obscene | 0.6177 |
| threat | 0.5112 |
| insult | 0.7639 |
| identity_attack | 0.5595 |
| sexual_explicit | 0.5272 |

### End-to-end

| Metric | Fixed | Nested |
|---|---:|---:|
| Macro F1 | 0.3496 | **0.4412** |

Nested per-label F1:

| Label | F1 |
|---|---:|
| obscene | 0.5115 |
| threat | 0.2951 |
| insult | 0.6343 |
| identity_attack | 0.3761 |
| sexual_explicit | 0.3892 |

The end-to-end gain is `+0.0916` absolute, approximately `+26.2%` relative.

## Previous replication

A second full run produced:

| Metric | Result |
|---|---:|
| Stage 1 F1 | 0.6351 |
| Stage 1 recall | 0.5780 |
| Stage 1 PR-AUC / AP | 0.7086 |
| Stage 1 ROC-AUC | 0.9195 |
| Stage 2 oracle Macro F1 | 0.5967 |
| End-to-end Macro F1 | **0.4427** |

## Publication reproduction

The final notebook published for the scientific-initiation delivery was executed on **2026-08-19** from commit `cfcebb2` on CPU because TensorFlow reported CUDA unavailable.

### Stage 1

| Metric | Fixed | Nested |
|---|---:|---:|
| Accuracy | 0.9253 | 0.9254 |
| Precision | 0.8389 | 0.6962 |
| Recall | 0.4092 | 0.5879 |
| F1 | 0.5501 | **0.6375** |
| Routing rate | 0.0544 | 0.0943 |
| PR-AUC / AP | 0.7090 | 0.7090 |
| ROC-AUC | 0.9194 | 0.9194 |

### Stage 2 oracle

| Metric | Fixed | Nested |
|---|---:|---:|
| Macro F1 | 0.4791 | **0.5977** |

### End-to-end

| Metric | Fixed | Nested |
|---|---:|---:|
| Macro F1 | 0.3474 | **0.4423** |

Nested per-label F1:

| Label | F1 |
|---|---:|
| obscene | 0.5112 |
| threat | 0.3002 |
| insult | 0.6291 |
| identity_attack | 0.3809 |
| sexual_explicit | 0.3900 |

This run improved end-to-end Macro F1 by `+0.0949` absolute (`+27.3%` relative) compared with its fixed-threshold evaluation.

The three full runs (`0.4412`, `0.4427`, `0.4423`) support a stable performance region around `0.44`.

## Scientific interpretation

The main finding is not a state-of-the-art claim. The evidence shows that:

1. the CNN + Bi-LSTM encoder retains useful discriminative information;
2. fixed probability thresholds were materially mismatched to the imbalanced hierarchical task;
3. nested threshold selection improved routing recall and rare-label F1 without changing the core architecture;
4. hierarchical propagation error remains measurable because Stage 2 cannot recover labels from samples that Stage 1 fails to route.

## Reporting rule

For academic reporting:

- use **0.4412** as the primary end-to-end Macro F1;
- report **0.4427** and **0.4423** as independent reproduction evidence;
- distinguish Stage 2 oracle from system end-to-end performance;
- do not report the historical `~0.90` notebook result as the current benchmark;
- do not claim production readiness or state of the art.
