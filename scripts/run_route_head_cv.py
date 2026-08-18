"""Run the controlled explicit Stage 1 routing-head experiment.

This experiment changes one training decision relative to the nested-threshold
hierarchical baseline: Stage 1 receives a third sigmoid output trained directly
on the binary ground-truth routing decision ``toxicity >= 0.4``.

The existing fractional ``toxicity`` and ``severe_toxicity`` outputs remain.
Stage 2, data handling, nested threshold selection, and outer evaluation remain
unchanged.

Within each fold, the same trained three-output Stage 1 model is evaluated in
two routing modes:

1. routing from the fractional toxicity output;
2. routing from the explicit binary route output.

Both routing thresholds are selected independently on the same inner validation
partition by end-to-end Macro F1 and are then frozen before outer evaluation.
This within-run control isolates which Stage 1 output is the better operational
router while sharing the exact same Stage 2 model and predictions.
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.callbacks import EarlyStopping

from run_hierarchical_cv import (
    CONFIG as BASE_CONFIG,
    build_parallel_model,
    fit_tokenizer,
    load_frame,
    print_gate_coverage,
    print_stage1_metrics,
    print_stage2_metrics,
    vectorize,
)
from hate_speech_detection.hierarchical_splits import (
    make_hierarchical_inner_split,
    make_hierarchical_splits,
)
from hate_speech_detection.target_strategy import (
    DEFAULT_GATE_THRESHOLD,
    DEFAULT_LABEL_THRESHOLD,
    STAGE2_TARGET_COLUMNS,
    analyze_gate_coverage,
    get_stage1_targets_with_route,
    get_stage2_binary_targets,
    get_stage2_targets,
)
from hate_speech_detection.threshold_selection import (
    apply_label_thresholds,
    select_label_thresholds,
    select_routing_threshold,
)


CONFIG = dict(BASE_CONFIG)
ROUTE_OUTPUT_INDEX = 2
TOXICITY_OUTPUT_INDEX = 0
SEVERE_OUTPUT_INDEX = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the controlled explicit Stage 1 routing-head experiment."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=CONFIG["N_SPLITS"],
        help="Number of outer CV folds (default: 2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG["EPOCHS"],
        help="Maximum epochs per model; EarlyStopping may stop sooner.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional deterministic sample limit for smoke/diagnostic runs.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    config = dict(CONFIG)
    config["N_SPLITS"] = args.n_splits
    config["EPOCHS"] = args.epochs

    random_state = config["RANDOM_STATE"]
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    frame = load_frame()
    if args.max_samples is not None and args.max_samples < len(frame):
        frame = (
            frame.sample(n=args.max_samples, random_state=random_state)
            .reset_index(drop=True)
        )
        print(
            f"Diagnostic sample enabled: {len(frame)} rows. "
            "Do not report sampled-run metrics as full-data evidence."
        )

    coverage = analyze_gate_coverage(
        frame,
        gate_threshold=config["GATE_THRESHOLD"],
        label_threshold=config["LABEL_THRESHOLD"],
    )
    print_gate_coverage(coverage)

    splits = make_hierarchical_splits(
        frame,
        n_splits=config["N_SPLITS"],
        random_state=random_state,
        gate_threshold=config["GATE_THRESHOLD"],
        label_threshold=config["LABEL_THRESHOLD"],
    )

    gate_true_parts = []
    toxicity_score_parts = []
    route_score_parts = []
    toxicity_pred_parts = []
    route_pred_parts = []
    severe_true_parts = []
    severe_pred_parts = []

    oracle_true_parts = []
    oracle_score_parts = []
    oracle_pred_parts = []

    end_to_end_true_parts = []
    toxicity_cascade_pred_parts = []
    route_cascade_pred_parts = []

    threshold_summaries = []

    for fold_number, split in enumerate(splits, 1):
        print(f"\nFold {fold_number}/{config['N_SPLITS']}")
        fold_seed = random_state + fold_number
        np.random.seed(fold_seed)
        tf.random.set_seed(fold_seed)

        outer_frame = frame.iloc[split.validation_indices]
        inner = make_hierarchical_inner_split(
            frame,
            split.train_indices,
            validation_fraction=config["INNER_VALIDATION_FRACTION"],
            random_state=fold_seed,
            gate_threshold=config["GATE_THRESHOLD"],
            label_threshold=config["LABEL_THRESHOLD"],
        )

        stage1_train_frame = frame.iloc[inner.train_indices]
        inner_validation_frame = frame.iloc[inner.validation_indices]
        stage2_train_frame = frame.iloc[inner.stage2_train_indices]
        stage2_stop_frame = frame.iloc[inner.stage2_validation_indices]

        print(
            "Stage 1 rows: "
            f"fit={len(stage1_train_frame)}, "
            f"inner_validation={len(inner_validation_frame)}, "
            f"outer_eval={len(outer_frame)}"
        )
        print(
            "Stage 2 rows: "
            f"fit={len(stage2_train_frame)}, "
            f"inner_validation={len(stage2_stop_frame)}, "
            f"outer_eval={len(split.stage2_validation_indices)}"
        )

        # Stage 1: same historical encoder, now with three sigmoid outputs:
        # toxicity soft, severe_toxicity soft, and explicit binary route.
        stage1_tokenizer = fit_tokenizer(stage1_train_frame["text"], config)
        x_stage1_train = vectorize(
            stage1_tokenizer,
            stage1_train_frame["text"],
            config,
        )
        x_stage1_stop = vectorize(
            stage1_tokenizer,
            inner_validation_frame["text"],
            config,
        )
        x_stage1_outer = vectorize(
            stage1_tokenizer,
            outer_frame["text"],
            config,
        )

        y_stage1_train = get_stage1_targets_with_route(
            stage1_train_frame,
            gate_threshold=config["GATE_THRESHOLD"],
        )
        y_stage1_stop = get_stage1_targets_with_route(
            inner_validation_frame,
            gate_threshold=config["GATE_THRESHOLD"],
        )
        y_stage1_outer = get_stage1_targets_with_route(
            outer_frame,
            gate_threshold=config["GATE_THRESHOLD"],
        )

        stage1_model = build_parallel_model(
            config,
            n_outputs=3,
            name=f"stage1_route_head_fold_{fold_number}",
        )
        stage1_history = stage1_model.fit(
            x_stage1_train,
            y_stage1_train,
            validation_data=(x_stage1_stop, y_stage1_stop),
            epochs=config["EPOCHS"],
            batch_size=config["BATCH_SIZE"],
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
            verbose=1,
        )
        stage1_best_epoch = int(np.argmin(stage1_history.history["val_loss"])) + 1
        print(f"Stage 1 best inner-validation epoch: {stage1_best_epoch}")

        stage1_stop_probabilities = stage1_model.predict(x_stage1_stop, verbose=0)
        stage1_outer_probabilities = stage1_model.predict(x_stage1_outer, verbose=0)

        # Stage 2 remains unchanged from the nested-threshold experiment.
        stage2_tokenizer = fit_tokenizer(stage2_train_frame["text"], config)
        x_stage2_train = vectorize(
            stage2_tokenizer,
            stage2_train_frame["text"],
            config,
        )
        x_stage2_stop = vectorize(
            stage2_tokenizer,
            stage2_stop_frame["text"],
            config,
        )
        y_stage2_train = get_stage2_targets(stage2_train_frame)
        y_stage2_stop = get_stage2_targets(stage2_stop_frame)

        stage2_model = build_parallel_model(
            config,
            n_outputs=len(STAGE2_TARGET_COLUMNS),
            name=f"stage2_route_head_fold_{fold_number}",
        )
        stage2_history = stage2_model.fit(
            x_stage2_train,
            y_stage2_train,
            validation_data=(x_stage2_stop, y_stage2_stop),
            epochs=config["EPOCHS"],
            batch_size=config["BATCH_SIZE"],
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
            verbose=1,
        )
        stage2_best_epoch = int(np.argmin(stage2_history.history["val_loss"])) + 1
        print(f"Stage 2 best inner-validation epoch: {stage2_best_epoch}")

        stage2_stop_probabilities = stage2_model.predict(x_stage2_stop, verbose=0)
        stage2_stop_true = get_stage2_binary_targets(
            stage2_stop_frame,
            label_threshold=config["LABEL_THRESHOLD"],
        )
        label_selection = select_label_thresholds(
            stage2_stop_true,
            stage2_stop_probabilities,
            reference_threshold=config["LABEL_THRESHOLD"],
        )

        # One Stage 2 prediction matrix on the full common inner validation set
        # supports both Stage 1 routing-head comparisons.
        x_stage2_common_stop = vectorize(
            stage2_tokenizer,
            inner_validation_frame["text"],
            config,
        )
        stage2_common_stop_probabilities = stage2_model.predict(
            x_stage2_common_stop,
            verbose=0,
        )
        inner_gate_true = y_stage1_stop[:, ROUTE_OUTPUT_INDEX].astype(np.int8)
        inner_stage2_true = get_stage2_binary_targets(
            inner_validation_frame,
            label_threshold=config["LABEL_THRESHOLD"],
        )

        toxicity_selection = select_routing_threshold(
            inner_gate_true,
            stage1_stop_probabilities[:, TOXICITY_OUTPUT_INDEX],
            inner_stage2_true,
            stage2_common_stop_probabilities,
            label_selection.thresholds,
            reference_threshold=config["GATE_THRESHOLD"],
        )
        route_selection = select_routing_threshold(
            inner_gate_true,
            stage1_stop_probabilities[:, ROUTE_OUTPUT_INDEX],
            inner_stage2_true,
            stage2_common_stop_probabilities,
            label_selection.thresholds,
            reference_threshold=0.5,
        )

        label_threshold_text = ", ".join(
            f"{label}={threshold:.2f}"
            for label, threshold in zip(
                STAGE2_TARGET_COLUMNS,
                label_selection.thresholds,
            )
        )
        print(
            "Selected inner thresholds: "
            f"toxicity_router={toxicity_selection.threshold:.2f} "
            f"(e2e={toxicity_selection.macro_f1:.4f}, "
            f"recall={toxicity_selection.gate_recall:.4f}); "
            f"route_head={route_selection.threshold:.2f} "
            f"(e2e={route_selection.macro_f1:.4f}, "
            f"recall={route_selection.gate_recall:.4f}); "
            f"{label_threshold_text}"
        )
        threshold_summaries.append(
            (
                fold_number,
                toxicity_selection,
                route_selection,
                label_selection.thresholds.copy(),
            )
        )

        outer_gate_true = y_stage1_outer[:, ROUTE_OUTPUT_INDEX].astype(np.int8)
        toxicity_outer_scores = stage1_outer_probabilities[:, TOXICITY_OUTPUT_INDEX]
        route_outer_scores = stage1_outer_probabilities[:, ROUTE_OUTPUT_INDEX]
        toxicity_outer_pred = (
            toxicity_outer_scores >= toxicity_selection.threshold
        ).astype(np.int8)
        route_outer_pred = (
            route_outer_scores >= route_selection.threshold
        ).astype(np.int8)

        gate_true_parts.append(outer_gate_true)
        toxicity_score_parts.append(toxicity_outer_scores.astype(np.float32))
        route_score_parts.append(route_outer_scores.astype(np.float32))
        toxicity_pred_parts.append(toxicity_outer_pred)
        route_pred_parts.append(route_outer_pred)
        severe_true_parts.append(y_stage1_outer[:, SEVERE_OUTPUT_INDEX].astype(np.float32))
        severe_pred_parts.append(
            stage1_outer_probabilities[:, SEVERE_OUTPUT_INDEX].astype(np.float32)
        )

        oracle_positions = np.flatnonzero(outer_gate_true)
        toxicity_positions = np.flatnonzero(toxicity_outer_pred)
        route_positions = np.flatnonzero(route_outer_pred)
        required_positions = np.flatnonzero(
            outer_gate_true.astype(bool)
            | toxicity_outer_pred.astype(bool)
            | route_outer_pred.astype(bool)
        )

        required_frame = outer_frame.iloc[required_positions]
        x_stage2_required = vectorize(
            stage2_tokenizer,
            required_frame["text"],
            config,
        )
        required_probabilities = stage2_model.predict(x_stage2_required, verbose=0)
        required_lookup = np.full(len(outer_frame), -1, dtype=np.int64)
        required_lookup[required_positions] = np.arange(required_positions.size)

        oracle_probability_rows = required_lookup[oracle_positions]
        oracle_probabilities = required_probabilities[oracle_probability_rows]
        oracle_frame = outer_frame.iloc[oracle_positions]
        oracle_true = get_stage2_binary_targets(
            oracle_frame,
            label_threshold=config["LABEL_THRESHOLD"],
        )
        oracle_pred = apply_label_thresholds(
            oracle_probabilities,
            label_selection.thresholds,
        )
        oracle_true_parts.append(oracle_true)
        oracle_score_parts.append(oracle_probabilities.astype(np.float32))
        oracle_pred_parts.append(oracle_pred)

        end_to_end_true = get_stage2_binary_targets(
            outer_frame,
            label_threshold=config["LABEL_THRESHOLD"],
        )
        toxicity_cascade_pred = np.zeros_like(end_to_end_true, dtype=np.int8)
        route_cascade_pred = np.zeros_like(end_to_end_true, dtype=np.int8)

        if toxicity_positions.size:
            probability_rows = required_lookup[toxicity_positions]
            toxicity_cascade_pred[toxicity_positions] = apply_label_thresholds(
                required_probabilities[probability_rows],
                label_selection.thresholds,
            )

        if route_positions.size:
            probability_rows = required_lookup[route_positions]
            route_cascade_pred[route_positions] = apply_label_thresholds(
                required_probabilities[probability_rows],
                label_selection.thresholds,
            )

        end_to_end_true_parts.append(end_to_end_true)
        toxicity_cascade_pred_parts.append(toxicity_cascade_pred)
        route_cascade_pred_parts.append(route_cascade_pred)

        del (
            stage1_model,
            stage2_model,
            stage1_tokenizer,
            stage2_tokenizer,
            x_stage1_train,
            x_stage1_stop,
            x_stage1_outer,
            x_stage2_train,
            x_stage2_stop,
            x_stage2_common_stop,
            x_stage2_required,
            required_probabilities,
            stage2_common_stop_probabilities,
        )
        keras_backend.clear_session()
        gc.collect()

    gate_true = np.concatenate(gate_true_parts)
    toxicity_scores = np.concatenate(toxicity_score_parts)
    route_scores = np.concatenate(route_score_parts)
    toxicity_pred = np.concatenate(toxicity_pred_parts)
    route_pred = np.concatenate(route_pred_parts)
    severe_true = np.concatenate(severe_true_parts)
    severe_pred = np.concatenate(severe_pred_parts)

    oracle_true = np.concatenate(oracle_true_parts)
    oracle_scores = np.concatenate(oracle_score_parts)
    oracle_pred = np.concatenate(oracle_pred_parts)

    end_to_end_true = np.concatenate(end_to_end_true_parts)
    toxicity_cascade_pred = np.concatenate(toxicity_cascade_pred_parts)
    route_cascade_pred = np.concatenate(route_cascade_pred_parts)

    print("\n--- Threshold selection summary ---")
    for fold_number, toxicity_sel, route_sel, label_thresholds in threshold_summaries:
        print(
            f"Fold {fold_number}: toxicity_router={toxicity_sel.threshold:.2f} "
            f"(inner e2e={toxicity_sel.macro_f1:.4f}, "
            f"gate recall={toxicity_sel.gate_recall:.4f}); "
            f"route_head={route_sel.threshold:.2f} "
            f"(inner e2e={route_sel.macro_f1:.4f}, "
            f"gate recall={route_sel.gate_recall:.4f})"
        )
        for label, threshold in zip(STAGE2_TARGET_COLUMNS, label_thresholds):
            print(f"  {label}: threshold={threshold:.2f}")

    print_stage1_metrics(
        "Stage 1 — fractional toxicity output as router",
        gate_true,
        toxicity_pred,
        toxicity_scores,
        severe_true,
        severe_pred,
    )
    print_stage1_metrics(
        "Stage 1 — explicit binary route output as router",
        gate_true,
        route_pred,
        route_scores,
        severe_true,
        severe_pred,
    )

    print_stage2_metrics(
        "Stage 2 oracle — nested tuned label thresholds",
        oracle_true,
        oracle_pred,
        oracle_scores,
    )

    print_stage2_metrics(
        "End-to-end — fractional toxicity router",
        end_to_end_true,
        toxicity_cascade_pred,
    )
    print_stage2_metrics(
        "End-to-end — explicit binary route head",
        end_to_end_true,
        route_cascade_pred,
    )

    toxicity_macro = f1_score(
        end_to_end_true,
        toxicity_cascade_pred,
        average="macro",
        zero_division=0,
    )
    route_macro = f1_score(
        end_to_end_true,
        route_cascade_pred,
        average="macro",
        zero_division=0,
    )
    print("\n--- Routing-head experiment comparison ---")
    print(f"Fractional toxicity router Macro F1: {toxicity_macro:.4f}")
    print(f"Explicit binary route head Macro F1: {route_macro:.4f}")
    print(f"Delta (route - toxicity): {route_macro - toxicity_macro:+.4f}")
    print("Previous PR #10 nested-threshold reference Macro F1: 0.4412")


if __name__ == "__main__":
    main()
