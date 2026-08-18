"""Run the hierarchical two-stage Civil Comments experiment.

Stage 1 trains on the full-dataset outer training partition with fractional
``toxicity`` and ``severe_toxicity`` targets. Its toxicity output is the
routing signal.

Stage 2 trains only on rows routed by the ground-truth training gate and keeps
five fine-grained targets as independent fractional outputs. No Stage 2
oversampling is performed.

Each outer training fold is split again into one common inner fit/validation
partition shared by both stages. Early stopping and threshold selection see
only this inner validation data; the outer validation fold remains untouched
until final metric computation.

The runner reports fixed-threshold reference metrics and nested threshold-tuned
metrics from the same trained models so threshold effects can be measured
without an extra training experiment.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from hate_speech_detection.hierarchical_splits import (
    make_hierarchical_inner_split,
    make_hierarchical_splits,
)
from hate_speech_detection.target_strategy import (
    ALL_TARGET_COLUMNS,
    DEFAULT_GATE_THRESHOLD,
    DEFAULT_LABEL_THRESHOLD,
    STAGE1_TARGET_COLUMNS,
    STAGE2_TARGET_COLUMNS,
    analyze_gate_coverage,
    get_stage1_targets,
    get_stage2_binary_targets,
    get_stage2_targets,
)
from hate_speech_detection.threshold_selection import (
    apply_label_thresholds,
    select_label_thresholds,
    select_routing_threshold,
)


CONFIG = {
    "MAX_WORDS": 20_000,
    "MAX_LEN": 50,
    "EMBEDDING_DIM": 128,
    "N_SPLITS": 2,
    "BATCH_SIZE": 128,
    "EPOCHS": 5,
    "INNER_VALIDATION_FRACTION": 0.10,
    "GATE_THRESHOLD": DEFAULT_GATE_THRESHOLD,
    "LABEL_THRESHOLD": DEFAULT_LABEL_THRESHOLD,
    "RANDOM_STATE": 42,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run leakage-safe hierarchical Civil Comments cross-validation."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=CONFIG["N_SPLITS"],
        help="Number of outer CV folds (default: 2 for fast initial experiments).",
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
        help=(
            "Optional deterministic sample limit for smoke/diagnostic runs. "
            "Metrics from a sampled run are not the full-dataset benchmark."
        ),
    )
    return parser.parse_args(argv)


def load_frame():
    """Load text plus the seven original fractional Civil Comments targets."""

    dataset = load_dataset("google/civil_comments", split="train").to_pandas()
    columns = ["text", *ALL_TARGET_COLUMNS]
    return dataset.loc[:, columns].reset_index(drop=True)


def build_parallel_model(config, *, n_outputs: int, name: str) -> Model:
    """Build the historical parallel CNN + Bi-LSTM encoder with sigmoid outputs."""

    inputs = Input(shape=(config["MAX_LEN"],), name=f"{name}_tokens")
    embedding = Embedding(
        config["MAX_WORDS"],
        config["EMBEDDING_DIM"],
        name=f"{name}_embedding",
    )(inputs)

    lstm_branch = Bidirectional(
        LSTM(32, return_sequences=False),
        name=f"{name}_bilstm",
    )(embedding)
    cnn_branches = [
        GlobalMaxPooling1D(name=f"{name}_pool_{kernel}")(
            Conv1D(32, kernel, activation="relu", name=f"{name}_conv_{kernel}")(
                embedding
            )
        )
        for kernel in (2, 3, 4)
    ]
    cnn_branch = concatenate(cnn_branches, name=f"{name}_cnn_concat")

    merged = concatenate([lstm_branch, cnn_branch], name=f"{name}_fusion")
    merged = Dropout(0.5, name=f"{name}_dropout")(merged)
    merged = Dense(64, activation="relu", name=f"{name}_dense")(merged)
    outputs = Dense(n_outputs, activation="sigmoid", name=f"{name}_outputs")(merged)

    model = Model(inputs, outputs, name=name)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
    )
    return model


def fit_tokenizer(texts, config) -> Tokenizer:
    tokenizer = Tokenizer(num_words=config["MAX_WORDS"])
    tokenizer.fit_on_texts(texts.tolist())
    return tokenizer


def vectorize(tokenizer: Tokenizer, texts, config) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts.tolist())
    return pad_sequences(sequences, maxlen=config["MAX_LEN"])


def print_gate_coverage(report) -> None:
    print("\n--- Ground-truth gate coverage analysis ---")
    print(
        f"Routed samples: {report['routed_samples']}/{report['total_samples']} "
        f"using toxicity >= {report['gate_threshold']:.2f}"
    )
    for label in STAGE2_TARGET_COLUMNS:
        stats = report["per_label"][label]
        print(
            f"{label}: positives={stats['positive_count']}, "
            f"missed={stats['missed_count']} "
            f"({stats['missed_rate']:.2%})"
        )
    overall = report["any_stage2_positive"]
    print(
        "any Stage 2 label: "
        f"positives={overall['positive_count']}, "
        f"missed={overall['missed_count']} ({overall['missed_rate']:.2%})"
    )


def safe_average_precision(y_true, scores) -> float:
    y_true = np.asarray(y_true)
    if np.sum(y_true) == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def safe_roc_auc(y_true, scores) -> float:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def format_metric(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.4f}"


def print_stage1_metrics(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    severe_true: np.ndarray,
    severe_pred: np.ndarray,
) -> None:
    print(f"\n--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Routing rate: {np.mean(y_pred):.4f}")
    print(
        "PR-AUC (average precision): "
        f"{format_metric(safe_average_precision(y_true, scores))}"
    )
    print(f"ROC-AUC: {format_metric(safe_roc_auc(y_true, scores))}")
    print(
        "severe_toxicity auxiliary MAE: "
        f"{mean_absolute_error(severe_true, severe_pred):.4f}"
    )


def print_stage2_metrics(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> None:
    print(f"\n--- {title} ---")
    print(
        "Macro F1: "
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )
    for index, label in enumerate(STAGE2_TARGET_COLUMNS):
        label_f1 = f1_score(
            y_true[:, index],
            y_pred[:, index],
            zero_division=0,
        )
        if scores is None:
            print(f"{label} F1: {label_f1:.4f}")
        else:
            label_ap = safe_average_precision(y_true[:, index], scores[:, index])
            print(
                f"{label} F1: {label_f1:.4f} | "
                f"PR-AUC/AP: {format_metric(label_ap)}"
            )


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
            "Do not report sampled-run metrics as the full-dataset benchmark."
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

    stage1_true_parts = []
    stage1_score_parts = []
    stage1_fixed_pred_parts = []
    stage1_tuned_pred_parts = []
    severe_true_parts = []
    severe_pred_parts = []

    oracle_true_parts = []
    oracle_score_parts = []
    oracle_fixed_pred_parts = []
    oracle_tuned_pred_parts = []

    end_to_end_true_parts = []
    end_to_end_fixed_pred_parts = []
    end_to_end_tuned_pred_parts = []

    threshold_summaries = []

    for fold_number, split in enumerate(splits, 1):
        print(f"\nFold {fold_number}/{config['N_SPLITS']}")
        fold_seed = random_state + fold_number
        np.random.seed(fold_seed)
        tf.random.set_seed(fold_seed)

        validation_frame = frame.iloc[split.validation_indices]

        inner = make_hierarchical_inner_split(
            frame,
            split.train_indices,
            validation_fraction=config["INNER_VALIDATION_FRACTION"],
            random_state=fold_seed,
            gate_threshold=config["GATE_THRESHOLD"],
            label_threshold=config["LABEL_THRESHOLD"],
        )

        stage1_train_frame = frame.iloc[inner.train_indices]
        stage1_stop_frame = frame.iloc[inner.validation_indices]
        stage2_train_frame = frame.iloc[inner.stage2_train_indices]
        stage2_stop_frame = frame.iloc[inner.stage2_validation_indices]

        print(
            "Stage 1 rows: "
            f"fit={len(stage1_train_frame)}, "
            f"inner_validation={len(stage1_stop_frame)}, "
            f"outer_eval={len(validation_frame)}"
        )
        print(
            "Stage 2 rows: "
            f"fit={len(stage2_train_frame)}, "
            f"inner_validation={len(stage2_stop_frame)}, "
            f"outer_eval={len(split.stage2_validation_indices)}"
        )

        # Stage 1 fitting and inner-validation predictions.
        stage1_tokenizer = fit_tokenizer(stage1_train_frame["text"], config)
        x_stage1_train = vectorize(
            stage1_tokenizer,
            stage1_train_frame["text"],
            config,
        )
        x_stage1_stop = vectorize(
            stage1_tokenizer,
            stage1_stop_frame["text"],
            config,
        )
        x_stage1_outer = vectorize(
            stage1_tokenizer,
            validation_frame["text"],
            config,
        )
        y_stage1_train = get_stage1_targets(stage1_train_frame)
        y_stage1_stop = get_stage1_targets(stage1_stop_frame)
        y_stage1_outer = get_stage1_targets(validation_frame)

        stage1_model = build_parallel_model(
            config,
            n_outputs=len(STAGE1_TARGET_COLUMNS),
            name=f"stage1_fold_{fold_number}",
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

        # Stage 2 fitting. Its validation rows are exactly the routed subset of
        # the same common inner validation partition used by Stage 1.
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
            name=f"stage2_fold_{fold_number}",
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

        # Select Stage 2 thresholds on routed inner-validation rows only.
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

        # Select the Stage 1 routing threshold on the full common inner
        # validation partition by the actual end-to-end Macro F1 objective.
        x_stage2_common_stop = vectorize(
            stage2_tokenizer,
            stage1_stop_frame["text"],
            config,
        )
        stage2_common_stop_probabilities = stage2_model.predict(
            x_stage2_common_stop,
            verbose=0,
        )
        inner_gate_true = (
            stage1_stop_frame["toxicity"].to_numpy()
            >= config["GATE_THRESHOLD"]
        ).astype(np.int8)
        inner_stage2_true = get_stage2_binary_targets(
            stage1_stop_frame,
            label_threshold=config["LABEL_THRESHOLD"],
        )
        routing_selection = select_routing_threshold(
            inner_gate_true,
            stage1_stop_probabilities[:, 0],
            inner_stage2_true,
            stage2_common_stop_probabilities,
            label_selection.thresholds,
            reference_threshold=config["GATE_THRESHOLD"],
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
            f"routing={routing_selection.threshold:.2f} "
            f"(inner end-to-end Macro F1={routing_selection.macro_f1:.4f}, "
            f"gate recall={routing_selection.gate_recall:.4f}, "
            f"routing rate={routing_selection.routing_rate:.4f}); "
            f"{label_threshold_text}"
        )
        threshold_summaries.append(
            (
                fold_number,
                routing_selection,
                label_selection.thresholds.copy(),
                label_selection.f1_scores.copy(),
            )
        )

        # Stage 1 outer evaluation: compare the historical fixed prediction
        # threshold with the inner-selected routing threshold.
        outer_gate_true = (
            validation_frame["toxicity"].to_numpy()
            >= config["GATE_THRESHOLD"]
        ).astype(np.int8)
        fixed_gate_pred = (
            stage1_outer_probabilities[:, 0] >= config["GATE_THRESHOLD"]
        ).astype(np.int8)
        tuned_gate_pred = (
            stage1_outer_probabilities[:, 0] >= routing_selection.threshold
        ).astype(np.int8)

        stage1_true_parts.append(outer_gate_true)
        stage1_score_parts.append(stage1_outer_probabilities[:, 0].astype(np.float32))
        stage1_fixed_pred_parts.append(fixed_gate_pred)
        stage1_tuned_pred_parts.append(tuned_gate_pred)
        severe_true_parts.append(y_stage1_outer[:, 1].astype(np.float32))
        severe_pred_parts.append(stage1_outer_probabilities[:, 1].astype(np.float32))

        # Predict Stage 2 only once for the union of rows needed by oracle,
        # fixed cascade, or tuned cascade evaluation.
        oracle_positions = np.flatnonzero(outer_gate_true)
        fixed_positions = np.flatnonzero(fixed_gate_pred)
        tuned_positions = np.flatnonzero(tuned_gate_pred)
        required_positions = np.flatnonzero(
            outer_gate_true.astype(bool)
            | fixed_gate_pred.astype(bool)
            | tuned_gate_pred.astype(bool)
        )

        required_frame = validation_frame.iloc[required_positions]
        x_stage2_required = vectorize(
            stage2_tokenizer,
            required_frame["text"],
            config,
        )
        required_probabilities = stage2_model.predict(
            x_stage2_required,
            verbose=0,
        )

        # Map local outer-validation positions to rows in required_probabilities.
        required_lookup = np.full(len(validation_frame), -1, dtype=np.int64)
        required_lookup[required_positions] = np.arange(required_positions.size)

        oracle_probability_rows = required_lookup[oracle_positions]
        oracle_probabilities = required_probabilities[oracle_probability_rows]
        oracle_frame = validation_frame.iloc[oracle_positions]
        oracle_true = get_stage2_binary_targets(
            oracle_frame,
            label_threshold=config["LABEL_THRESHOLD"],
        )
        oracle_fixed_pred = (
            oracle_probabilities >= config["LABEL_THRESHOLD"]
        ).astype(np.int8)
        oracle_tuned_pred = apply_label_thresholds(
            oracle_probabilities,
            label_selection.thresholds,
        )

        oracle_true_parts.append(oracle_true)
        oracle_score_parts.append(oracle_probabilities.astype(np.float32))
        oracle_fixed_pred_parts.append(oracle_fixed_pred)
        oracle_tuned_pred_parts.append(oracle_tuned_pred)

        end_to_end_true = get_stage2_binary_targets(
            validation_frame,
            label_threshold=config["LABEL_THRESHOLD"],
        )
        end_to_end_fixed_pred = np.zeros_like(end_to_end_true, dtype=np.int8)
        end_to_end_tuned_pred = np.zeros_like(end_to_end_true, dtype=np.int8)

        if fixed_positions.size:
            fixed_probability_rows = required_lookup[fixed_positions]
            fixed_probabilities = required_probabilities[fixed_probability_rows]
            end_to_end_fixed_pred[fixed_positions] = (
                fixed_probabilities >= config["LABEL_THRESHOLD"]
            ).astype(np.int8)

        if tuned_positions.size:
            tuned_probability_rows = required_lookup[tuned_positions]
            tuned_probabilities = required_probabilities[tuned_probability_rows]
            end_to_end_tuned_pred[tuned_positions] = apply_label_thresholds(
                tuned_probabilities,
                label_selection.thresholds,
            )

        end_to_end_true_parts.append(end_to_end_true)
        end_to_end_fixed_pred_parts.append(end_to_end_fixed_pred)
        end_to_end_tuned_pred_parts.append(end_to_end_tuned_pred)

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

    stage1_true = np.concatenate(stage1_true_parts)
    stage1_scores = np.concatenate(stage1_score_parts)
    stage1_fixed_pred = np.concatenate(stage1_fixed_pred_parts)
    stage1_tuned_pred = np.concatenate(stage1_tuned_pred_parts)
    severe_true = np.concatenate(severe_true_parts)
    severe_pred = np.concatenate(severe_pred_parts)

    oracle_true = np.concatenate(oracle_true_parts)
    oracle_scores = np.concatenate(oracle_score_parts)
    oracle_fixed_pred = np.concatenate(oracle_fixed_pred_parts)
    oracle_tuned_pred = np.concatenate(oracle_tuned_pred_parts)

    end_to_end_true = np.concatenate(end_to_end_true_parts)
    end_to_end_fixed_pred = np.concatenate(end_to_end_fixed_pred_parts)
    end_to_end_tuned_pred = np.concatenate(end_to_end_tuned_pred_parts)

    print("\n--- Threshold selection summary ---")
    for fold_number, routing, label_thresholds, label_f1_scores in threshold_summaries:
        print(
            f"Fold {fold_number}: routing={routing.threshold:.2f}, "
            f"inner end-to-end Macro F1={routing.macro_f1:.4f}, "
            f"inner gate recall={routing.gate_recall:.4f}, "
            f"inner routing rate={routing.routing_rate:.4f}"
        )
        for label, threshold, inner_f1 in zip(
            STAGE2_TARGET_COLUMNS,
            label_thresholds,
            label_f1_scores,
        ):
            print(
                f"  {label}: threshold={threshold:.2f}, "
                f"inner oracle F1={inner_f1:.4f}"
            )

    print_stage1_metrics(
        "Stage 1 toxicity gate — fixed prediction threshold 0.40",
        stage1_true,
        stage1_fixed_pred,
        stage1_scores,
        severe_true,
        severe_pred,
    )
    print_stage1_metrics(
        "Stage 1 toxicity gate — nested tuned routing threshold",
        stage1_true,
        stage1_tuned_pred,
        stage1_scores,
        severe_true,
        severe_pred,
    )

    print_stage2_metrics(
        "Stage 2 oracle — fixed label threshold 0.50",
        oracle_true,
        oracle_fixed_pred,
        oracle_scores,
    )
    print_stage2_metrics(
        "Stage 2 oracle — nested tuned label thresholds",
        oracle_true,
        oracle_tuned_pred,
        oracle_scores,
    )

    print_stage2_metrics(
        "End-to-end — fixed 0.40 routing / 0.50 labels",
        end_to_end_true,
        end_to_end_fixed_pred,
    )
    print_stage2_metrics(
        "End-to-end — nested tuned thresholds",
        end_to_end_true,
        end_to_end_tuned_pred,
    )


if __name__ == "__main__":
    main()
