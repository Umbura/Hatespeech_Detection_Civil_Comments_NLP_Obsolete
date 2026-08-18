"""Run the hierarchical two-stage Civil Comments experiment.

Stage 1 trains on the full dataset with fractional ``toxicity`` and
``severe_toxicity`` targets. Its toxicity output is the routing signal.

Stage 2 trains only on rows routed by the ground-truth training gate and keeps
five fine-grained targets as independent fractional outputs. No Stage 2
oversampling is performed.

The runner reports three different views and must not conflate them:

1. Stage 1 toxicity-gate performance;
2. Stage 2 oracle performance on ground-truth routed validation rows;
3. end-to-end subtype performance after predicted Stage 1 routing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, recall_score
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

from hate_speech_detection.hierarchical_splits import make_hierarchical_splits
from hate_speech_detection.target_strategy import (
    ALL_TARGET_COLUMNS,
    STAGE1_TARGET_COLUMNS,
    STAGE2_TARGET_COLUMNS,
    analyze_gate_coverage,
    get_stage1_targets,
    get_stage2_binary_targets,
    get_stage2_targets,
)


CONFIG = {
    "MAX_WORDS": 20_000,
    "MAX_LEN": 50,
    "EMBEDDING_DIM": 128,
    "N_SPLITS": 5,
    "BATCH_SIZE": 128,
    "EPOCHS": 5,
    "GATE_THRESHOLD": 0.5,
    "LABEL_THRESHOLD": 0.5,
    "RANDOM_STATE": 42,
}


def load_frame():
    """Load the text plus the seven original fractional Civil Comments targets."""

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


def fit_tokenizer(texts) -> Tokenizer:
    tokenizer = Tokenizer(num_words=CONFIG["MAX_WORDS"])
    tokenizer.fit_on_texts(texts.tolist())
    return tokenizer


def vectorize(tokenizer: Tokenizer, texts) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts.tolist())
    return pad_sequences(sequences, maxlen=CONFIG["MAX_LEN"])


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


def main() -> None:
    random_state = CONFIG["RANDOM_STATE"]
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    frame = load_frame()
    coverage = analyze_gate_coverage(
        frame,
        gate_threshold=CONFIG["GATE_THRESHOLD"],
        label_threshold=CONFIG["LABEL_THRESHOLD"],
    )
    print_gate_coverage(coverage)

    splits = make_hierarchical_splits(
        frame,
        n_splits=CONFIG["N_SPLITS"],
        random_state=random_state,
        gate_threshold=CONFIG["GATE_THRESHOLD"],
        label_threshold=CONFIG["LABEL_THRESHOLD"],
    )

    stage1_true = []
    stage1_pred = []
    severe_true = []
    severe_pred = []
    oracle_true = []
    oracle_pred = []
    end_to_end_true = []
    end_to_end_pred = []

    for fold_number, split in enumerate(splits, 1):
        print(f"\nFold {fold_number}/{CONFIG['N_SPLITS']}")
        fold_seed = random_state + fold_number
        np.random.seed(fold_seed)
        tf.random.set_seed(fold_seed)

        train_frame = frame.iloc[split.train_indices]
        validation_frame = frame.iloc[split.validation_indices]

        # Stage 1: tokenizer and learned preprocessing see training rows only.
        stage1_tokenizer = fit_tokenizer(train_frame["text"])
        x_stage1_train = vectorize(stage1_tokenizer, train_frame["text"])
        x_stage1_validation = vectorize(stage1_tokenizer, validation_frame["text"])
        y_stage1_train = get_stage1_targets(train_frame)
        y_stage1_validation = get_stage1_targets(validation_frame)

        stage1_model = build_parallel_model(
            CONFIG,
            n_outputs=len(STAGE1_TARGET_COLUMNS),
            name=f"stage1_fold_{fold_number}",
        )
        stage1_model.fit(
            x_stage1_train,
            y_stage1_train,
            validation_data=(x_stage1_validation, y_stage1_validation),
            epochs=CONFIG["EPOCHS"],
            batch_size=CONFIG["BATCH_SIZE"],
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
            verbose=1,
        )
        stage1_probabilities = stage1_model.predict(
            x_stage1_validation,
            verbose=0,
        )

        true_gate = (
            validation_frame["toxicity"].to_numpy()
            >= CONFIG["GATE_THRESHOLD"]
        ).astype(np.int8)
        predicted_gate = (
            stage1_probabilities[:, 0] >= CONFIG["GATE_THRESHOLD"]
        ).astype(np.int8)
        stage1_true.extend(true_gate.tolist())
        stage1_pred.extend(predicted_gate.tolist())
        severe_true.extend(y_stage1_validation[:, 1].tolist())
        severe_pred.extend(stage1_probabilities[:, 1].tolist())

        # Stage 2: no oversampling. The tokenizer is fitted only on routed
        # training rows from the current outer fold.
        stage2_train_frame = frame.iloc[split.stage2_train_indices]
        stage2_validation_frame = frame.iloc[split.stage2_validation_indices]
        stage2_tokenizer = fit_tokenizer(stage2_train_frame["text"])
        x_stage2_train = vectorize(stage2_tokenizer, stage2_train_frame["text"])
        x_stage2_validation = vectorize(
            stage2_tokenizer,
            stage2_validation_frame["text"],
        )
        y_stage2_train = get_stage2_targets(stage2_train_frame)
        y_stage2_validation = get_stage2_targets(stage2_validation_frame)

        stage2_model = build_parallel_model(
            CONFIG,
            n_outputs=len(STAGE2_TARGET_COLUMNS),
            name=f"stage2_fold_{fold_number}",
        )
        stage2_model.fit(
            x_stage2_train,
            y_stage2_train,
            validation_data=(x_stage2_validation, y_stage2_validation),
            epochs=CONFIG["EPOCHS"],
            batch_size=CONFIG["BATCH_SIZE"],
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
            verbose=1,
        )

        oracle_probabilities = stage2_model.predict(
            x_stage2_validation,
            verbose=0,
        )
        oracle_true_binary = get_stage2_binary_targets(
            stage2_validation_frame,
            label_threshold=CONFIG["LABEL_THRESHOLD"],
        )
        oracle_pred_binary = (
            oracle_probabilities >= CONFIG["LABEL_THRESHOLD"]
        ).astype(np.int8)
        oracle_true.extend(oracle_true_binary.tolist())
        oracle_pred.extend(oracle_pred_binary.tolist())

        # End-to-end: only samples routed by the predicted Stage 1 gate receive
        # Stage 2 predictions. All other subtype outputs remain zero.
        routed_positions = np.flatnonzero(predicted_gate)
        end_to_end_fold_pred = np.zeros(
            (len(validation_frame), len(STAGE2_TARGET_COLUMNS)),
            dtype=np.int8,
        )
        if routed_positions.size:
            routed_frame = validation_frame.iloc[routed_positions]
            x_routed = vectorize(stage2_tokenizer, routed_frame["text"])
            routed_probabilities = stage2_model.predict(x_routed, verbose=0)
            end_to_end_fold_pred[routed_positions] = (
                routed_probabilities >= CONFIG["LABEL_THRESHOLD"]
            ).astype(np.int8)

        end_to_end_fold_true = get_stage2_binary_targets(
            validation_frame,
            label_threshold=CONFIG["LABEL_THRESHOLD"],
        )
        end_to_end_true.extend(end_to_end_fold_true.tolist())
        end_to_end_pred.extend(end_to_end_fold_pred.tolist())

        keras_backend.clear_session()

    stage1_true_array = np.asarray(stage1_true, dtype=np.int8)
    stage1_pred_array = np.asarray(stage1_pred, dtype=np.int8)
    oracle_true_array = np.asarray(oracle_true, dtype=np.int8)
    oracle_pred_array = np.asarray(oracle_pred, dtype=np.int8)
    end_to_end_true_array = np.asarray(end_to_end_true, dtype=np.int8)
    end_to_end_pred_array = np.asarray(end_to_end_pred, dtype=np.int8)

    print("\n--- Stage 1 toxicity gate ---")
    print(f"Accuracy: {accuracy_score(stage1_true_array, stage1_pred_array):.4f}")
    print(f"F1: {f1_score(stage1_true_array, stage1_pred_array):.4f}")
    print(f"Recall: {recall_score(stage1_true_array, stage1_pred_array):.4f}")
    print(
        "severe_toxicity auxiliary MAE: "
        f"{mean_absolute_error(severe_true, severe_pred):.4f}"
    )

    print("\n--- Stage 2 oracle (ground-truth routing) ---")
    print(
        "Macro F1: "
        f"{f1_score(oracle_true_array, oracle_pred_array, average='macro', zero_division=0):.4f}"
    )
    for index, label in enumerate(STAGE2_TARGET_COLUMNS):
        score = f1_score(
            oracle_true_array[:, index],
            oracle_pred_array[:, index],
            zero_division=0,
        )
        print(f"{label} F1: {score:.4f}")

    print("\n--- End-to-end (predicted Stage 1 routing) ---")
    print(
        "Macro F1: "
        f"{f1_score(end_to_end_true_array, end_to_end_pred_array, average='macro', zero_division=0):.4f}"
    )
    for index, label in enumerate(STAGE2_TARGET_COLUMNS):
        score = f1_score(
            end_to_end_true_array[:, index],
            end_to_end_pred_array[:, index],
            zero_division=0,
        )
        print(f"{label} F1: {score:.4f}")


if __name__ == "__main__":
    main()
