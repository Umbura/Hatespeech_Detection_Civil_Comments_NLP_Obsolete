"""Run the repaired leakage-safe CNN + Bi-LSTM cross-validation experiment.

This runner preserves the historical target-label rule and model architecture
while changing the evaluation order so validation data remains untouched.
It intentionally does not claim to solve the confirmed overfitting or the
unresolved target-label design documented in ``docs/KNOWN_ISSUES.md``.
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
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
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
from tensorflow.keras.utils import to_categorical

from hate_speech_detection.cv_pipeline import make_stratified_splits, prepare_fold


CONFIG = {
    "MAX_WORDS": 20_000,
    "MAX_LEN": 50,
    "EMBEDDING_DIM": 128,
    "N_SPLITS": 5,
    "BATCH_SIZE": 128,
    "EPOCHS": 5,
    "TARGET_SAMPLES": 5_000,
    "TOXICITY_THRESHOLD": 0.5,
    "RANDOM_STATE": 42,
}

LABEL_COLUMNS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


def load_labeled_data():
    """Load Civil Comments and preserve the historical single-label rule."""

    dataset = load_dataset("civil_comments", split="train").to_pandas()
    dataset["non_toxic"] = (dataset[LABEL_COLUMNS].sum(axis=1) == 0).astype(int)

    def get_label(row):
        for label in LABEL_COLUMNS:
            if row[label] > CONFIG["TOXICITY_THRESHOLD"]:
                return label
        return "non_toxic"

    dataset["label"] = dataset.apply(get_label, axis=1)
    return dataset[["text", "label"]].copy()


def build_model(config, n_classes):
    """Build the historical parallel CNN + Bi-LSTM architecture."""

    inputs = Input(shape=(config["MAX_LEN"],))
    embedding = Embedding(config["MAX_WORDS"], config["EMBEDDING_DIM"])(inputs)

    lstm_branch = Bidirectional(LSTM(32, return_sequences=False))(embedding)
    cnn_branches = [
        GlobalMaxPooling1D()(Conv1D(32, kernel, activation="relu")(embedding))
        for kernel in (2, 3, 4)
    ]
    cnn_branch = concatenate(cnn_branches)

    merged = concatenate([lstm_branch, cnn_branch])
    merged = Dropout(0.5)(merged)
    merged = Dense(64, activation="relu")(merged)
    outputs = Dense(n_classes, activation="softmax")(merged)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    random_state = CONFIG["RANDOM_STATE"]
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    frame = load_labeled_data()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(frame["label"])
    n_classes = len(label_encoder.classes_)

    splits = make_stratified_splits(
        encoded_labels,
        n_splits=CONFIG["N_SPLITS"],
        random_state=random_state,
    )

    all_true = []
    all_pred = []
    fold_scores = []

    for fold_number, (train_indices, validation_indices) in enumerate(splits, 1):
        print(f"\nFold {fold_number}/{CONFIG['N_SPLITS']}")

        prepared = prepare_fold(
            frame,
            train_indices,
            validation_indices,
            target_samples=CONFIG["TARGET_SAMPLES"],
            tokenizer_factory=Tokenizer,
            num_words=CONFIG["MAX_WORDS"],
            random_state=random_state,
        )

        train_sequences = prepared.tokenizer.texts_to_sequences(
            prepared.train_frame["text"]
        )
        validation_sequences = prepared.tokenizer.texts_to_sequences(
            prepared.validation_frame["text"]
        )

        x_train = pad_sequences(train_sequences, maxlen=CONFIG["MAX_LEN"])
        x_validation = pad_sequences(
            validation_sequences,
            maxlen=CONFIG["MAX_LEN"],
        )

        y_train_raw = label_encoder.transform(prepared.train_frame["label"])
        y_validation_raw = label_encoder.transform(
            prepared.validation_frame["label"]
        )
        y_train = to_categorical(y_train_raw, num_classes=n_classes)
        y_validation = to_categorical(y_validation_raw, num_classes=n_classes)

        weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train_raw),
            y=y_train_raw,
        )
        class_weights = dict(zip(np.unique(y_train_raw), weights))

        model = build_model(CONFIG, n_classes)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=CONFIG["EPOCHS"],
            batch_size=CONFIG["BATCH_SIZE"],
            callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
            class_weight=class_weights,
            verbose=1,
        )

        predictions = np.argmax(model.predict(x_validation, verbose=0), axis=1)
        fold_f1 = f1_score(y_validation_raw, predictions, average="macro")
        fold_scores.append(fold_f1)
        all_true.extend(y_validation_raw)
        all_pred.extend(predictions)
        print(f"Fold {fold_number} macro F1: {fold_f1:.4f}")

    overall_accuracy = accuracy_score(all_true, all_pred)
    overall_f1 = f1_score(all_true, all_pred, average="macro")
    overall_recall = recall_score(all_true, all_pred, average="macro")

    print("\n--- Leakage-safe cross-validation summary ---")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Macro F1: {overall_f1:.4f}")
    print(f"Macro Recall: {overall_recall:.4f}")
    print(f"Mean fold macro F1: {np.mean(fold_scores):.4f}")


if __name__ == "__main__":
    main()
