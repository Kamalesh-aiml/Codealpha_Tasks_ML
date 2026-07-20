"""
preprocess.py
=============
Orchestrates the full preprocessing pipeline:

    1. Discover and label all audio files (data_loader).
    2. Extract MFCC features for every file (feature_extraction).
    3. Encode string emotion labels into integers.
    4. Split into train / test sets.
    5. Save all arrays and the fitted LabelEncoder to disk.

Also exposes loader helpers used by train.py, evaluate.py, and predict.py
so artifact loading logic lives in exactly one place.

Run once before training:
    python -m src.preprocess
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
)
from src.data_loader import load_dataset
from src.feature_extraction import extract_features


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(labels: list[str]) -> tuple[np.ndarray, LabelEncoder]:
    """Encode string emotion labels into integer class indices.

    Args:
        labels (list[str]): Raw emotion strings, e.g. ``["happy", "sad"]``.

    Returns:
        tuple[np.ndarray, LabelEncoder]:
            - ``y_encoded``: integer array, shape ``(n,)``, dtype ``int32``.
            - ``label_encoder``: fitted encoder; save it so predict.py can
              call ``inverse_transform`` to recover emotion names later.

    Raises:
        ValueError: If ``labels`` is empty.
    """
    if not labels:
        raise ValueError("Cannot encode an empty label list.")

    encoder = LabelEncoder()
    y_encoded: np.ndarray = encoder.fit_transform(labels).astype(np.int32)
    return y_encoded, encoder


# ---------------------------------------------------------------------------
# Feature extraction loop
# ---------------------------------------------------------------------------

def extract_dataset_features(
    dataset_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Extract MFCC features for every row in the dataset DataFrame.

    Args:
        dataset_df (pd.DataFrame): Columns ``"filepath"`` and ``"emotion"``
            as returned by ``load_dataset()``.

    Returns:
        tuple[np.ndarray, list[str]]:
            - ``X``: float32 array, shape ``(n_samples, N_MFCC, MAX_PAD_LENGTH)``.
            - ``labels``: emotion strings in the same row order as ``X``.

    Raises:
        RuntimeError: If extraction fails for any file (includes filepath in
            the message so the bad file is easy to find).
    """
    total: int = len(dataset_df)
    features: list[np.ndarray] = []
    labels: list[str] = []

    for i, (_, row) in enumerate(dataset_df.iterrows(), start=1):
        filepath: str = row["filepath"]
        emotion: str  = row["emotion"]

        # Print progress every 100 files to avoid flooding the terminal.
        if i == 1 or i % 100 == 0 or i == total:
            print(f"  Processing file {i}/{total} ...")

        try:
            mfcc = extract_features(filepath)
        except (FileNotFoundError, RuntimeError) as error:
            raise RuntimeError(
                f"Feature extraction failed for '{filepath}': {error}"
            ) from error

        features.append(mfcc)
        labels.append(emotion)

    X: np.ndarray = np.array(features, dtype=np.float32)
    return X, labels


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_dataset(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split feature matrix and labels into stratified train and test sets.

    Stratification keeps class proportions equal in both partitions —
    important because RAVDESS "neutral" has half as many samples as other
    emotions.

    Args:
        X (np.ndarray): Feature matrix, shape ``(n, N_MFCC, MAX_PAD_LENGTH)``.
        y (np.ndarray): Integer labels, shape ``(n,)``.

    Returns:
        tuple: ``X_train, X_test, y_train, y_test``
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# Persist / load processed data
# ---------------------------------------------------------------------------

def save_processed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
) -> None:
    """Save split arrays and the fitted LabelEncoder to disk.

    Args:
        X_train: Training feature matrix.
        X_test:  Test feature matrix.
        y_train: Training labels.
        y_test:  Test labels.
        encoder: Fitted ``LabelEncoder`` to pickle.

    Raises:
        OSError: If any file cannot be written.
    """
    try:
        np.save(X_TRAIN_PATH, X_train)
        np.save(X_TEST_PATH,  X_test)
        np.save(Y_TRAIN_PATH, y_train)
        np.save(Y_TEST_PATH,  y_test)
    except OSError as error:
        raise OSError(f"Failed to save NumPy arrays: {error}") from error

    try:
        with open(LABEL_ENCODER_PATH, "wb") as fh:
            pickle.dump(encoder, fh)
    except OSError as error:
        raise OSError(
            f"Failed to save label encoder to '{LABEL_ENCODER_PATH}': {error}"
        ) from error


def load_processed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached train/test arrays from disk.

    Called by ``train.py`` and ``evaluate.py`` to skip re-extraction.

    Returns:
        tuple: ``X_train, X_test, y_train, y_test``

    Raises:
        FileNotFoundError: If any ``.npy`` file is missing (preprocessing
            has not been run yet).
    """
    for path in (X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Processed data not found: '{path}'. "
                "Run 'python -m src.preprocess' first."
            )

    return (
        np.load(X_TRAIN_PATH),
        np.load(X_TEST_PATH),
        np.load(Y_TRAIN_PATH),
        np.load(Y_TEST_PATH),
    )


def load_label_encoder() -> LabelEncoder:
    """Load the fitted LabelEncoder from disk.

    Used by ``evaluate.py`` and ``predict.py`` to decode integer
    predictions back into emotion name strings.

    Returns:
        LabelEncoder: The fitted encoder saved during preprocessing.

    Raises:
        FileNotFoundError: If the encoder file does not exist.
        RuntimeError: If the pickle file cannot be loaded.
    """
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Label encoder not found: '{LABEL_ENCODER_PATH}'. "
            "Run 'python -m src.preprocess' first."
        )

    try:
        with open(LABEL_ENCODER_PATH, "rb") as fh:
            return pickle.load(fh)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load label encoder: {error}"
        ) from error


def load_trained_model():
    """Load the saved Keras model from disk.

    Centralised here so both ``evaluate.py`` and ``predict.py`` share the
    same loading logic without importing from each other.

    Returns:
        tf.keras.Model: The trained model ready for inference.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If the model file cannot be loaded.
    """
    import tensorflow as tf  # Deferred import — TF is heavy; only load when needed.

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: '{MODEL_PATH}'. "
            "Run 'python -m src.train' first."
        )

    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load model from '{MODEL_PATH}': {error}"
        ) from error


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_dataset() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder
]:
    """Run the complete preprocessing pipeline end-to-end.

    Workflow:
        load dataset → extract features → encode labels →
        split → save → return artifacts

    Returns:
        tuple: ``X_train, X_test, y_train, y_test, label_encoder``

    Raises:
        FileNotFoundError: Raw dataset directory missing.
        ValueError: No audio files found or invalid filename format.
        RuntimeError: Feature extraction failed for a file.
        OSError: Could not write output files.
    """
    print("Loading dataset...")
    dataset_df: pd.DataFrame = load_dataset()
    print(f"\nDataset size: {len(dataset_df)} samples")

    print("\nExtracting features...")
    X, raw_labels = extract_dataset_features(dataset_df)

    print("\nEncoding labels...")
    y_encoded, encoder = encode_labels(raw_labels)
    print(f"Classes found: {list(encoder.classes_)}")

    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y_encoded)
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")
    print(f"\nClasses: {len(encoder.classes_)}")

    save_processed_data(X_train, X_test, y_train, y_test, encoder)
    print("\nProcessed data saved successfully.")

    return X_train, X_test, y_train, y_test, encoder


if __name__ == "__main__":
    preprocess_dataset()
