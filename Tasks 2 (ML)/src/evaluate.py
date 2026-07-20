"""
evaluate.py
===========
Evaluates the trained model on the held-out test set and saves a
confusion matrix plot.

Dependencies:
    - preprocess.py  → load_processed_data(), load_label_encoder(),
                       load_trained_model()
    - feature_extraction.py → reshape_for_cnn()

Run:
    python -m src.evaluate
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from src.config import CONFUSION_MATRIX_PATH
from src.feature_extraction import reshape_for_cnn
from src.preprocess import load_label_encoder, load_processed_data, load_trained_model


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
) -> None:
    """Print test accuracy, loss, and a per-class classification report.

    Args:
        model (tf.keras.Model): Trained model.
        X_test (np.ndarray): Test features, CNN-ready shape
            ``(n, N_MFCC, MAX_PAD_LENGTH, 1)``.
        y_test (np.ndarray): True integer labels, shape ``(n,)``.
        encoder (LabelEncoder): Fitted encoder for label decoding.

    Raises:
        RuntimeError: If model evaluation or prediction fails.
    """
    try:
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    except Exception as error:
        raise RuntimeError(f"Model evaluation failed: {error}") from error

    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}\n")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Classification Report:")
    print(
        classification_report(
            y_test, y_pred,
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    plot_confusion_matrix(y_test, y_pred, encoder.classes_)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
) -> None:
    """Compute and save a labelled confusion matrix heatmap.

    Args:
        y_true (np.ndarray): Ground-truth integer labels.
        y_pred (np.ndarray): Predicted integer labels.
        class_names (np.ndarray): Emotion name strings in label index order.

    Raises:
        OSError: If the image cannot be saved to disk.
    """
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("True Emotion")
    ax.set_title("Confusion Matrix - Speech Emotion Recognition")
    fig.tight_layout()

    try:
        fig.savefig(CONFUSION_MATRIX_PATH)
    except OSError as error:
        raise OSError(f"Failed to save confusion matrix: {error}") from error
    finally:
        plt.close(fig)

    print(f"Confusion matrix saved -> '{CONFUSION_MATRIX_PATH}'")


def run_evaluation_pipeline() -> None:
    """Execute the full evaluation pipeline.

    Raises:
        FileNotFoundError: Model, encoder, or processed data missing.
        RuntimeError: Evaluation or prediction fails.
    """
    print("Step 1/3  Loading model and label encoder...")
    model   = load_trained_model()
    encoder = load_label_encoder()

    print("Step 2/3  Loading test data...")
    _, X_test, _, y_test = load_processed_data()
    X_test = reshape_for_cnn(X_test)
    print(f"  Test samples : {X_test.shape[0]}\n")

    print("Step 3/3  Evaluating...\n")
    evaluate_model(model, X_test, y_test, encoder)


if __name__ == "__main__":
    run_evaluation_pipeline()
