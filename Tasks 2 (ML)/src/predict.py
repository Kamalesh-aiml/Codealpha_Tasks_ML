"""
predict.py
==========
Runs inference on a single new audio file using the trained model.

Uses the identical feature extraction pipeline as training (same
function, same config constants) to guarantee train/inference
consistency — a common source of production bugs.

Run:
    python -m src.predict <path_to_wav_file>
"""

import sys
from pathlib import Path

import numpy as np

from src.feature_extraction import extract_features, reshape_for_cnn
from src.preprocess import load_label_encoder, load_trained_model


def predict_emotion(file_path: str | Path) -> tuple[str, dict[str, float]]:
    """Predict the emotion expressed in a single audio file.

    Args:
        file_path (str | Path): Path to any ``.wav`` file.

    Returns:
        tuple[str, dict[str, float]]:
            - ``predicted_emotion``: Most likely emotion label string.
            - ``confidence_scores``: Mapping of every emotion label to its
              softmax probability (0.0 – 1.0).

    Raises:
        FileNotFoundError: Audio file, trained model, or label encoder not
            found.
        RuntimeError: Feature extraction or inference fails.
    """
    # ----- Feature extraction (same pipeline as training) ----- #
    features: np.ndarray = extract_features(Path(file_path))

    # Single-sample batch: (N_MFCC, MAX_PAD_LENGTH) → (1, N_MFCC, MAX_PAD_LENGTH, 1)
    X: np.ndarray = reshape_for_cnn(np.expand_dims(features, axis=0))

    # ----- Load artifacts ----- #
    model   = load_trained_model()
    encoder = load_label_encoder()

    # ----- Inference ----- #
    try:
        predictions: np.ndarray = model.predict(X, verbose=0)[0]
    except Exception as error:
        raise RuntimeError(
            f"Prediction failed for '{file_path}': {error}"
        ) from error

    predicted_index: int  = int(np.argmax(predictions))
    predicted_emotion: str = encoder.inverse_transform([predicted_index])[0]

    confidence_scores: dict[str, float] = {
        emotion: float(score)
        for emotion, score in zip(encoder.classes_, predictions)
    }

    return predicted_emotion, confidence_scores


def print_prediction_report(
    file_path: str | Path,
    predicted_emotion: str,
    confidence_scores: dict[str, float],
) -> None:
    """Print a human-readable prediction summary to stdout.

    Displays a simple ASCII bar chart sorted by confidence so the most
    likely emotions are immediately visible.

    Args:
        file_path (str | Path): The file that was classified.
        predicted_emotion (str): Model's top prediction.
        confidence_scores (dict[str, float]): Per-class probabilities.
    """
    print(f"\nFile             : {file_path}")
    print(f"Predicted Emotion: {predicted_emotion.upper()}\n")
    print("Confidence Scores:")

    sorted_scores = sorted(
        confidence_scores.items(), key=lambda item: item[1], reverse=True
    )
    for emotion, score in sorted_scores:
        bar = "*" * int(score * 40)
        print(f"  {emotion:<12} {score:>6.2%}  {bar}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <path_to_wav_file>")
        sys.exit(1)

    input_path = sys.argv[1]

    try:
        emotion, scores = predict_emotion(input_path)
        print_prediction_report(input_path, emotion, scores)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}")
        sys.exit(1)
