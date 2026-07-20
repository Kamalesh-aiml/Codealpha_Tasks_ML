"""
feature_extraction.py
======================
Loads audio files and converts them into fixed-size MFCC feature arrays
suitable for CNN input.

This module is intentionally "pure": no directory traversal, no labels,
no file saving. It transforms: file path → NumPy array.

Also owns `reshape_for_cnn()` because adding the channel dimension is a
feature-shaping step, not a training step, and is needed by both
preprocess.py and predict.py.
"""

from pathlib import Path

import librosa
import numpy as np

from src.config import MAX_PAD_LENGTH, N_MFCC, SAMPLE_RATE


def load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    """Load a `.wav` file and resample it to the project sample rate.

    Args:
        file_path (Path): Path to the audio file.

    Returns:
        tuple[np.ndarray, int]: (signal, sample_rate) where sample_rate
            always equals ``SAMPLE_RATE`` from config.

    Raises:
        FileNotFoundError: If the file does not exist on disk.
        RuntimeError: If librosa cannot decode the file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: '{file_path}'")

    try:
        signal, sample_rate = librosa.load(str(file_path), sr=SAMPLE_RATE)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load audio file '{file_path}': {error}"
        ) from error

    return signal, sample_rate


def pad_or_truncate(mfcc: np.ndarray) -> np.ndarray:
    """Force the time axis of an MFCC array to exactly ``MAX_PAD_LENGTH``.

    CNNs require every sample to have an identical shape. Audio clips
    vary in length, so we zero-pad short clips and truncate long ones.

    Args:
        mfcc (np.ndarray): 2-D array of shape ``(N_MFCC, time_frames)``.

    Returns:
        np.ndarray: Array of shape ``(N_MFCC, MAX_PAD_LENGTH)``.

    Raises:
        ValueError: If ``mfcc`` is not 2-D.
    """
    if mfcc.ndim != 2:
        raise ValueError(
            f"Expected a 2-D MFCC array (n_mfcc, time_frames), "
            f"got {mfcc.ndim}-D array."
        )

    current_length: int = mfcc.shape[1]

    if current_length < MAX_PAD_LENGTH:
        pad_width: int = MAX_PAD_LENGTH - current_length
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    elif current_length > MAX_PAD_LENGTH:
        mfcc = mfcc[:, :MAX_PAD_LENGTH]

    return mfcc


def extract_mfcc(file_path: Path) -> np.ndarray:
    """Extract a fixed-size MFCC feature matrix from one audio file.

    Args:
        file_path (Path): Path to the ``.wav`` file.

    Returns:
        np.ndarray: Shape ``(N_MFCC, MAX_PAD_LENGTH)``.

    Raises:
        FileNotFoundError: Propagated from ``load_audio()``.
        RuntimeError: If MFCC computation fails.
    """
    signal, sample_rate = load_audio(file_path)

    try:
        mfcc: np.ndarray = librosa.feature.mfcc(
            y=signal, sr=sample_rate, n_mfcc=N_MFCC
        )
    except Exception as error:
        raise RuntimeError(
            f"MFCC extraction failed for '{file_path}': {error}"
        ) from error

    return pad_or_truncate(mfcc)


def extract_features(file_path: Path) -> np.ndarray:
    """Public wrapper — the function all other modules should call.

    Keeping this as a stable public API means the internal strategy
    (e.g. switching from MFCC to mel-spectrograms) can change without
    touching every caller.

    Args:
        file_path (Path): Path to the ``.wav`` file.

    Returns:
        np.ndarray: Shape ``(N_MFCC, MAX_PAD_LENGTH)``.
    """
    return extract_mfcc(file_path)


def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """Add a trailing channel dimension so Conv2D layers accept the input.

    ``Conv2D`` expects shape ``(samples, height, width, channels)``.
    Our MFCC arrays are ``(samples, N_MFCC, MAX_PAD_LENGTH)``, so we
    append a channel dimension of 1 (grayscale image analogy).

    This function lives here — not in train.py — because it is a
    feature-shaping step required by both preprocessing and inference.

    Args:
        X (np.ndarray): Array of shape ``(samples, N_MFCC, MAX_PAD_LENGTH)``
            or ``(N_MFCC, MAX_PAD_LENGTH)`` for a single sample.

    Returns:
        np.ndarray: Array with one extra trailing dimension.

    Raises:
        ValueError: If ``X`` has fewer than 2 dimensions.
    """
    if X.ndim < 2:
        raise ValueError(
            f"Expected at least a 2-D array, got shape {X.shape}."
        )
    return np.expand_dims(X, axis=-1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.feature_extraction <path_to_wav>")
        sys.exit(1)

    features = extract_features(Path(sys.argv[1]))
    print(f"MFCC shape:         {features.shape}")
    print(f"CNN-ready shape:    {reshape_for_cnn(features).shape}")
