"""
model.py
========
Defines the CNN architecture for Speech Emotion Recognition.

Responsibilities:
    - Build and compile the Keras model.
    - Nothing else. No data loading, no training loop, no file I/O.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import LEARNING_RATE, MAX_PAD_LENGTH, N_MFCC, NUM_CLASSES


def build_model(
    input_shape: tuple[int, int, int] = (N_MFCC, MAX_PAD_LENGTH, 1),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """Build and compile a CNN for multi-class emotion classification.

    Architecture (3 conv blocks → dense head):
        Conv2D(32) → BN → MaxPool → Dropout
        Conv2D(64) → BN → MaxPool → Dropout
        Conv2D(128)→ BN → MaxPool → Dropout
        Flatten → Dense(128) → Dropout → Dense(num_classes, softmax)

    Rationale:
        - Filter counts double each block (32 → 64 → 128): early layers
          capture low-level time-frequency patterns; deeper layers combine
          them into abstract emotion-related representations.
        - BatchNormalization stabilises and speeds up training.
        - Dropout(0.25 / 0.5) reduces overfitting on the small RAVDESS set.
        - ``sparse_categorical_crossentropy`` works directly with integer
          labels (no one-hot conversion needed).

    Args:
        input_shape (tuple[int, int, int]): ``(N_MFCC, MAX_PAD_LENGTH, 1)``.
            The trailing ``1`` is the single channel (like greyscale image).
        num_classes (int): Number of emotion categories to predict.
        learning_rate (float): Adam optimiser learning rate.

    Returns:
        tf.keras.Model: Compiled model ready for ``model.fit()``.

    Raises:
        ValueError: If ``num_classes < 2`` or any ``input_shape``
            dimension is non-positive.
    """
    if num_classes < 2:
        raise ValueError(
            f"num_classes must be >= 2, got {num_classes}."
        )
    if any(dim <= 0 for dim in input_shape):
        raise ValueError(
            f"All input_shape dimensions must be positive, got {input_shape}."
        )

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            # --- Block 1 ---
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # --- Block 2 ---
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # --- Block 3 ---
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # --- Classification head ---
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="speech_emotion_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
