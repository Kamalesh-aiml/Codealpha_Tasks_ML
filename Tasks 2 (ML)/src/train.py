"""
train.py
========
Trains the CNN model on preprocessed MFCC features.

Pipeline:
    1. Load cached train/test arrays from disk (preprocess.py).
    2. Carve a validation split from the training data.
    3. Reshape arrays for Conv2D input (add channel dimension).
    4. Build and compile the model (model.py).
    5. Train with EarlyStopping and ModelCheckpoint callbacks.
    6. Save the best model and a training history plot.

Run:
    python -m src.train
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    MODEL_PATH,
    RANDOM_SEED,
    TRAINING_HISTORY_PATH,
    VAL_SIZE,
)
from src.feature_extraction import reshape_for_cnn
from src.model import build_model
from src.preprocess import load_processed_data


def get_validation_split(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carve a stratified validation set out of the training data.

    Keeping validation data separate from the test set prevents early
    stopping from being tuned on the final evaluation partition.

    For small datasets where stratification is impossible (fewer samples
    than classes), stratification is automatically disabled.

    Args:
        X_train (np.ndarray): Full training feature matrix.
        y_train (np.ndarray): Full training labels.

    Returns:
        tuple: ``X_tr, X_val, y_tr, y_val``
    """
    # Calculate minimum samples required per class for stratification
    n_classes = len(np.unique(y_train))
    n_val_samples = int(np.ceil(len(y_train) * VAL_SIZE))
    
    # If dataset is too small for stratified split, disable stratification
    stratify = y_train if n_val_samples >= n_classes else None
    
    return train_test_split(
        X_train, y_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )


def build_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create Keras training callbacks.

    - ``EarlyStopping``: halts training when ``val_loss`` stops improving
      and restores the best weights automatically.
    - ``ModelCheckpoint``: saves the best model observed during training
      to ``MODEL_PATH``, so a crash after the best epoch still preserves
      the best weights.

    Returns:
        list[tf.keras.callbacks.Callback]: Configured callbacks.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    return [early_stopping, checkpoint]


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Save accuracy and loss curves to ``TRAINING_HISTORY_PATH``.

    Args:
        history (tf.keras.callbacks.History): Object returned by
            ``model.fit()``.

    Raises:
        OSError: If the image file cannot be written to disk.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    try:
            fig.savefig(TRAINING_HISTORY_PATH)
    except OSError as error:
        raise OSError(f"Failed to save training history plot: {error}") from error
    finally:
        plt.close(fig)

        print(f"Training history plot saved -> '{TRAINING_HISTORY_PATH}'")


def run_training_pipeline() -> tf.keras.Model:
    """Execute the full training pipeline.

    Returns:
        tf.keras.Model: Trained model with best weights restored.

    Raises:
        FileNotFoundError: If processed data has not been generated yet.
        OSError: If the model or plot cannot be saved.
    """
    # ------------------------------------------------------------------ #
    # Step 1 — Load data
    # ------------------------------------------------------------------ #
    print("Step 1/5  Loading processed data...")
    X_train_full, X_test, y_train_full, y_test = load_processed_data()
    print(f"  Full train : {X_train_full.shape}  |  Test : {X_test.shape}")

    # ------------------------------------------------------------------ #
    # Step 2 — Carve out a validation split (not the test set)
    # ------------------------------------------------------------------ #
    print("\nStep 2/5  Creating validation split...")
    X_tr, X_val, y_tr, y_val = get_validation_split(X_train_full, y_train_full)
    print(f"  Train : {X_tr.shape}  |  Val : {X_val.shape}")

    # ------------------------------------------------------------------ #
    # Step 3 — Add channel dimension for Conv2D  →  (N, H, W, 1)
    # ------------------------------------------------------------------ #
    print("\nStep 3/5  Reshaping for CNN...")
    X_tr  = reshape_for_cnn(X_tr)
    X_val = reshape_for_cnn(X_val)
    X_test_reshaped = reshape_for_cnn(X_test)
    print(f"  Train shape : {X_tr.shape}")

    # ------------------------------------------------------------------ #
    # Step 4 — Build model
    # ------------------------------------------------------------------ #
    print("\nStep 4/5  Building model...")
    model = build_model()
    model.summary()

    # ------------------------------------------------------------------ #
    # Step 5 — Train
    # ------------------------------------------------------------------ #
    print("\nStep 5/5  Training...")
    callbacks = build_callbacks()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(history)

    loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"\nTest accuracy : {accuracy:.4f}  |  Test loss : {loss:.4f}")
    print(f"Model saved   -> '{MODEL_PATH}'")

    return model


if __name__ == "__main__":
    run_training_pipeline()
