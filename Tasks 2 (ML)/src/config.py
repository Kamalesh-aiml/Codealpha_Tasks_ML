"""
config.py
=========
Central configuration file for the Speech Emotion Recognition project.

This module is the single source of truth for all file paths,
hyperparameters, and constants. No other module should hardcode these
values — they must always be imported from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT ROOT & DIRECTORY PATHS
# ---------------------------------------------------------------------------
# Path(__file__) → src/config.py  |  .parent → src/  |  .parent → project/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODEL_DIR: Path = PROJECT_ROOT / "models" / "saved_model"

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# PROCESSED DATA FILE PATHS
# ---------------------------------------------------------------------------
# Pre-split arrays saved by preprocess.py so extraction runs only once.
X_TRAIN_PATH: Path = PROCESSED_DATA_DIR / "X_train.npy"
X_TEST_PATH: Path  = PROCESSED_DATA_DIR / "X_test.npy"
Y_TRAIN_PATH: Path = PROCESSED_DATA_DIR / "y_train.npy"
Y_TEST_PATH: Path  = PROCESSED_DATA_DIR / "y_test.npy"

# ---------------------------------------------------------------------------
# MODEL & ENCODER ARTIFACT PATHS
# ---------------------------------------------------------------------------
MODEL_PATH: Path         = MODEL_DIR / "emotion_cnn_model.keras"
LABEL_ENCODER_PATH: Path = MODEL_DIR / "label_encoder.pkl"

# ---------------------------------------------------------------------------
# OUTPUT PLOT PATHS
# ---------------------------------------------------------------------------
TRAINING_HISTORY_PATH: Path  = PLOTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH: Path  = PLOTS_DIR / "confusion_matrix.png"

# ---------------------------------------------------------------------------
# AUDIO CONFIGURATION
# ---------------------------------------------------------------------------
SAMPLE_RATE: int    = 22050  # Hz — librosa resamples every file to this rate
N_MFCC: int         = 40     # Number of MFCC coefficients to extract
MAX_PAD_LENGTH: int = 174    # Fixed time-frame length (pad / truncate to this)

# ---------------------------------------------------------------------------
# TRAINING CONFIGURATION
# ---------------------------------------------------------------------------
BATCH_SIZE: int              = 32
EPOCHS: int                  = 30
LEARNING_RATE: float         = 0.001
EARLY_STOPPING_PATIENCE: int = 8   # Epochs without val_loss improvement before stop

# ---------------------------------------------------------------------------
# DATA SPLIT CONFIGURATION
# ---------------------------------------------------------------------------
TEST_SIZE: float  = 0.2   # 20 % of full dataset held out as the test set
VAL_SIZE: float   = 0.1   # 10 % of training data used as validation during training
RANDOM_SEED: int  = 42    # Fixed seed for reproducible splits

# ---------------------------------------------------------------------------
# RAVDESS EMOTION MAPPING
# ---------------------------------------------------------------------------
# Filename format: 03-01-06-01-02-01-12.wav
# Field index 2 (0-based) encodes the emotion class.
EMOTION_MAP: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

NUM_CLASSES: int = len(EMOTION_MAP)  # Derived automatically — stays in sync


# ---------------------------------------------------------------------------
# DIRECTORY INITIALISATION
# ---------------------------------------------------------------------------
def create_required_directories() -> None:
    """Create all required project directories if they do not already exist.

    Called automatically on import so every downstream script can safely
    read/write files without hitting a missing-directory error.

    Raises:
        OSError: If a directory cannot be created (e.g. permission denied).
    """
    required_dirs: list[Path] = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        PLOTS_DIR,
        NOTEBOOKS_DIR,
    ]
    for directory in required_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise OSError(
                f"Failed to create directory '{directory}': {error}"
            ) from error


create_required_directories()
