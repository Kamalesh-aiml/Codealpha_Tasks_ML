"""
data_loader.py
===============
Discovers RAVDESS audio files inside `data/raw/` and parses their
emotion labels from the standardized RAVDESS filename format.

This module performs no audio signal processing - it only handles
file discovery and filename parsing. Audio feature extraction lives
in `feature_extraction.py`.

RAVDESS filename format (7 dash-separated 2-digit fields), e.g.:
    03-01-06-01-02-01-12.wav
    │  │  │  │  │  │  └── Actor ID
    │  │  │  │  │  └───── Repetition
    │  │  │  │  └──────── Statement
    │  │  │  └─────────── Intensity
    │  │  └────────────── Emotion  <-- field we care about (index 2)
    │  └───────────────── Vocal channel
    └──────────────────── Modality
"""

from pathlib import Path

import pandas as pd

from src.config import EMOTION_MAP, RAW_DATA_DIR

# Expected number of dash-separated fields in a valid RAVDESS filename.
EXPECTED_FILENAME_PARTS: int = 7

# Index of the emotion code within the dash-separated filename fields.
EMOTION_CODE_INDEX: int = 2


def get_audio_files() -> list[Path]:
    """Find all `.wav` audio files inside the raw data directory.

    Searches `RAW_DATA_DIR` recursively, which correctly handles
    RAVDESS's "Actor_01/", "Actor_02/", ... subfolder structure.

    Returns:
        list[Path]: Sorted list of paths to every `.wav` file found
            under `RAW_DATA_DIR`. Sorting guarantees reproducible
            ordering across runs and operating systems.

    Raises:
        FileNotFoundError: If `RAW_DATA_DIR` does not exist.
        ValueError: If `RAW_DATA_DIR` exists but contains no `.wav`
            files.
    """
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: '{RAW_DATA_DIR}'. "
            "Please download the RAVDESS dataset and place it there."
        )

    audio_files: list[Path] = sorted(RAW_DATA_DIR.rglob("*.wav"))

    if not audio_files:
        raise ValueError(
            f"No '.wav' files found inside '{RAW_DATA_DIR}'. "
            "Please verify the dataset was extracted correctly."
        )

    return audio_files


def extract_emotion_label(file_path: Path) -> str:
    """Extract the human-readable emotion label from a RAVDESS filename.

    Args:
        file_path (Path): Path to a `.wav` file following the RAVDESS
            naming convention (e.g., "03-01-06-01-02-01-12.wav").

    Returns:
        str: The emotion label (e.g., "fearful"), as defined in
            `EMOTION_MAP`.

    Raises:
        ValueError: If the filename does not contain the expected
            number of dash-separated fields, or if the extracted
            emotion code is not a recognized key in `EMOTION_MAP`.
    """
    filename_stem: str = file_path.stem
    parts: list[str] = filename_stem.split("-")

    if len(parts) != EXPECTED_FILENAME_PARTS:
        raise ValueError(
            f"Invalid RAVDESS filename format: '{file_path.name}'. "
            f"Expected {EXPECTED_FILENAME_PARTS} dash-separated fields, "
            f"got {len(parts)}."
        )

    emotion_code: str = parts[EMOTION_CODE_INDEX]

    if emotion_code not in EMOTION_MAP:
        raise ValueError(
            f"Unrecognized emotion code '{emotion_code}' in file "
            f"'{file_path.name}'. Valid codes are: "
            f"{sorted(EMOTION_MAP.keys())}."
        )

    return EMOTION_MAP[emotion_code]


def load_dataset() -> pd.DataFrame:
    """Build a DataFrame mapping every discovered audio file to its
    parsed emotion label.

    This is the main public function other modules (e.g.,
    `preprocess.py`) should call to obtain the full dataset index.

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - "filepath" (str): Absolute path to the `.wav` file.
            - "emotion" (str): The corresponding emotion label.

    Raises:
        FileNotFoundError: Propagated from `get_audio_files()` if the
            raw data directory is missing.
        ValueError: Propagated from `get_audio_files()` or
            `extract_emotion_label()` if no files are found or a
            filename cannot be parsed.
    """
    audio_files: list[Path] = get_audio_files()

    records: list[dict[str, str]] = []
    for file_path in audio_files:
        emotion_label: str = extract_emotion_label(file_path)
        records.append(
            {
                "filepath": str(file_path),
                "emotion": emotion_label,
            }
        )

    dataset_df: pd.DataFrame = pd.DataFrame(records, columns=["filepath", "emotion"])
    return dataset_df


if __name__ == "__main__":
    # Manual sanity check when running this file directly:
    #   python -m src.data_loader
    df = load_dataset()
    print(f"Total audio files found: {len(df)}")
    print(df.head())
    print("\nEmotion distribution:")
    print(df["emotion"].value_counts())
