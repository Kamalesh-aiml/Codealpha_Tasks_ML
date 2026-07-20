"""
main.py
=======
Single command-line entry point for the Speech Emotion Recognition project.

Usage:
    python main.py --stage preprocess
    python main.py --stage train
    python main.py --stage evaluate
    python main.py --stage predict --file path/to/audio.wav
    python main.py --stage all
"""

import argparse
import sys


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments with ``stage`` and optional
            ``file`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition - pipeline runner.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["preprocess", "train", "evaluate", "predict", "all"],
        help=(
            "Pipeline stage to run:\n"
            "  preprocess  - extract MFCC features and save to disk\n"
            "  train       - train the CNN model\n"
            "  evaluate    - evaluate on the held-out test set\n"
            "  predict     - classify a single audio file (requires --file)\n"
            "  all         - run preprocess -> train -> evaluate in sequence"
        ),
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .wav file (required when --stage predict).",
    )
    return parser.parse_args()


def main() -> None:
    """Route the requested pipeline stage and handle top-level errors.

    Raises:
        SystemExit: On invalid arguments or unrecoverable pipeline errors.
    """
    args = parse_arguments()

    try:
        if args.stage == "preprocess":
            from src.preprocess import preprocess_dataset
            preprocess_dataset()

        elif args.stage == "train":
            from src.train import run_training_pipeline
            run_training_pipeline()

        elif args.stage == "evaluate":
            from src.evaluate import run_evaluation_pipeline
            run_evaluation_pipeline()

        elif args.stage == "predict":
            if not args.file:
                print("Error: --file is required when --stage predict.")
                sys.exit(1)
            from src.predict import predict_emotion, print_prediction_report
            emotion, scores = predict_emotion(args.file)
            print_prediction_report(args.file, emotion, scores)

        elif args.stage == "all":
            from src.preprocess import preprocess_dataset
            from src.train import run_training_pipeline
            from src.evaluate import run_evaluation_pipeline

            separator = "=" * 60
            print(f"{separator}\nPREPROCESSING\n{separator}")
            preprocess_dataset()

            print(f"\n{separator}\nTRAINING\n{separator}")
            run_training_pipeline()

            print(f"\n{separator}\nEVALUATION\n{separator}")
            run_evaluation_pipeline()

    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"\nPipeline error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
