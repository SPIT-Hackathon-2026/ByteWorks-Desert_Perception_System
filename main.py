"""
Entry-point wrapper for the offroad segmentation pipeline.

Usage
-----
    python main.py train   [--model convnext_head] [--epochs 10] ...
    python main.py test    [--model convnext_head] [--model_path ...] ...
    python main.py visualize [--input_dir ...]
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    command = sys.argv.pop(1)  # remove sub-command so argparse in each module works

    if command == "train":
        from offroad_training_pipeline.train import main as train_main
        train_main()
    elif command == "test":
        from offroad_training_pipeline.test import main as test_main
        test_main()
    elif command == "visualize":
        from offroad_training_pipeline.visualize import main as vis_main
        vis_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, test, visualize")
        sys.exit(1)


if __name__ == "__main__":
    main()
