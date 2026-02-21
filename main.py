"""Entry-point wrapper for the segmentation pipelines.

Usage — DINOv2 pipeline
-----------------------
    python main.py train     [--model segformer_head] [--epochs 8] ...
    python main.py test      [--model segformer_head] [--lime] ...
    python main.py explain   [--model segformer_head] [--num_images 5] ...
    python main.py visualize [--input_dir ...]

Usage — U-MixFormer pipeline
----------------------------
    python main.py umix-train    [--epochs 50] ...
    python main.py umix-eval     [--split test] [--xai] [--compare] ...
    python main.py umix-xai      [--num_images 5] ...
    python main.py umix-compare   [--split test] ...
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
    elif command == "explain":
        from offroad_training_pipeline.explain_lime import main as explain_main
        explain_main()
    elif command == "visualize":
        from offroad_training_pipeline.visualize import main as vis_main
        vis_main()
    # ---- U-MixFormer pipeline commands ----
    elif command == "umix-train":
        from umixformer_pipeline.train import main as umix_train
        umix_train()
    elif command == "umix-eval":
        from umixformer_pipeline.evaluate import main as umix_eval
        umix_eval()
    elif command == "umix-xai":
        from umixformer_pipeline.explain_shap import main as umix_xai
        umix_xai()
    elif command == "umix-compare":
        from umixformer_pipeline.compare_preproc import main as umix_compare
        umix_compare()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, test, explain, visualize, "
              "umix-train, umix-eval, umix-xai, umix-compare")
        sys.exit(1)


if __name__ == "__main__":
    main()
