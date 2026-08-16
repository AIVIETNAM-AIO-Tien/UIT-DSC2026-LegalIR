import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute and cache tokenized corpus & queries."
    )

    # Input paths
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to train.json",
    )
    parser.add_argument(
        "--contexts",
        type=str,
        required=True,
        help="Path to selected-contexts directory",
    )

    # Chunking configuration
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Chunk size in tokenizer units.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=256,
        help="Chunk overlap in tokenizer units.",
    )

    # Output path
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cache/",
        help="Directory where preprocessed pickles will be saved.",
    )

    return parser.parse_args()


