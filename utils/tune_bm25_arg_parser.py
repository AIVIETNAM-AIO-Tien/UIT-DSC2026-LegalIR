import argparse


DEFAULT_K_VALUES = [5, 10, 20, 50, 100]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast BM25 Indexing, Retrieval, and Evaluation using cached preprocessed data."
    )

    # ----------------------------------------------------------
    # Cached chunks and queries input paths
    # ----------------------------------------------------------
    parser.add_argument(
        "--chunks-file",
        type=str,
        required=True,
        help="Path to preprocessed chunks (.pkl)",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        required=True,
        help="Path to preprocessed queries (.pkl)",
    )

    # ----------------------------------------------------------
    # BM25 Variant & Dynamic Hyperparameters
    # ----------------------------------------------------------
    parser.add_argument(
        "--variant",
        type=str,
        default="okapi",
        choices=["okapi", "bm25l", "bm25plus"],
        help="BM25 variant.",
    )

    parser.add_argument(
        "--k1",
        type=float,
        nargs="+",
        default=[1.5],
        help="BM25 k1 parameter(s). Provide multiple values for grid search (e.g. --k1 1.2 1.5 2.0).",
    )

    parser.add_argument(
        "--b",
        type=float,
        nargs="+",
        default=[0.75],
        help="BM25 b parameter(s). Provide multiple values for grid search (e.g. --b 0.3 0.5 0.75).",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.25,
        help="BM25Okapi epsilon parameter.",
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=0.5,
        help="BM25L/BM25Plus delta parameter.",
    )

    # ----------------------------------------------------------
    # Retrieval / Evaluation
    # ----------------------------------------------------------
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=100,
        help="Number of documents retrieved per query.",
    )
    parser.add_argument(
        "--eval-k",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help="Recall@K evaluation values.",
    )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/bm25/result_fast.json",
        help="Path to save evaluation summary JSON.",
    )
    parser.add_argument(
        "--save-retrieval",
        type=str,
        default=None,
        help="Optional path to save raw document retrieval predictions.",
    )

    return parser.parse_args()


