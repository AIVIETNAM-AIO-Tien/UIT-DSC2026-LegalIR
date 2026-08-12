import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.dense import DenseRetriever
from src.data.loader import load_queries
from src.submission.builder import build_submission


MODEL_NAME = "AITeamVN/Vietnamese_Embedding_v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate official Dense Retrieval submission."
    )

    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to public-official.json",
    )

    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Path to dense_chunks.json",
    )

    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to dense_chunk_embeddings.npy",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dense/submission.json",
        help="Output submission.json path.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="SentenceTransformer model name.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents submitted per query.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ==========================================================
    # Validate top-k
    # ==========================================================

    if args.top_k <= 0:
        raise ValueError(
            "top-k must be greater than 0."
        )

    if args.top_k > 5:
        raise ValueError(
            "Official submission allows at most 5 documents."
        )

    # ==========================================================
    # Load test queries
    # ==========================================================

    print("=" * 60)
    print("Loading test queries")
    print("=" * 60)

    queries = load_queries(args.test)

    print(
        f"Loaded {len(queries)} queries."
    )

    # ==========================================================
    # Load Dense model
    # ==========================================================

    print("=" * 60)
    print("Loading Dense model")
    print("=" * 60)

    model = SentenceTransformer(
        args.model
    )

    print(
        f"Model: {args.model}"
    )

    print(
        "Max sequence length:",
        model.max_seq_length
    )

    # ==========================================================
    # Load Dense Retriever
    # ==========================================================

    print("=" * 60)
    print("Loading Dense Retriever")
    print("=" * 60)

    retriever = DenseRetriever(
        model_name=MODEL_NAME
    )

    print(
            "Model:",
            retriever.model_name,
        )

    print(
            "Max sequence length:",
            retriever.model.max_seq_length,
        )


    # ==========================================================
    # Load precomputed Dense index
    # ==========================================================

    print("=" * 60)
    print("Loading Dense index")
    print("=" * 60)

    retriever.load_index(
        chunks_path=args.chunks,
        embeddings_path=args.embeddings,
    )

    # ==========================================================
    # Retrieval
    # ==========================================================

    retrieval_results = {}

    print("=" * 60)
    print("Running Dense Retrieval")
    print("=" * 60)

    for index, query in enumerate(
        queries,
        start=1,
    ):

        print(
            f"[{index}/{len(queries)}] "
            f"Query ID: {query.query_id}"
        )

        results = retriever.retrieve(
            query=query,
        )

        retrieval_results[
            query.query_id
        ] = [
            result.document_id
            for result in results
        ]

    # ==========================================================
    # Build submission
    # ==========================================================

    print("=" * 60)
    print("Building submission")
    print("=" * 60)

    submission = build_submission(
        retrieval_results=retrieval_results,
        output_path=args.output,
        top_k=args.top_k,
    )

    # ==========================================================
    # Summary
    # ==========================================================

    print(
        f"Generated submission for "
        f"{len(submission)} queries."
    )

    print(
        f"Output: {args.output}"
    )

    print("=" * 60)
    print("Submission preview")
    print("=" * 60)

    preview_count = 3

    for index, (
        question_id,
        value,
    ) in enumerate(
        submission.items()
    ):

        if index >= preview_count:
            break

        print(
            question_id,
            "->",
            value["answer"],
        )


if __name__ == "__main__":
    main()