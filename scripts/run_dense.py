import argparse
import json
from pathlib import Path

import numpy as np

from src.data.loader import (
    load_documents,
    load_queries,
)

from src.retrieval.dense import DenseRetriever
from src.retrieval.aggregation import aggregate_max_score

from src.evaluation.evaluator import Evaluator


MODEL_NAME = "AITeamVN/Vietnamese_Embedding_v2"


DEFAULT_K_VALUES = [
    5,
    10,
    20,
    50,
    100,
]


# ==============================================================
# Argument Parser
# ==============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run Dense Retrieval benchmark "
            "on training data."
        )
    )

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------

    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to train.json",
    )

    parser.add_argument(
        "--contexts",
        type=str,
        default=None,
        help=(
            "Path to selected-contexts directory. "
            "Required when building Dense index."
        ),
    )

    # ----------------------------------------------------------
    # Optional precomputed index
    # ----------------------------------------------------------

    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help=(
            "Path to precomputed dense_chunks.json. "
            "If provided together with --embeddings, "
            "Dense index will be loaded."
        ),
    )

    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help=(
            "Path to precomputed "
            "dense_chunk_embeddings.npy."
        ),
    )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dense/result.json",
        help="Path to benchmark result JSON.",
    )

    parser.add_argument(
        "--output-chunks",
        type=str,
        default="outputs/dense/dense_chunks.json",
        help=(
            "Path to save Dense chunks when "
            "building the index."
        ),
    )

    parser.add_argument(
        "--output-embeddings",
        type=str,
        default="outputs/dense/dense_chunk_embeddings.npy",
        help=(
            "Path to save Dense embeddings when "
            "building the index."
        ),
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="SentenceTransformer model name.",
    )

    # ----------------------------------------------------------
    # Chunking
    # ----------------------------------------------------------

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Dense chunk size in model tokens.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=256,
        help="Dense chunk overlap in model tokens.",
    )

    # ----------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )

    # ----------------------------------------------------------
    # Retrieval / Evaluation
    # ----------------------------------------------------------

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help=(
            "Maximum number of documents retrieved "
            "per query."
        ),
    )
    
    parser.add_argument(
        "--eval-k",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help="Recall@K values.",
    )

    return parser.parse_args()


# ==============================================================
# Save JSON
# ==============================================================

def save_json(
    data,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4,
        )


# ==============================================================
# Main
# ==============================================================

def main():

    args = parse_args()

    # ==========================================================
    # Validation
    # ==========================================================

    if args.top_k <= 0:
        raise ValueError(
            "--top-k must be greater than 0."
        )

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be greater than 0."
        )

    if args.overlap < 0:
        raise ValueError(
            "--overlap must be >= 0."
        )

    if args.overlap >= args.chunk_size:
        raise ValueError(
            "--overlap must be smaller than "
            "--chunk-size."
        )

    # ----------------------------------------------------------
    # --chunks and --embeddings must appear together
    # ----------------------------------------------------------

    if (
        (args.chunks is None)
        != (args.embeddings is None)
    ):
        raise ValueError(
            "--chunks and --embeddings must be "
            "provided together."
        )

    use_precomputed_index = (
        args.chunks is not None
        and args.embeddings is not None
    )

    # ==========================================================
    # Load train queries
    # ==========================================================

    print("=" * 60)
    print("Loading train queries")
    print("=" * 60)

    queries = load_queries(
        args.train
    )

    print(
        f"Loaded {len(queries)} queries."
    )

    # ==========================================================
    # Load Dense Retriever
    # ==========================================================

    print("=" * 60)
    print("Loading Dense Retriever")
    print("=" * 60)

    retriever = DenseRetriever(
        model_name=args.model
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
    # Load / Build Dense index
    # ==========================================================

    if use_precomputed_index:

        # ======================================================
        # LOAD
        # ======================================================

        print("=" * 60)
        print("Loading precomputed Dense index")
        print("=" * 60)

        print(
            "Chunks:",
            args.chunks,
        )

        print(
            "Embeddings:",
            args.embeddings,
        )

        retriever.load_index(
            chunks_path=args.chunks,
            embeddings_path=args.embeddings,
        )

        print(
            "Loaded chunks:",
            len(retriever.chunks),
        )

        print(
            "Embedding shape:",
            retriever.chunk_embeddings.shape,
        )

    else:

        # ======================================================
        # BUILD
        # ======================================================

        if args.contexts is None:
            raise ValueError(
                "--contexts is required when "
                "building the Dense index."
            )

        print("=" * 60)
        print("Building Dense index")
        print("=" * 60)

        # ------------------------------------------------------
        # Load legal documents
        # ------------------------------------------------------

        print(
            "Loading legal corpus..."
        )

        documents = load_documents(
            args.contexts
        )

        print(
            f"Loaded {len(documents)} documents."
        )

        # ------------------------------------------------------
        # Build chunks and embeddings
        # ------------------------------------------------------

        print("=" * 60)
        print("Building Dense chunks and embeddings")
        print("=" * 60)

        retriever.build(
            documents=documents,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            show_progress_bar=True,
        )

        print(
            "Total Dense chunks:",
            len(retriever.chunks),
        )

        print(
            "Embedding shape:",
            retriever.chunk_embeddings.shape,
        )

        # ------------------------------------------------------
        # Save Index
        # ------------------------------------------------------

        print("=" * 60)
        print("Saving Index")
        print("=" * 60)

        retriever.save_index(
            args.output_chunks,
            args.output_embeddings
        )

        print(
            "Saved:",
            args.output_chunks,
        )

        print(
            "Saved:",
            args.output_embeddings,
        )

    # ==========================================================
    # Sanity Check
    # ==========================================================

    if (
        retriever.chunk_embeddings is None
    ):
        raise RuntimeError(
            "Dense chunk embeddings are not loaded."
        )

    if (
        len(retriever.chunks)
        != len(retriever.chunk_embeddings)
    ):
        raise ValueError(
            "Number of chunks does not match "
            "number of embeddings."
        )

    # ==========================================================
    # Dense Retrieval
    # ==========================================================

    print("=" * 60)
    print("Running Dense Retrieval")
    print("=" * 60)

    retrieval_results = {}

    for index, query in enumerate(
        queries,
        start=1,
    ):

        results = retriever.retrieve(
            query=query,
        )

        results = aggregate_max_score(
            results=results,
            top_k=args.top_k
        )

        retrieval_results[
            query.query_id
        ] = [
            result.document_id
            for result in results
        ]

        if (
            index % 50 == 0
            or index == len(queries)
        ):

            print(
                f"Processed "
                f"{index}/{len(queries)} queries"
            )

    # ==========================================================
    # Build Ground Truth
    # ==========================================================

    print("=" * 60)
    print("Preparing Ground Truth")
    print("=" * 60)

    ground_truth = {
        query.query_id: query.answers
        for query in queries
    }

    # ==========================================================
    # Evaluation
    # ==========================================================

    print("=" * 60)
    print("Evaluating Dense Retrieval")
    print("=" * 60)

    evaluator = Evaluator(
        k_values=args.eval_k
    )

    evaluation = evaluator.evaluate(
        ground_truth=ground_truth,
        retrieved=retrieval_results,
    )

    # ==========================================================
    # Build final result
    # ==========================================================

    result = {
        "method": "dense",
        "model": args.model,

        "config": {
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "top_k_document": args.top_k,
            "batch_size": args.batch_size,
        },

        "index": {
            "mode": (
                "load"
                if use_precomputed_index
                else "build"
            ),
            "chunks": (
                args.chunks
                if use_precomputed_index
                else args.output_chunks
            ),
            "embeddings": (
                args.embeddings
                if use_precomputed_index
                else args.output_embeddings
            ),
            "num_chunks": len(
                retriever.chunks
            ),
            "embedding_shape": list(
                retriever.chunk_embeddings.shape
            ),
        },

        "num_queries": len(queries),

        "evaluation": evaluation,
    }

    # ==========================================================
    # Save result
    # ==========================================================

    print("=" * 60)
    print("Saving benchmark result")
    print("=" * 60)

    save_json(
        result,
        args.output,
    )

    print(
        "Saved:",
        args.output,
    )

    # ==========================================================
    # Print Recall Summary
    # ==========================================================

    print()
    print("=" * 60)
    print("Dense Retrieval Results")
    print("=" * 60)

    aggregate = evaluation["aggregate"]

    for k in args.eval_k:

        key = f"recall@{k}"

        print(
            f"{key:<12}: "
            f"{aggregate[key]:.4f}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()