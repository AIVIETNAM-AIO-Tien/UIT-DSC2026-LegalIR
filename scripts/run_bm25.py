import argparse
import json
from pathlib import Path

from src.retrieval.rank_bm25 import BM25Okapi, BM25L, BM25Plus

from src.data.loader import load_documents, load_queries
from src.preprocessing.normalize import normalize_text
from src.preprocessing.tokenize import tokenize_vietnamese, decode_vietnamese
from src.chunking.fixed_size import FixedSizeChunker
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.aggregation import aggregate_max_score
from src.evaluation.evaluator import Evaluator


# ==============================================================
# BM25 variants
# ==============================================================

BM25_VARIANTS = {
    "okapi": BM25Okapi,
    "bm25l": BM25L,
    "bm25plus": BM25Plus,
}

DEFAULT_K_VALUES = [
    5,
    10,
    20,
    50,
    100,
]

TOP_K_CHUNKS = 1000


# ==============================================================
# Arguments
# ==============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run BM25 baseline on LegalIR train set."
    )

    # ----------------------------------------------------------
    # Data
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
        required=True,
        help="Path to selected-contexts directory",
    )

    # ----------------------------------------------------------
    # BM25
    # ----------------------------------------------------------

    parser.add_argument(
        "--variant",
        type=str,
        default="okapi",
        choices=[
            "okapi",
            "bm25l",
            "bm25plus",
        ],
        help="BM25 variant.",
    )

    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter.",
    )

    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter.",
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
    # Chunking
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Retrieval / evaluation
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
        help="Recall@K values.",
    )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/bm25/result.json",
        help="Path to result.json.",
    )

    parser.add_argument(
        "--save-retrieval",
        type=str,
        default=None,
        help="Optional path to save ranked document results.",
    )

    return parser.parse_args()


# ==============================================================
# BM25 configuration
# ==============================================================

def build_bm25_kwargs(args):

    if args.variant == "okapi":

        return {
            "k1": args.k1,
            "b": args.b,
            "epsilon": args.epsilon,
        }

    if args.variant in {
        "bm25l",
        "bm25plus",
    }:

        return {
            "k1": args.k1,
            "b": args.b,
            "delta": args.delta,
        }

    raise ValueError(
        f"Unknown BM25 variant: {args.variant}"
    )


# ==============================================================
# Main
# ==============================================================

def main():

    args = parse_args()

    # ==========================================================
    # Validate
    # ==========================================================

    if args.retrieval_k <= 0:
        raise ValueError(
            "--retrieval-k must be > 0"
        )

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be > 0"
        )

    if args.overlap < 0:
        raise ValueError(
            "--overlap must be >= 0"
        )

    if args.overlap >= args.chunk_size:
        raise ValueError(
            "--overlap must be smaller than --chunk-size"
        )

    # ==========================================================
    # Configuration
    # ==========================================================

    bm25_class = BM25_VARIANTS[args.variant]

    bm25_kwargs = build_bm25_kwargs(args)

    print("=" * 60)
    print("BM25 CONFIGURATION")
    print("=" * 60)

    print(
        f"Variant       : {args.variant}"
    )

    print(
        f"Parameters     : {bm25_kwargs}"
    )

    print(
        f"Chunk size     : {args.chunk_size}"
    )

    print(
        f"Chunk overlap  : {args.overlap}"
    )

    print(
        f"Retrieval K    : {args.retrieval_k}"
    )

    print(
        f"Evaluation K   : {args.eval_k}"
    )

    # ==========================================================
    # Load documents
    # ==========================================================

    print("=" * 60)
    print("Loading legal documents")
    print("=" * 60)

    documents = load_documents(
        args.contexts
    )

    print(
        f"Loaded {len(documents)} documents."
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
    # Normalization
    # ==========================================================

    print("=" * 60)
    print("Normalizing documents")
    print("=" * 60)

    for document in documents:
        document.text = normalize_text(document.text)

    # ==========================================================
    # Fixed-size chunking
    # ==========================================================

    print("=" * 60)
    print("Building BM25 chunks")
    print("=" * 60)

    chunker = FixedSizeChunker(
        tokenize_fn=tokenize_vietnamese,
        decode_fn=decode_vietnamese,
        tokenizer_name="pyvi",
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    # ----------------------------------------------------------
    # IMPORTANT:
    #
    # The exact call below depends on the FixedSizeChunker
    # interface currently implemented in the project.
    #
    # If the chunker already accepts Document objects directly,
    # use:
    #
    #     chunks = chunker.chunk(documents)
    #
    # ----------------------------------------------------------

    chunks = []

    for document in documents:
        document_chunks = chunker.chunk(document)
        chunks.extend(document_chunks)

    print(
        f"Total chunks: {len(chunks)}"
    )

    # ==========================================================
    # Build BM25 Retriever
    # ==========================================================

    print("=" * 60)
    print("Building BM25 index")
    print("=" * 60)

    retriever = BM25Retriever(
        tokenize_fn=tokenize_vietnamese,
        bm25_class=bm25_class,
        bm25_kwargs=bm25_kwargs,
    )

    retriever.fit(
        chunks
    )

    print(
        "BM25 index built successfully."
    )

    # ==========================================================
    # Retrieval
    # ==========================================================

    print("=" * 60)
    print("Running BM25 Retrieval")
    print("=" * 60)

    retrieved_documents = {}

    for index, query in enumerate(
        queries,
        start=1,
    ):

        print(
            f"[{index}/{len(queries)}] "
            f"Query ID: {query.query_id}"
        )

        # ------------------------------------------------------
        # Chunk-level retrieval
        # ------------------------------------------------------

        chunk_results = retriever.retrieve(
            query=query,
        )

        # ------------------------------------------------------
        # Chunk → Document
        # ------------------------------------------------------

        document_results = aggregate_max_score(
            chunk_results,
            top_k=args.retrieval_k,
        )

        retrieved_documents[
            query.query_id
        ] = [
            result.document_id
            for result in document_results
        ]

    # ==========================================================
    # Evaluation
    # ==========================================================

    print("=" * 60)
    print("Evaluating Recall")
    print("=" * 60)

    ground_truth = {
        query.query_id: query.answers
        for query in queries
    }

    evaluator = Evaluator(
        k_values=args.eval_k
    )

    evaluation = evaluator.evaluate(
        ground_truth=ground_truth,
        retrieved=retrieved_documents,
    )

    # ==========================================================
    # Build result
    # ==========================================================

    result = {
        "configuration": {
            "bm25_variant": args.variant,
            "bm25_parameters": bm25_kwargs,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "retrieval_k_document": args.retrieval_k,
            "evaluation_k": args.eval_k,
        },

        "dataset": {
            "num_documents": len(documents),
            "num_queries": len(queries),
            "num_chunks": len(chunks),
        },

        "evaluation": evaluation,
    }

    # ==========================================================
    # Save result
    # ==========================================================

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("=" * 60)
    print("Finished")
    print("=" * 60)

    print(
        f"Result saved to: {output_path}"
    )

    # ==========================================================
    # Save retrieval results if requested
    # ==========================================================

    if args.save_retrieval:

        retrieval_path = Path(
            args.save_retrieval
        )

        retrieval_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with retrieval_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                retrieved_documents,
                f,
                ensure_ascii=False,
                indent=4,
            )

        print(
            f"Retrieval results saved to: "
            f"{retrieval_path}"
        )

    # ==========================================================
    # Print aggregate metrics
    # ==========================================================

    print("=" * 60)
    print("Recall Results")
    print("=" * 60)

    for metric, value in evaluation[
        "aggregate"
    ].items():

        print(
            f"{metric}: {value:.4f}"
        )


if __name__ == "__main__":
    main()