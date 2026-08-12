import argparse
import json
from pathlib import Path

from src.data.loader import load_queries
from src.retrieval.dense import DenseRetriever


def build_submission(
    queries,
    retriever: DenseRetriever,
    top_k: int = 5,
):
    """
    Build submission results.

    Dense baseline logic:

        Query
          ↓
        Query embedding
          ↓
        Dot product with ALL chunk embeddings
          ↓
        MAX score per document
          ↓
        Rank documents
          ↓
        Top-k documents
    """

    submission = {}

    total = len(queries)

    for i, query in enumerate(
        queries,
        start=1,
    ):
        print(
            f"[{i}/{total}] "
            f"Processing query {query.query_id}"
        )

        # ------------------------------------------------------
        # Query → scores for ALL chunks
        # ------------------------------------------------------

        scores = retriever.score_query(
            query
        )

        # ------------------------------------------------------
        # MAX score per document
        #
        # This follows the original Kaggle baseline.
        # ------------------------------------------------------

        document_scores = {}

        for chunk_index, score in enumerate(
            scores
        ):
            document_id = (
                retriever.chunks[
                    chunk_index
                ].document_id
            )

            score = float(score)

            if (
                document_id not in document_scores
                or score > document_scores[document_id]
            ):
                document_scores[document_id] = score

        # ------------------------------------------------------
        # Rank documents
        # ------------------------------------------------------

        ranked_documents = sorted(
            document_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # ------------------------------------------------------
        # Top-k document IDs
        # ------------------------------------------------------

        top_documents = [
            document_id
            for document_id, _ in ranked_documents[:top_k]
        ]

        # ------------------------------------------------------
        # Submission format
        # ------------------------------------------------------

        submission[query.query_id] = {
            "answer": top_documents
        }

    return submission


def save_submission(
    submission,
    output_path: str | Path,
):
    """
    Save submission.json as UTF-8 JSON.
    """

    output_path = Path(
        output_path
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
            submission,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print()
    print("=" * 60)
    print("Submission saved")
    print("=" * 60)
    print(
        f"Path: {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Dense Retrieval submission "
            "for UIT-DSC2026 LegalIR."
        )
    )

    # ----------------------------------------------------------
    # Test queries
    # ----------------------------------------------------------

    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help=(
            "Path to public-official.json"
        ),
    )

    # ----------------------------------------------------------
    # Dense index
    # ----------------------------------------------------------

    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help=(
            "Path to dense_chunks.json"
        ),
    )

    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help=(
            "Path to dense_chunk_embeddings.npy"
        ),
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default=(
            "AITeamVN/"
            "Vietnamese_Embedding_v2"
        ),
        help=(
            "SentenceTransformer model name."
        ),
    )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output submission.json path."
        ),
    )

    # ----------------------------------------------------------
    # Top-k
    # ----------------------------------------------------------

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of documents per query."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ==========================================================
    # Validate arguments
    # ==========================================================

    if args.top_k <= 0:
        raise ValueError(
            "--top-k must be > 0"
        )

    # ==========================================================
    # Loading test queries
    # ==========================================================

    print("=" * 60)
    print("Loading test queries")
    print("=" * 60)

    queries = load_queries(
        args.test
    )

    print(
        f"Loaded {len(queries)} queries."
    )

    # ==========================================================
    # Loading Dense Retriever
    # ==========================================================

    print("=" * 60)
    print("Loading Dense Retriever")
    print("=" * 60)

    retriever = DenseRetriever(
        model_name=args.model,
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
    # Build submission
    # ==========================================================

    print("=" * 60)
    print("Generating submission")
    print("=" * 60)

    submission = build_submission(
        queries=queries,
        retriever=retriever,
        top_k=args.top_k,
    )

    # ==========================================================
    # Save
    # ==========================================================

    save_submission(
        submission=submission,
        output_path=args.output,
    )

    # ==========================================================
    # Basic validation
    # ==========================================================

    print()
    print("=" * 60)
    print("Validation")
    print("=" * 60)

    print(
        "Number of queries:",
        len(submission),
    )

    if submission:
        first_query_id = next(
            iter(submission)
        )

        print(
            "Example query_id:",
            first_query_id,
        )

        print(
            "Example answer:",
            submission[first_query_id]["answer"],
        )

    invalid = []

    for query_id, item in submission.items():

        if "answer" not in item:
            invalid.append(query_id)
            continue

        answers = item["answer"]

        if not isinstance(
            answers,
            list,
        ):
            invalid.append(query_id)
            continue

        if len(answers) > args.top_k:
            invalid.append(query_id)

        if not all(
            isinstance(doc_id, str)
            for doc_id in answers
        ):
            invalid.append(query_id)

    if invalid:
        raise ValueError(
            "Invalid submission entries: "
            f"{invalid[:10]}"
        )

    print(
        "Submission format: OK"
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()