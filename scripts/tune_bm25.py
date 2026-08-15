import argparse
import json
import pickle
import time

from pathlib import Path
from os import cpu_count

# Project imports
from src.evaluation.evaluator import Evaluator
from src.preprocessing.tokenize import tokenize_vietnamese
from src.retrieval.aggregation import aggregate_max_score
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.bm25_parrallel import ParallelBM25Retriever
from src.retrieval.rank_bm25 import BM25L, BM25Okapi, BM25Plus

BM25_VARIANTS = {
    "okapi": BM25Okapi,
    "bm25l": BM25L,
    "bm25plus": BM25Plus,
}

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

    # Accept multiple values for k1 and b to enable parameter sweeps
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


def build_bm25_kwargs(variant, k1, b, epsilon, delta):
    if variant == "okapi":
        return {"k1": k1, "b": b, "epsilon": epsilon}
    elif variant in {"bm25l", "bm25plus"}:
        return {"k1": k1, "b": b, "delta": delta}
    raise ValueError(f"Unknown BM25 variant: {variant}")


def run_experiment(
    chunks,
    queries,
    ground_truth,
    bm25_class,
    bm25_kwargs,
    retrieval_k,
    eval_k,
    evaluator,
):
    # ----------------------------------------------------------
    # Build BM25 Index
    # ----------------------------------------------------------
    print('Fitting start.')
    t_start = time.time()

    retriever = ParallelBM25Retriever(
        tokenize_fn=tokenize_vietnamese,
        bm25_class=bm25_class,
        bm25_kwargs=bm25_kwargs,
    )
    # Use parallel tokenization during fit
    retriever.fit_parallel(
        chunks,
        max_workers=max(1, cpu_count()//2)
    )

    fit_time = time.time() - t_start
    print(f'Fitted, took {fit_time}')

    # ----------------------------------------------------------
    # Chunk Retrieval & Score Aggregation
    # ----------------------------------------------------------
    print('Retrieval starts.')
    t_start = time.time()
    retrieved_documents = {}

    total_queries = len(queries)
    log_interval = max(1, total_queries // 10) if total_queries > 0 else 1

    for i, query in enumerate(queries, 1):
        chunk_results = retriever.retrieve(query=query, chunks=chunks)

        document_results = aggregate_max_score(
            chunk_results,
            top_k=retrieval_k,
        )

        retrieved_documents[query.query_id] = [
            res.document_id for res in document_results
        ]

        if i % log_interval == 0 or i == total_queries:
            progress_percent = int((i / total_queries) * 100) if total_queries > 0 else 100
            print(f'Retrieve progress: {progress_percent}%')

    retrieval_time = time.time() - t_start

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------
    t_start = time.time()
    evaluation = evaluator.evaluate(
        ground_truth=ground_truth,
        retrieved=retrieved_documents,
    )
    eval_time = time.time() - t_start

    return evaluation, retrieved_documents, fit_time, retrieval_time, eval_time


def main():
    args = parse_args()
    bm25_class = BM25_VARIANTS[args.variant]

    # Load cached data directly
    print("=" * 60)
    print("LOADING PREPROCESSED CACHE")
    print("=" * 60)
    t0 = time.time()

    with open(args.chunks_file, "rb") as f:
        chunks = pickle.load(f)

    with open(args.queries_file, "rb") as f:
        queries = pickle.load(f)

    print(
        f" Loaded {len(chunks)} chunks & {len(queries)} queries in {time.time() - t0:.2f}s"
    )

    # Construct ground truth mapping
    ground_truth = {query.query_id: query.answers for query in queries}
    evaluator = Evaluator(k_values=args.eval_k)

    grid_results = []
    best_score = -1.0
    best_params = None
    best_predictions = None

    total_experiments = len(args.k1) * len(args.b)
    exp_counter = 0

    print("\n" + "=" * 60)
    print(f"RUNNING EXPERIMENTS ({total_experiments} Total Combination(s))")
    print("=" * 60)

    # ----------------------------------------------------------
    # Hyperparameter Search Loop over k1 and b
    # ----------------------------------------------------------
    for k1_val in args.k1:
        for b_val in args.b:
            exp_counter += 1
            bm25_kwargs = build_bm25_kwargs(
                args.variant, k1_val, b_val, args.epsilon, args.delta
            )

            print(
                f"\n[{exp_counter}/{total_experiments}] Testing k1={k1_val}, b={b_val}..."
            )

            evaluation, retrieved_docs, fit_t, ret_t, eval_t = run_experiment(
                chunks=chunks,
                queries=queries,
                ground_truth=ground_truth,
                bm25_class=bm25_class,
                bm25_kwargs=bm25_kwargs,
                retrieval_k=args.retrieval_k,
                eval_k=args.eval_k,
                evaluator=evaluator,
            )

            # Store iteration metrics
            primary_metric = list(evaluation["aggregate"].keys())[0]
            primary_score = evaluation["aggregate"][primary_metric]

            print(
                f"   Index fit: {fit_t:.2f}s | Retrieval: {ret_t:.2f}s | Eval: {eval_t:.2f}s"
            )
            print(f"   {primary_metric}: {primary_score:.4f}")

            run_summary = {
                "parameters": {"k1": k1_val, "b": b_val},
                "bm25_kwargs": bm25_kwargs,
                "evaluation": evaluation,
            }
            grid_results.append(run_summary)

            if primary_score > best_score:
                best_score = primary_score
                best_params = {"k1": k1_val, "b": b_val}
                best_predictions = retrieved_docs

    # ----------------------------------------------------------
    # Step 8: Export Results & Summary Reporting
    # ----------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "configuration": {
            "bm25_variant": args.variant,
            "retrieval_k": args.retrieval_k,
            "eval_k": args.eval_k,
        },
        "best_parameters": best_params,
        "best_score": best_score,
        "all_experiments": grid_results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    if args.save_retrieval and best_predictions:
        retrieval_path = Path(args.save_retrieval)
        retrieval_path.parent.mkdir(parents=True, exist_ok=True)
        with retrieval_path.open("w", encoding="utf-8") as f:
            json.dump(best_predictions, f, ensure_ascii=False, indent=4)
        print(f"\nBest retrieval predictions saved to: {retrieval_path}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f" Best Parameters : {best_params}")
    print(f" Best Score      : {best_score:.4f}")
    print(f" Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
