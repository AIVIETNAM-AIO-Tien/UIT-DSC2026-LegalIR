import argparse
import json
import pickle
import time

from dataclasses import asdict
from pathlib import Path
from os import cpu_count

# Project imports
from src.evaluation.evaluator import Evaluator
from src.preprocessing.tokenize import tokenize_vietnamese
from src.retrieval.aggregation import aggregate_max_score
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.rank_bm25 import BM25L, BM25Okapi, BM25Plus
from utils.bm25_arg_parser import parse_args

BM25_VARIANTS = {
    "okapi": BM25Okapi,
    "bm25l": BM25L,
    "bm25plus": BM25Plus,
}

# ==============================================================
# BM25 configuration
# ==============================================================

def build_bm25_kwargs(variant, k1, b, epsilon=0, delta=0):
    if variant == "okapi":
        return {"k1": k1, "b": b, "epsilon": epsilon}
    elif variant in {"bm25l", "bm25plus"}:
        return {"k1": k1, "b": b, "delta": delta}
    raise ValueError(f"Unknown BM25 variant: {variant}")

# ==============================================================
# BM25 experiment
# ==============================================================

def run_experiment(
    chunks,
    queries,
    ground_truth,
    bm25_class,
    bm25_kwargs,
    retrieval_k_chunks,
    retrieval_k_docs,
    evaluator,
):
    # ----------------------------------------------------------
    # Build BM25 Index
    # ----------------------------------------------------------
    print('Fitting start.')
    t_start = time.time()

    retriever = BM25Retriever(
        tokenize_fn=tokenize_vietnamese,
        bm25_class=bm25_class,
        bm25_kwargs=bm25_kwargs,
    )
    # Use parallel tokenization during fit
    retriever.fit(
        chunks,
        parallel=True
    )

    fit_time = time.time() - t_start
    print(f'Fitted, took {fit_time}')

    # ----------------------------------------------------------
    # Chunk Retrieval & Score Aggregation
    # ----------------------------------------------------------
    print('Retrieval starts.')
    t_start = time.time()
    retrieved_documents = {}
    chunk_results = {}
    document_results = {}

    total_queries = len(queries)
    log_interval = max(1, total_queries // 50) if total_queries > 0 else 1

    for i, query in enumerate(queries, 1):
        chunk_results[query.query_id] = retriever.retrieve(query=query, top_k=retrieval_k_chunks)

        document_results[query.query_id] = aggregate_max_score(chunk_results[query.query_id], top_k=retrieval_k_docs)

        retrieved_documents[query.query_id] = [ res.document_id for res in document_results[query.query_id] ]

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

    return evaluation, retrieved_documents, chunk_results, document_results, fit_time, retrieval_time, eval_time

def main():

    args = parse_args()
    # ==========================================================
    # Validate
    # ==========================================================

    if args.retrieval_k_chunks <= 0:
        raise ValueError(
            "--retrieval-k-chunks must be > 0"
        )

    if args.retrieval_k_docs <= 0:
        raise ValueError(
            "--retrieval-k-docs must be > 0"
        )

    # ==========================================================
    # Configuration
    # ==========================================================

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

    total_experiments = len(args.k1) * len(args.b) * (len(args.epsilon) if args.variant == "okapi" else len(args.delta))
    exp_counter = 0

    print("\n" + "=" * 60)
    print(f"RUNNING EXPERIMENTS ({total_experiments} Total Combination(s))")
    print("=" * 60)


    # ----------------------------------------------------------
    # Hyperparameter Search Loop
    # ----------------------------------------------------------
    
    for k1_val in args.k1:
        for b_val in args.b:
            for penalty_val in (args.epsilon if args.variant == "okapi" else args.delta):
                exp_counter += 1
                if args.variant == "okapi":
                    bm25_kwargs = build_bm25_kwargs(
                        args.variant, k1_val, b_val, epsilon=penalty_val
                    )
                else:    
                    bm25_kwargs = build_bm25_kwargs(
                        args.variant, k1_val, b_val, delta=penalty_val
                    )
                                        
                print(
                    f"\n[{exp_counter}/{total_experiments}] Testing k1={k1_val}, b={b_val}, penalty={penalty_val},..."
                )

                evaluation, retrieved_docs, top_chunk_results, top_document_results,fit_t, ret_t, eval_t = run_experiment(
                    chunks=chunks,
                    queries=queries,
                    ground_truth=ground_truth,
                    bm25_class=bm25_class,
                    bm25_kwargs=bm25_kwargs,
                    retrieval_k_chunks=args.retrieval_k_chunks,
                    retrieval_k_docs=args.retrieval_k_docs,
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
                    "bm25_kwargs": bm25_kwargs,
                    "evaluation": evaluation["aggregate"],
                }
                grid_results.append(run_summary)

                if primary_score > best_score:
                    best_score = primary_score
                    best_params = {"k1": k1_val, "b": b_val, "penalty:": penalty_val}
                    best_predictions = retrieved_docs


    # ----------------------------------------------------------
    # Export Results & Summary Reporting
    # ----------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "configuration": {
            "bm25_variant": args.variant,
            "retrieval_k_chunks": args.retrieval_k_chunks,
            "retrieval_k_docs": args.retrieval_k_docs,
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
        
    if args.save_top_chunks_result and top_chunk_results:
        chunks_result_path = Path(args.save_top_chunks_result)
        chunks_result_path.parent.mkdir(parents=True, exist_ok=True)

        with chunks_result_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    query_id: [asdict(result) for result in results]
                    for query_id, results in top_chunk_results.items()
                },
                f,
                ensure_ascii=False,
                indent=4
            )

        print(f"\nTop Chunk results saved to: {chunks_result_path}")


    if args.save_top_docs_result and top_document_results:
        docs_result_path = Path(args.save_top_docs_result)
        docs_result_path.parent.mkdir(parents=True, exist_ok=True)

        with docs_result_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    query_id: [asdict(result) for result in results]
                    for query_id, results in top_document_results.items()
                },
                f,
                ensure_ascii=False,
                indent=4
            )

        print(f"\nTop Document results saved to: {docs_result_path}")
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f" Best Parameters : {best_params}")
    print(f" Best Score      : {best_score:.4f}")
    print(f" Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
    
