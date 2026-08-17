import argparse
from os import cpu_count
from pathlib import Path
import pickle
import time

from src.data.loader import load_documents, load_queries
from src.preprocessing.normalize import normalize_text
from src.preprocessing.tokenize import tokenize_vietnamese, decode_vietnamese
from src.chunking.fixed_size_parallel import FixedSizeChunkerParallel
from utils.bm25_preprocess_arg_parser import parse_args


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE: OFFLINE PREPROCESSING & CACHING")
    print("=" * 60)

    # ----------------------------------------------------------
    # Load Documents & Queries
    # ----------------------------------------------------------
    t0 = time.time()
    print("Loading raw documents and queries...")
    documents = load_documents(args.contexts)
    queries = load_queries(args.train)
    print(f"Loaded {len(documents)} documents and {len(queries)} queries in {time.time() - t0:.2f}s")

    # ----------------------------------------------------------
    # Text Normalization
    # ----------------------------------------------------------
    t0 = time.time()
    print("\nNormalizing document text...")
    for doc in documents:
        doc.text = normalize_text(doc.text)
    print(f"Document normalization complete in {time.time() - t0:.2f}s")

    # ----------------------------------------------------------
    # Parallel Chunking
    # ----------------------------------------------------------
    t0 = time.time()
    print("\nChunking documents...")
    chunker = FixedSizeChunkerParallel(
        tokenize_fn=tokenize_vietnamese,
        decode_fn=decode_vietnamese,
        tokenizer_name="pyvi",
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_workers=max(1, cpu_count()-2)
    )

    chunks = chunker.chunk_batch(documents)

    print(f"Generated {len(chunks)} total chunks in {time.time() - t0:.2f}s")

    print(f"Tokenization complete in {time.time() - t0:.2f}s")

    # ----------------------------------------------------------
    # Cache Preprocessed Artifacts
    # ----------------------------------------------------------
    print("\nSaving Preprocessed Data to Disk...")

    chunks_file = (
        output_dir / f"chunks_cs{args.chunk_size}_ov{args.overlap}.pkl"
    )
    queries_file = output_dir / "queries_processed.pkl"

    t0 = time.time()
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(queries_file, "wb") as f:
        pickle.dump(queries, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f" Saved preprocessed chunks to: {chunks_file}")
    print(f" Saved preprocessed queries to: {queries_file}")
    print(f" Serialization finished in {time.time() - t0:.2f}s.")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
