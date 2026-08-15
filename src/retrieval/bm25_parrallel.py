from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from functools import partial
from src.retrieval.bm25 import BM25Retriever

# Helper top-level function for multiprocessing pickling
def _tokenize_chunk_text(chunk, tokenize_fn):
    return tokenize_fn(chunk.text)

class ParallelBM25Retriever(BM25Retriever):
    def fit_parallel(self, chunks, max_workers=None):
        if max_workers is None:
            max_workers = max(1, cpu_count() - 2)

        # Parallel Tokenization across CPU cores
        worker_fn = partial(_tokenize_chunk_text, tokenize_fn=self.tokenize_fn)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map chunk tokenization across available processes
            corpus_tokens = list(executor.map(worker_fn, chunks, chunksize=500))

        # Initialize underlying BM25 class with pre-tokenized corpus
        self.bm25 = self.bm25_class(
            corpus=corpus_tokens,
            **self.bm25_kwargs
        )
        
        return self
