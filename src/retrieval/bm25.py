from collections.abc import Callable
import multiprocessing as mp
from functools import partial
import heapq
import numpy as np

from .rank_bm25 import BM25Okapi
from src.data.schema import Chunk, Query, RetrievalResult
from src.retrieval.base import Retriever



class BM25Retriever(Retriever):
    """
    BM25 chunk-level retriever.

    The BM25 algorithm is configurable so that the same
    retriever interface can be used with:

        - BM25Okapi
        - BM25L
        - BM25Plus

    The Vietnamese tokenization is handled separately by
    tokenize_fn.
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], list[str]],
        bm25_class=BM25Okapi,
        bm25_kwargs: dict | None = None,

    ):
        self.tokenize_fn = tokenize_fn
        self.bm25_class = bm25_class
        self.bm25_kwargs = bm25_kwargs or {}

        self._bm25 = None
        self._chunks_len = 0
        # To save memory
        self._chunk_metadata: list[tuple[str, str]] = [] # (chunk_id, document_id)

    def fit(
        self,
        chunks: list[Chunk],
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> "BM25Retriever":
        if not chunks:
            raise ValueError("chunks must not be empty")

        self._chunks_len = len(chunks)

        self._chunk_metadata = [
            (chunk.chunk_id, chunk.document_id) for chunk in chunks
        ]

        if parallel:
            if max_workers is None:
                max_workers = max(1, mp.cpu_count() // 2)

            chunksize = max(32, len(chunks) // (max_workers * 4))
            texts = [chunk.text for chunk in chunks]

            with mp.Pool(processes=max_workers) as pool:
                corpus_tokens = pool.map(
                    self.tokenize_fn, texts, chunksize=chunksize
                )
        else:
            corpus_tokens = [self.tokenize_fn(chunk.text) for chunk in chunks]

        self._bm25 = self.bm25_class(corpus_tokens, **self.bm25_kwargs)

        return self


    def retrieve(self, query: Query, top_k: int = 1000) -> list[RetrievalResult]:
        if not self._bm25:
            raise RuntimeError("No index available.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = self.tokenize_fn(query.question)
        scores = self._bm25.get_scores(query_tokens)

        top_k_count = min(top_k, len(scores))
        partitioned_indices = np.argpartition(scores, -top_k_count)[
            -top_k_count:
        ]
        sorted_indices = partitioned_indices[
            np.argsort(-scores[partitioned_indices])
        ]

        return [
            RetrievalResult(
                chunk_id=self._chunk_metadata[idx][0],
                document_id=self._chunk_metadata[idx][1],
                score=float(scores[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(sorted_indices, start=1)
        ]