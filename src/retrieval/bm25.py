from collections.abc import Callable
import multiprocessing as mp
from functools import partial
import heapq
from typing import NamedTuple

import numpy as np

from .rank_bm25 import BM25, BM25Okapi
from src.data.schema import Chunk, Query, RetrievalResult
from src.retrieval.base import Retriever
from src.types import CorpusTokens, RawText, Token, TokenizerFunc


type ChunkId = str
type DocumentId = str
class ChunkMetaData(NamedTuple):
    chunk_id: ChunkId
    document_id: DocumentId


def _tokenize_chunk_text(chunk: Chunk, tokenize_fn: TokenizerFunc) -> list[Token]:
    return tokenize_fn(chunk.text)


type ChunkId = str
type DocumentId = str
class ChunkMetaData(NamedTuple):
    chunk_id: ChunkId
    document_id: DocumentId


def _tokenize_chunk_text(chunk: Chunk, tokenize_fn: TokenizerFunc) -> list[Token]:
    return tokenize_fn(chunk.text)


class BM25Retriever(Retriever):
    def __init__(
        self,
        tokenize_fn: TokenizerFunc,
        bm25_class: type[BM25] = BM25Okapi,
        bm25_kwargs: dict | None = None,
    ):
        self.tokenize_fn: TokenizerFunc = tokenize_fn
        self.bm25_class: type[BM25] = bm25_class
        self.bm25_kwargs = bm25_kwargs or {}

        self._bm25: BM25 | None = None
        self._chunks_len: int = 0
        # To save memory
        self._chunk_metadata: list[ChunkMetaData] = []

    def fit(
        self,
        chunks: list[Chunk],
        parallel: bool = False,
        max_workers: int = max(1, mp.cpu_count() // 2),
    ) -> "BM25Retriever":
        if not chunks:
            raise ValueError("chunks must not be empty")

        self._chunks_len = len(chunks)

        self._chunk_metadata = [
            ChunkMetaData(chunk.chunk_id, chunk.document_id) for chunk in chunks
        ]

        if parallel:
            chunksize: int = max(32, len(chunks) // (max_workers * 4))
            texts: list[RawText] = [chunk.text for chunk in chunks]

            with mp.Pool(processes=max_workers) as pool:
                corpus_tokens: CorpusTokens = pool.map(
                    self.tokenize_fn, texts, chunksize=chunksize
                )
        else:
            corpus_tokens: CorpusTokens = [self.tokenize_fn(chunk.text) for chunk in chunks]

        self._bm25 = self.bm25_class(corpus_tokens, **self.bm25_kwargs)

        return self

    def retrieve(self, query: Query, top_k: int = 100) -> list[RetrievalResult]:
        if not self._bm25:
            raise RuntimeError("No index available.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens: list[Token] = self.tokenize_fn(query.question)
        scores = self._bm25.get_scores(query_tokens)

        top_k_count: int = min(top_k, len(scores))
        partitioned_indices = np.argpartition(scores, -top_k_count)[
            -top_k_count:
        ]
        sorted_indices = partitioned_indices[
            np.argsort(-scores[partitioned_indices])
        ]

        return [
            RetrievalResult(
                chunk_id=self._chunk_metadata[idx].chunk_id,
                document_id=self._chunk_metadata[idx].document_id,
                score=float(scores[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(sorted_indices, start=1)
        ]
