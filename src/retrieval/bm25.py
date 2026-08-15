from collections.abc import Callable

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
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def fit(
        self,
        chunks: list[Chunk],
    ) -> None:
        """Build the BM25 index from chunks."""

        if not chunks:
            raise ValueError(
                "chunks must not be empty"
            )

        self._chunks = chunks

        self._tokenized_corpus = [
            self.tokenize_fn(chunk.text)
            for chunk in chunks
        ]

        self._bm25 = self.bm25_class(
            self._tokenized_corpus,
            **self.bm25_kwargs,
        )

    def retrieve(
        self,
        query: Query,
        chunks: list[Chunk] | None = None,
        top_k: int = 1000,
    ) -> list[RetrievalResult]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if chunks is None:
            chunks = self._chunks
            
        if not chunks:
            return []

        # Build index if needed.
        if self._bm25 is None:
            self.fit(chunks)

        query_tokens = self.tokenize_fn(
            query.question
        )

        scores = self._bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(chunks)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            chunk = chunks[index]

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(scores[index]),
                    rank=rank,
                )
            )

        return results