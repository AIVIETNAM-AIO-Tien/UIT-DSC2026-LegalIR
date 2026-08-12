from collections.abc import Callable

from .rank_bm25 import BM25Okapi

from src.data.schema import Chunk, Query, RetrievalResult
from src.retrieval.base import Retriever


class BM25Retriever(Retriever):
    """
    BM25 chunk-level retriever using rank_bm25.BM25Okapi.
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.tokenize_fn = tokenize_fn
        self.k1 = k1
        self.b = b

        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

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

        tokenized_corpus = [
            self.tokenize_fn(chunk.text)
            for chunk in chunks
        ]

        self._bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

    def retrieve(
        self,
        query: Query,
        chunks: list[Chunk],
        top_k: int = 1000,
    ) -> list[RetrievalResult]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

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