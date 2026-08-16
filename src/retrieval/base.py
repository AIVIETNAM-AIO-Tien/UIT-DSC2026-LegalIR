from abc import ABC, abstractmethod

from src.data.schema import Chunk, Query, RetrievalResult


class Retriever(ABC):
    """Base interface for chunk-level retrieval."""

    @abstractmethod
    def retrieve(
        self,
        query: Query,
        top_k: int = 100,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k relevant chunks for a query.

        Parameters
        ----------
        query : Query
            Query to retrieve against.

        chunks : list[Chunk]
            Chunk corpus.

        top_k : int
            Number of chunk candidates to return.

        Returns
        -------
        list[RetrievalResult]
            Ranked chunk-level retrieval results.
        """
        raise NotImplementedError
