from abc import ABC, abstractmethod

from src.data.schema import Chunk, Document


class Chunker(ABC):
    """Base interface for document chunkers."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks.

        Parameters
        ----------
        document : Document
            Document to split.

        Returns
        -------
        list[Chunk]
            Generated chunks.
        """
        raise NotImplementedError