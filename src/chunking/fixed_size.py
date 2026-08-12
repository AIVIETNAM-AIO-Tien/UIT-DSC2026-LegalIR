from collections.abc import Callable

from src.data.schema import Chunk, Document
from src.chunking.base import Chunker


class FixedSizeChunker(Chunker):
    """
    Split a document into fixed-size token chunks.

    The tokenizer is injected into the chunker so that
    different retrieval methods can use different token
    boundaries.

    Example:
        BM25 -> PyVi tokenizer
        Dense -> Model tokenizer

    Parameters
    ----------
    tokenize_fn : Callable[[str], list]
        Function that converts text into a sequence of tokens.

    decode_fn : Callable[[list], str]
        Function that converts a token sequence back into text.

    tokenizer_name : str
        Name of the tokenizer used to create the chunks.

    chunk_size : int
        Maximum number of tokens in each chunk.

    overlap : int
        Number of overlapping tokens between consecutive chunks.
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], list],
        decode_fn: Callable[[list], str],
        tokenizer_name: str,
        chunk_size: int = 2048,
        overlap: int = 256,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if overlap < 0:
            raise ValueError(
                "overlap must be greater than or equal to 0"
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        if not tokenizer_name:
            raise ValueError(
                "tokenizer_name must not be empty"
            )

        self.tokenize_fn = tokenize_fn
        self.decode_fn = decode_fn
        self.tokenizer_name = tokenizer_name
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        """
        Split a document into fixed-size token chunks.

        Parameters
        ----------
        document : Document
            Document to split.

        Returns
        -------
        list[Chunk]
            Generated chunks.
        """

        tokens = self.tokenize_fn(document.text)

        if not tokens:
            return []

        chunks = []

        step = self.chunk_size - self.overlap

        for chunk_index, start in enumerate(
            range(0, len(tokens), step)
        ):
            end = start + self.chunk_size

            chunk_tokens = tokens[start:end]

            if not chunk_tokens:
                break

            chunk_text = self.decode_fn(
                chunk_tokens
            )

            chunk_id = (
                f"{document.document_id}_"
                f"{chunk_index:03d}"
            )

            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                text=chunk_text,
                chunk_index=chunk_index,
                metadata={
                    "chunking_method": "fixed_size",
                    "tokenizer": self.tokenizer_name,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                },
            )

            chunks.append(chunk)

            if end >= len(tokens):
                break

        return chunks