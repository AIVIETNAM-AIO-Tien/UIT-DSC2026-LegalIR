import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from itertools import chain

from src.data.schema import Chunk, Document
from src.chunking.base import Chunker


class FixedSizeChunkerParallel(Chunker):
    """
    Split documents into fixed-size token chunks using multi-core parallel processing.

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
    max_workers : int, optional
        Number of CPU processes to spin up. Defaults to total logical CPU cores.
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], list],
        decode_fn: Callable[[list], str],
        tokenizer_name: str,
        chunk_size: int = 2048,
        overlap: int = 256,
        max_workers: int | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0:
            raise ValueError("overlap must be greater than or equal to 0")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        if not tokenizer_name:
            raise ValueError("tokenizer_name must not be empty")

        self.tokenize_fn = tokenize_fn
        self.decode_fn = decode_fn
        self.tokenizer_name = tokenizer_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_workers = max_workers or os.cpu_count() or 4

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a single document into fixed-size token chunks.
        """
        tokens = self.tokenize_fn(document.text)

        if not tokens:
            return []

        chunks = []
        step = self.chunk_size - self.overlap

        for chunk_index, start in enumerate(range(0, len(tokens), step)):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            if not chunk_tokens:
                break

            chunk_text = self.decode_fn(chunk_tokens)
            chunk_id = f"{document.document_id}_{chunk_index:03d}"

            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                text=chunk_text,
                chunk_index=chunk_index,
                metadata={
                    "chunking_method": "fixed_size_parallel",
                    "tokenizer": self.tokenizer_name,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                },
            )

            chunks.append(chunk)

            if end >= len(tokens):
                break

        return chunks

    def chunk_batch(
        self,
        documents: list[Document],
        chunksize: int = 10,
    ) -> list[Chunk]:
        """
        Split a list of documents into chunks in parallel across all available CPU cores.

        Parameters
        ----------
        documents : list[Document]
            List of documents to process.
        chunksize : int
            Number of documents submitted per IPC batch to worker processes.

        Returns
        -------
        list[Chunk]
            Flattened list of generated chunks across all input documents.
        """

        if not documents:
            return []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(
                self.chunk,
                documents,
                chunksize=chunksize,
            )

        return list(chain.from_iterable(results))
