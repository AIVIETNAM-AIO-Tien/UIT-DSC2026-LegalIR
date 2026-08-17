from os import cpu_count
from concurrent.futures import ProcessPoolExecutor
from itertools import chain

from src.data.schema import Chunk, Document
from src.chunking.base import Chunker

class ParallelChunker():
    """
    """
    def __init__(
        self,
        chunker: Chunker,
        max_workers: int | None=None,
    ):
        self.max_workers = max_workers or cpu_count() or 4
        self.chunker = chunker
    
    def chunk_batch(
        self, 
        documents: list[Document],
        chunksize: int = 10,
    ) -> list[Chunk]:
        """
        """
        if not documents:
            return []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(
                self.chunker.chunk,
                documents,
                chunksize=chunksize
            )
            
        return list(chain.from_iterable(results))

            
        