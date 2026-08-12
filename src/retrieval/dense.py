from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data.schema import Chunk, Query, RetrievalResult
from src.retrieval.base import Retriever


class DenseRetriever(Retriever):
    """
    Dense Retriever.

    Baseline:
        Model:
            AITeamVN/Vietnamese_Embedding_v2

        Chunking:
            tokenizer-based
            chunk_size=2048
            overlap=256

        Embedding:
            SentenceTransformer.encode()
            normalize_embeddings=False

        Similarity:
            dot product

    The retriever supports two workflows:

        1. build()
           Build chunks and corpus embeddings from documents.

        2. load_index()
           Load existing dense_chunks.json and
           dense_chunk_embeddings.npy.

    Retrieval itself is query-only:
        Query
          ↓
        Query embedding
          ↓
        Dot product
          ↓
        Top-k chunk results
    """

    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Embedding_v2",
        device: str | None = None,
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.tokenizer = self.model.tokenizer

        self.chunks: list[Chunk] = []

        self.chunk_embeddings: np.ndarray | None = None

    # ==============================================================
    # Chunking
    # ==============================================================

    @staticmethod
    def _chunk_text_by_tokenizer(
        text: str,
        tokenizer,
        chunk_size: int = 2048,
        overlap: int = 256,
    ) -> list[dict]:
        """
        Split text into fixed-size chunks using the model tokenizer.

        This follows the existing Kaggle baseline.
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be > 0"
            )

        if overlap < 0:
            raise ValueError(
                "overlap must be >= 0"
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        # ----------------------------------------------------------
        # Tokenize without truncation
        # ----------------------------------------------------------

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        if not token_ids:
            return []

        step = chunk_size - overlap

        chunks = []

        for start in range(
            0,
            len(token_ids),
            step,
        ):
            end = start + chunk_size

            chunk_ids = token_ids[start:end]

            if not chunk_ids:
                break

            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            chunks.append(
                {
                    "start_token": start,
                    "end_token": min(
                        end,
                        len(token_ids),
                    ),
                    "text": chunk_text,
                }
            )

            if end >= len(token_ids):
                break

        return chunks

    # ==============================================================
    # Build chunks
    # ==============================================================

    def build_chunks(
        self,
        documents,
        chunk_size: int = 2048,
        overlap: int = 256,
    ) -> list[Chunk]:
        """
        Build Dense chunks from normalized documents.

        Expected document format:

            documents[doc_id]["normalized_passage"]

        This follows the existing Kaggle baseline.
        """

        dense_chunks = []

        for doc_id, doc in tqdm(
            documents.items(),
            desc="Creating dense chunks",
        ):
            text = doc["normalized_passage"]

            chunks = self._chunk_text_by_tokenizer(
                text,
                self.tokenizer,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for chunk_index, chunk in enumerate(
                chunks
            ):
                chunk_id = (
                    f"{doc_id}_{chunk_index}"
                )

                dense_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=str(doc_id),
                        text=chunk["text"],
                        chunk_index=chunk_index,
                        metadata={
                            "chunking_method": "fixed_size",
                            "tokenizer": "sentence_transformer",
                            "chunk_size": chunk_size,
                            "overlap": overlap,
                            "start_token": chunk["start_token"],
                            "end_token": chunk["end_token"],
                        },
                    )
                )

        self.chunks = dense_chunks

        return dense_chunks

    # ==============================================================
    # Build embeddings
    # ==============================================================

    def build_embeddings(
        self,
        batch_size: int = 16,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode all Dense chunks.

        This is the expensive operation.

        It follows the existing Kaggle baseline:

            batch_size=16
            convert_to_numpy=True
            normalize_embeddings=False
        """

        if not self.chunks:
            raise RuntimeError(
                "No Dense chunks available. "
                "Call build_chunks() first."
            )

        chunk_texts = [
            chunk.text
            for chunk in self.chunks
        ]

        self.chunk_embeddings = self.model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        self.chunk_embeddings = np.asarray(
            self.chunk_embeddings
        )

        return self.chunk_embeddings

    # ==============================================================
    # Build complete Dense index
    # ==============================================================

    def build(
        self,
        documents,
        chunk_size: int = 2048,
        overlap: int = 256,
        batch_size: int = 16,
        show_progress_bar: bool = True,
    ):
        """
        Build the complete Dense index.

        Pipeline:

            documents
                ↓
            tokenizer-based chunking
                ↓
            Dense chunks
                ↓
            SentenceTransformer
                ↓
            chunk embeddings
        """

        self.build_chunks(
            documents=documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        self.build_embeddings(
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        return (
            self.chunks,
            self.chunk_embeddings,
        )

    # ==============================================================
    # Save
    # ==============================================================

    def save_index(
        self,
        chunks_path: str | Path,
        embeddings_path: str | Path,
    ):
        """
        Save Dense chunks and embeddings.

        Output format is compatible with the existing Kaggle
        baseline artifacts.
        """

        if self.chunk_embeddings is None:
            raise RuntimeError(
                "No embeddings available."
            )

        chunks_path = Path(chunks_path)
        embeddings_path = Path(embeddings_path)

        chunks_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        embeddings_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------------
        # Save chunks
        # ----------------------------------------------------------

        dense_chunks = []

        for chunk in self.chunks:

            dense_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "start_token": chunk.metadata[
                        "start_token"
                    ],
                    "end_token": chunk.metadata[
                        "end_token"
                    ],
                    "text": chunk.text,
                }
            )

        with open(
            chunks_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                dense_chunks,
                f,
                ensure_ascii=False,
            )

        # ----------------------------------------------------------
        # Save embeddings
        # ----------------------------------------------------------

        np.save(
            embeddings_path,
            self.chunk_embeddings,
        )

        print(
            f"Saved chunks: {chunks_path}"
        )

        print(
            f"Saved embeddings: {embeddings_path}"
        )

    # ==============================================================
    # Load
    # ==============================================================

    def load_index(
        self,
        chunks_path: str | Path,
        embeddings_path: str | Path,
    ):
        """
        Load an existing Dense index.

        This avoids recomputing all corpus embeddings.
        """

        chunks_path = Path(chunks_path)
        embeddings_path = Path(embeddings_path)

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Cannot find Dense chunks: {chunks_path}"
            )

        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Cannot find Dense embeddings: {embeddings_path}"
            )

        # ----------------------------------------------------------
        # Load chunks
        # ----------------------------------------------------------

        with open(
            chunks_path,
            "r",
            encoding="utf-8",
        ) as f:

            dense_chunks = json.load(f)

        if not isinstance(
            dense_chunks,
            list,
        ):
            raise ValueError(
                "dense_chunks.json must contain a list."
            )

        self.chunks = []

        for item in dense_chunks:

            self.chunks.append(
                Chunk(
                    chunk_id=str(
                        item["chunk_id"]
                    ),
                    document_id=str(
                        item["document_id"]
                    ),
                    text=item["text"],
                    chunk_index=int(
                        item["chunk_index"]
                    ),
                    metadata={
                        "chunking_method": "fixed_size",
                        "tokenizer": "sentence_transformer",
                        "chunk_size": 2048,
                        "overlap": 256,
                        "start_token": item[
                            "start_token"
                        ],
                        "end_token": item[
                            "end_token"
                        ],
                    },
                )
            )

        # ----------------------------------------------------------
        # Load embeddings
        # ----------------------------------------------------------

        self.chunk_embeddings = np.load(
            embeddings_path
        )

        # ----------------------------------------------------------
        # Validate
        # ----------------------------------------------------------

        if self.chunk_embeddings.ndim != 2:
            raise ValueError(
                "Dense embeddings must be a 2D array. "
                f"Got shape: "
                f"{self.chunk_embeddings.shape}"
            )

        if len(self.chunks) != len(
            self.chunk_embeddings
        ):
            raise ValueError(
                "Number of chunks and embeddings "
                "do not match: "
                f"{len(self.chunks)} chunks vs "
                f"{len(self.chunk_embeddings)} embeddings."
            )

        print(
            f"Loaded {len(self.chunks):,} Dense chunks."
        )

        print(
            "Embedding shape:",
            self.chunk_embeddings.shape,
        )

    # ==============================================================
    # Query encoding
    # ==============================================================

    def encode_query(
        self,
        query: Query,
    ) -> np.ndarray:
        """
        Encode a single query.

        Same configuration as the Kaggle baseline.
        """

        query_embedding = self.model.encode(
            query.question,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return np.asarray(
            query_embedding
        )

    # ==============================================================
    # Score
    # ==============================================================

    def compute_scores(
        self,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Dot-product similarity.

            scores = chunk_embeddings @ query_embedding
        """

        if self.chunk_embeddings is None:
            raise RuntimeError(
                "Dense index has not been built or loaded."
            )

        return (
            self.chunk_embeddings
            @ query_embedding
        )

    # ==============================================================
    # Retrieve
    # ==============================================================

    def retrieve(
        self,
        query: Query,
        chunks: list[Chunk] | None = None,
        top_k: int = 1000,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k Dense chunks.

        Returns chunk-level RetrievalResult.

        Document aggregation is handled separately.
        """

        if self.chunk_embeddings is None:
            raise RuntimeError(
                "Dense index has not been built or loaded."
            )

        if chunks is None:
            chunks = self.chunks

        if len(chunks) != len(
            self.chunk_embeddings
        ):
            raise ValueError(
                "Number of chunks and embeddings "
                "do not match."
            )

        if not chunks:
            return []

        # ----------------------------------------------------------
        # Query embedding
        # ----------------------------------------------------------

        query_embedding = self.encode_query(
            query
        )

        # ----------------------------------------------------------
        # Dot product
        # ----------------------------------------------------------

        scores = self.compute_scores(
            query_embedding
        )

        # ----------------------------------------------------------
        # Top-k
        # ----------------------------------------------------------

        k = min(
            top_k,
            len(scores),
        )

        top_indices = np.argpartition(
            -scores,
            k - 1,
        )[:k]

        top_indices = top_indices[
            np.argsort(
                -scores[top_indices]
            )
        ]

        # ----------------------------------------------------------
        # RetrievalResult
        # ----------------------------------------------------------

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):
            chunk = chunks[index]

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(
                        scores[index]
                    ),
                    rank=rank,
                )
            )

        return results