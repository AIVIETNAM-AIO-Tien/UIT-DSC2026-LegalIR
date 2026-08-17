from dataclasses import dataclass

from src.types import RawText


@dataclass
class Document:
    document_id: str
    text: RawText
    metadata: dict


@dataclass
class Query:
    query_id: str
    question: RawText
    answers: list[str] | None

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: RawText
    chunk_index: int
    metadata: dict


@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    score: float
    rank: int
