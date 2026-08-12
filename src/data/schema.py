from dataclasses import dataclass


@dataclass
class Document:
    document_id: str
    text: str
    metadata: dict


@dataclass
class Query:
    query_id: str
    question: str
    answers: list[str] | None

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict


@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    score: float
    rank: int