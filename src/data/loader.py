import json
from pathlib import Path

from .schema import Document, Query


def load_documents(contexts_dir: str | Path) -> list[Document]:
    """
    Load legal documents from selected-contexts directory.

    Each file is expected to have the format:

    {
        "id": 740,
        "name": "...",
        "link": "...",
        "passage": "..."
    }

    Returns
    -------
    list[Document]
        List of loaded legal documents.
    """

    contexts_dir = Path(contexts_dir)

    if not contexts_dir.exists():
        raise FileNotFoundError(
            f"Context directory not found: {contexts_dir}"
        )

    documents = []

    for file_path in sorted(contexts_dir.glob("context_*.json")):

        with file_path.open(
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        document = Document(
            document_id=str(data["id"]),
            text=data["passage"],
            metadata={
                "name": data.get("name"),
                "link": data.get("link"),
            },
        )

        documents.append(document)

    return documents


def load_queries(path: str) -> list[Query]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    queries = []

    for query_id, item in data.items():

        answers = item.get("answer")

        if answers is not None:
            answers = [
                str(answer_id)
                for answer_id in answers
            ]

        queries.append(
            Query(
                query_id=str(query_id),
                question=item["question"],
                answers=answers,
            )
        )

    return queries