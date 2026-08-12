from src.data.schema import RetrievalResult


def aggregate_max_score(
    results: list[RetrievalResult],
    top_k: int = 100,
) -> list[RetrievalResult]:
    """
    Aggregate chunk-level retrieval results into
    document-level results using maximum score.

    For each document, only the highest-scoring
    chunk is retained as its representative result.

    Parameters
    ----------
    results : list[RetrievalResult]
        Chunk-level retrieval results.

    top_k : int
        Number of documents to return.

    Returns
    -------
    list[RetrievalResult]
        Document-level ranking represented by the
        highest-scoring chunk of each document.
    """

    best_by_document: dict[str, RetrievalResult] = {}

    for result in results:

        document_id = result.document_id

        current = best_by_document.get(document_id)

        if (
            current is None
            or result.score > current.score
        ):
            best_by_document[document_id] = result

    ranked = sorted(
        best_by_document.values(),
        key=lambda result: result.score,
        reverse=True,
    )

    ranked = ranked[:top_k]

    # Re-assign rank at document level.
    ranked = [
        RetrievalResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            score=result.score,
            rank=rank,
        )
        for rank, result in enumerate(
            ranked,
            start=1,
        )
    ]

    return ranked