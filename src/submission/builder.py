import json
from pathlib import Path


def build_submission(
    retrieval_results: dict[str, list[str]],
    output_path: str | Path,
    top_k: int = 5,
):
    """
    Build official submission.json.

    Parameters
    ----------
    retrieval_results:
        Mapping:

            question_id -> ranked document_ids

        Document IDs must already be sorted by
        descending relevance.

    output_path:
        Path to submission.json.

    top_k:
        Number of documents submitted per question.
        Official competition limit is 5.

    Returns
    -------
    dict
        Submission object.
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    if top_k > 5:
        raise ValueError(
            "Official submission allows at most 5 documents per question."
        )

    submission = {}

    for question_id, document_ids in retrieval_results.items():

        submission[str(question_id)] = {
            "answer": [
                str(document_id)
                for document_id in document_ids[:top_k]
            ]
        }

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            submission,
            f,
            ensure_ascii=False,
            indent=4,
        )

    return submission