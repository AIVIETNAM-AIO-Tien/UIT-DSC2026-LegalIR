from typing import Sequence


def recall_at_k(
    ground_truth: Sequence[str],
    retrieved: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Recall@K:
        |Ground Truth ∩ Top-K Retrieved|
        --------------------------------
              |Ground Truth|

    Parameters
    ----------
    ground_truth:
        Document IDs that are considered relevant.

    retrieved:
        Ranked retrieved document IDs.
        The first element is rank 1.

    k:
        Number of top retrieved documents to evaluate.

    Returns
    -------
    float
        Recall@K in [0, 1].
    """

    if k <= 0:
        raise ValueError("k must be > 0")

    if not ground_truth:
        return 0.0

    gt = set(str(doc_id) for doc_id in ground_truth)

    top_k = set(
        str(doc_id)
        for doc_id in retrieved[:k]
    )

    return len(gt & top_k) / len(gt)