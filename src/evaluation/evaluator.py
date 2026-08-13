from typing import Sequence

from src.evaluation.metrics import recall_at_k


DEFAULT_K_VALUES = [
    5,
    10,
    20,
    50,
    100,
]


class Evaluator:
    """
    Evaluate retrieval results over multiple queries.

    This class is independent of the retrieval method.

    Expected input:

        ground_truth:
            query_id -> list[document_id]

        retrieved:
            query_id -> ranked list[document_id]

    Example
    -------
        ground_truth = {
            "19826": ["44802", "65293"],
            "88634": ["33079"],
        }

        retrieved = {
            "19826": ["33079", "44802", ...],
            "88634": ["12345", "33079", ...],
        }
    """

    def __init__(
        self,
        k_values: Sequence[int] | None = None,
    ):
        if k_values is None:
            k_values = DEFAULT_K_VALUES

        self.k_values = list(k_values)

        if not self.k_values:
            raise ValueError(
                "k_values must not be empty"
            )

        if any(k <= 0 for k in self.k_values):
            raise ValueError(
                "All k values must be > 0"
            )

        self.k_values = sorted(
            set(self.k_values)
        )

    def evaluate_query(
        self,
        ground_truth: Sequence[str],
        retrieved: Sequence[str],
    ) -> dict[str, float]:
        """
        Evaluate one query.
        """

        results = {}

        for k in self.k_values:
            results[f"recall@{k}"] = recall_at_k(
                ground_truth=ground_truth,
                retrieved=retrieved,
                k=k,
            )

        return results

    def evaluate(
        self,
        ground_truth: dict[str, Sequence[str]],
        retrieved: dict[str, Sequence[str]],
    ) -> dict:
        """
        Evaluate all queries.

        Returns
        -------
        dict
            {
                "aggregate": {
                    "recall@1": ...,
                    "recall@5": ...,
                    ...
                },
                "per_query": {
                    "query_id": {
                        "recall@1": ...,
                        ...
                    }
                }
            }
        """

        per_query = {}

        for query_id, gt in ground_truth.items():

            retrieved_docs = retrieved.get(
                query_id,
                [],
            )

            per_query[str(query_id)] = (
                self.evaluate_query(
                    ground_truth=gt,
                    retrieved=retrieved_docs,
                )
            )

        # ----------------------------------------------------------
        # Aggregate
        # ----------------------------------------------------------

        aggregate = {}

        num_queries = len(per_query)

        if num_queries == 0:
            for k in self.k_values:
                aggregate[f"recall@{k}"] = 0.0

        else:
            for k in self.k_values:

                metric_name = f"recall@{k}"

                values = [
                    result[metric_name]
                    for result in per_query.values()
                ]

                aggregate[metric_name] = (
                    sum(values) / len(values)
                )

        return {
            "aggregate": aggregate,
            "per_query": per_query,
        }