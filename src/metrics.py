"""Evaluation metrics shared by recommendation models.

The functions in this module intentionally use only the Python standard library.
This keeps the metric definitions easy to inspect and makes them usable by every
model without converting results to a framework-specific data type.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from typing import TypeVar


UserId = TypeVar("UserId", bound=Hashable)
ItemId = TypeVar("ItemId", bound=Hashable)


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Return root mean squared error for paired rating predictions."""
    actual_values = list(actual)
    predicted_values = list(predicted)
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted must have the same length")
    if not actual_values:
        raise ValueError("RMSE requires at least one prediction")
    squared_error = sum(
        (float(observed) - float(estimate)) ** 2
        for observed, estimate in zip(actual_values, predicted_values)
    )
    return math.sqrt(squared_error / len(actual_values))


def validate_recommendations(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    seen_items: Mapping[UserId, Collection[ItemId]],
    *,
    k: int | None = None,
) -> None:
    """Raise ``ValueError`` if a recommendation list is not valid.

    A recommender must not repeat an item or recommend an item already observed
    by that user. When ``k`` is supplied, lists must contain at most ``k`` items.
    """
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    for user_id, items in recommendations.items():
        if k is not None and len(items) > k:
            raise ValueError(f"user {user_id!r} has more than {k} recommendations")
        if len(set(items)) != len(items):
            raise ValueError(f"user {user_id!r} has duplicate recommendations")
        overlap = set(items).intersection(seen_items.get(user_id, ()))
        if overlap:
            raise ValueError(
                f"user {user_id!r} was recommended seen items: {sorted(overlap, key=str)!r}"
            )


def recall_at_k(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    relevant_items: Mapping[UserId, Collection[ItemId]],
    k: int,
) -> float:
    """Return mean Recall@K over users that have at least one relevant item.

    Recall@K answers a simple question: among the games the user actually liked,
    what fraction appeared in the first K recommendations?
    """
    _require_positive_k(k)
    scores = []
    for user_id, relevant in relevant_items.items():
        relevant_set = set(relevant)
        if not relevant_set:
            continue
        recommended = set(recommendations.get(user_id, ())[:k])
        scores.append(len(recommended.intersection(relevant_set)) / len(relevant_set))
    return _mean_user_score(scores, "Recall@K")


def ndcg_at_k(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    relevant_items: Mapping[UserId, Collection[ItemId]],
    k: int,
) -> float:
    """Return mean binary NDCG@K over users with relevant items.

    NDCG gives more credit when a relevant game appears near the top. The score
    is divided by the best possible ordering, so 1.0 means a perfect ranking.
    """
    _require_positive_k(k)
    scores = []
    for user_id, relevant in relevant_items.items():
        relevant_set = set(relevant)
        if not relevant_set:
            continue
        ranking = recommendations.get(user_id, ())[:k]
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, item_id in enumerate(ranking)
            if item_id in relevant_set
        )
        ideal_hits = min(k, len(relevant_set))
        ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        scores.append(dcg / ideal_dcg)
    return _mean_user_score(scores, "NDCG@K")


def catalog_coverage(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    catalog_items: Collection[ItemId],
    *,
    k: int | None = None,
) -> float:
    """Return the fraction of catalog items exposed by the recommendations."""
    catalog = set(catalog_items)
    if not catalog:
        raise ValueError("catalog coverage requires a non-empty catalog")
    if k is not None:
        _require_positive_k(k)
    exposed: set[ItemId] = set()
    for items in recommendations.values():
        exposed.update(items if k is None else items[:k])
    unknown = exposed.difference(catalog)
    if unknown:
        raise ValueError(f"recommendations contain items outside the catalog: {unknown!r}")
    return len(exposed) / len(catalog)


def _require_positive_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def _mean_user_score(scores: list[float], metric_name: str) -> float:
    if not scores:
        raise ValueError(f"{metric_name} requires at least one user with relevant items")
    return sum(scores) / len(scores)
