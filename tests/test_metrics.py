from __future__ import annotations

import math
import unittest

from src.metrics import (
    catalog_coverage,
    ndcg_at_k,
    recall_at_k,
    rmse,
    validate_recommendations,
)


class MetricsTest(unittest.TestCase):
    def test_rmse_matches_hand_calculation(self) -> None:
        # Squared errors are 1 and 4, so RMSE = sqrt((1 + 4) / 2).
        self.assertAlmostEqual(rmse([3, 4], [2, 6]), math.sqrt(2.5))

    def test_recall_at_k_matches_hand_calculation(self) -> None:
        recommendations = {"alice": [1, 2, 3], "bob": [4, 5, 6]}
        relevant = {"alice": {2, 9}, "bob": {4}}

        # Alice: 1/2, Bob: 1/1, macro average: (0.5 + 1.0) / 2.
        self.assertAlmostEqual(recall_at_k(recommendations, relevant, k=2), 0.75)

    def test_ndcg_at_k_rewards_a_relevant_item_by_rank(self) -> None:
        recommendations = {"alice": [1, 2, 3]}
        relevant = {"alice": {2, 3}}

        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        ideal_dcg = 1 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(
            ndcg_at_k(recommendations, relevant, k=3),
            dcg / ideal_dcg,
        )

    def test_catalog_coverage_counts_unique_exposed_items(self) -> None:
        recommendations = {"alice": [1, 2, 3], "bob": [2, 4, 5]}

        # At K=2, the exposed games are {1, 2, 4}: 3 of the 5-game catalog.
        self.assertAlmostEqual(
            catalog_coverage(recommendations, {1, 2, 3, 4, 5}, k=2),
            0.6,
        )

    def test_validation_accepts_unique_unseen_items(self) -> None:
        validate_recommendations(
            {"alice": [2, 3], "bob": [4]},
            {"alice": {1}, "bob": {1, 2}},
            k=2,
        )

    def test_validation_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_recommendations({"alice": [2, 2]}, {"alice": {1}})

    def test_validation_rejects_seen_items(self) -> None:
        with self.assertRaisesRegex(ValueError, "seen items"):
            validate_recommendations({"alice": [1, 2]}, {"alice": {1}})

    def test_invalid_or_empty_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            rmse([1], [1, 2])
        with self.assertRaisesRegex(ValueError, "non-empty catalog"):
            catalog_coverage({}, set())
        with self.assertRaisesRegex(ValueError, "positive"):
            recall_at_k({}, {"alice": {1}}, k=0)
        with self.assertRaisesRegex(ValueError, "at least one user"):
            ndcg_at_k({}, {"alice": set()}, k=10)


if __name__ == "__main__":
    unittest.main()
