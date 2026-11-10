# ItemCF reduced-dataset validation summary

This report uses all users in the reduced assignment dataset for training and validation. The test split was not read. 24 ItemCF configurations
were evaluated in addition to the popularity baseline.

## Best ItemCF configuration

- Similarity: `adjusted_cosine`
- Scoring: `weighted_mean`
- Neighbors: `120`
- Minimum overlap: `10`
- Shrinkage: `250.0`
- RMSE: 1.4001
- Recall@10: 0.0962
- NDCG@10: 0.0711
- Catalog coverage: 0.1107

The best ItemCF NDCG@10 is higher than the popularity control in this
reduced validation dataset (0.0534). This is a useful diagnostic, not a
reason to inspect the test set. The winning configuration can now be frozen before one final test evaluation.

## Visual outputs

- `ranking_metrics.png`: Recall, NDCG, and coverage on separate axes.
- `rmse_vs_ndcg.png`: rating-prediction error versus ranking quality.
- `search_trials.png`: random-search and local-grid validation trials.
- `hyperparameter_effects.png`: shrinkage, neighbor K, and support effects by similarity.
