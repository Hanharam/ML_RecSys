# Final ItemCF test result

The validation winner was frozen before this evaluation. No test metric was
used to select or change the model or its hyperparameters.

## Frozen configuration

- Similarity: `adjusted_cosine`
- Scoring: `weighted_mean`
- Neighbors: `120`
- Minimum overlap: `10`
- Shrinkage: `250.0`
- Minimum similarity: `0.1`

## Validation and final test metrics

| Split | Rating error (RMSE) ↓ | Liked-game hit rate (Recall@10) ↑ | Ranking quality (NDCG@10) ↑ | Game diversity (Coverage) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Validation | 1.4001 | 0.0962 | 0.0711 | 0.1107 |
| **Test (final)** | **1.3974** | **0.1005** | **0.0735** | **0.1107** |

- Test rating predictions: 88,677
- Test users with at least one relevant game: 8,721
- Fit time: 0.26 seconds
- Test evaluation time: 1.95 seconds
