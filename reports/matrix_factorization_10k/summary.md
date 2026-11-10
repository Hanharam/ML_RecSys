# Matrix Factorization validation 결과

## 한눈에 보는 결론

10,000명 전체 reduced dataset에서 총 8개 hyperparameter 조합을 비교했다. 최종 모델은 **validation NDCG@10**이 가장 높은 조합으로 선택했으며, 이 과정에서 test data는 읽지 않았다.

- Best trial: `T2`
- Latent factors: `32`
- Learning rate: `0.02`
- Regularization: `0.01`
- Epochs: `18`
- RMSE: **1.2492**
- Recall@10: **0.0324**
- NDCG@10: **0.0235**
- Catalog coverage: **0.6087**

## 무엇을 구현했나

Matrix Factorization은 큰 user-game rating matrix를 두 개의 작은 embedding matrix로 나눈다. 각 user와 game은 `32`차원의 latent vector를 가지며, 두 vector의 dot product가 취향의 궁합을 나타낸다. 여기에 전체 평균, user bias, game bias를 더해 rating을 예측한다.

`predicted rating = global mean + user bias + game bias + user embedding · game embedding`

user bias는 평소 점수를 높거나 낮게 주는 성향을, game bias는 전반적으로 평가가 좋은지 나쁜지를 담당한다. embedding은 그 두 효과만으로 설명되지 않는 취향 pattern을 학습한다.

## Hyperparameter를 어떻게 선택했나

- `latent factors`: 취향을 몇 개의 숨은 축으로 표현할지 결정한다. 너무 작으면 표현력이 부족하고, 너무 크면 overfitting과 계산량이 증가할 수 있다.
- `learning rate`: 한 번의 update 크기다. 너무 작으면 학습이 느리고, 너무 크면 불안정해질 수 있다.
- `regularization`: embedding과 bias가 지나치게 커지는 것을 막아 overfitting을 줄인다.
- `epochs`: 전체 training interaction을 반복해서 학습하는 횟수다.

선택 기준은 Top-10 추천 순서를 반영하는 NDCG@10이다. RMSE가 낮아도 사용자가 좋아할 게임을 상위에 배치하지 못할 수 있으므로 RMSE만으로 모델을 고르지 않았다. NDCG@10이 같으면 Recall@10, RMSE, coverage 순서로 비교했다.

실제로 가장 낮은 RMSE는 `T4`의 **1.1923**였지만, 이 trial의 NDCG@10은 **0.0225**로 `T2`보다 낮았다. 반대로 선택된 `T2`는 RMSE가 더 높지만 NDCG@10과 coverage가 높았다. 즉, rating 값을 정확히 맞히는 설정과 좋은 추천 순서를 만드는 설정이 같지 않을 수 있다는 결과다.

## Metric 읽는 법

| Metric | 의미 | 좋은 방향 |
| --- | --- | --- |
| RMSE | 실제 rating과 예측 rating의 평균적인 오차 크기 | 낮을수록 좋음 |
| Recall@10 | 실제로 좋아한 게임 중 Top-10에 들어간 비율 | 높을수록 좋음 |
| NDCG@10 | 좋아한 게임이 Top-10의 앞쪽에 배치됐는지까지 반영 | 높을수록 좋음 |
| Catalog coverage | 추천 결과 전체가 catalog의 몇 %를 노출했는지 | 높을수록 추천 범위가 넓음 |

## 결과 해석 시 주의점

이 결과는 validation 결과다. Hyperparameter 선택에 validation을 사용했으므로 최종 성능은 `best_mf.json`의 frozen configuration을 held-out test에 정확히 한 번 평가해 확인한다.

## 생성된 파일

- `mf_trials.csv`: 모든 trial의 설정과 metric
- `best_mf.json`: validation winner와 frozen hyperparameter
- `ranking_metrics.png`: Recall@10, NDCG@10, coverage 비교
- `rmse_vs_ndcg.png`: rating prediction과 ranking quality의 관계
- `hyperparameter_effects.png`: learning rate, factors, regularization의 관계
