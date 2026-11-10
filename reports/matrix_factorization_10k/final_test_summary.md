# Matrix Factorization 최종 test 결과

## 결론

Validation에서 선택한 hyperparameter를 변경하지 않고 held-out test에 정확히 한 번 평가했다. 따라서 아래 test metric은 tuning에 다시 사용하지 않은 최종 결과다.

| Split | RMSE ↓ | Recall@10 ↑ | NDCG@10 ↑ | Catalog coverage ↑ |
| --- | ---: | ---: | ---: | ---: |
| Validation | 1.2492 | 0.0324 | 0.0235 | 0.6087 |
| **Test (final)** | **1.2416** | **0.0367** | **0.0269** | **0.6120** |

## Frozen configuration

- Latent factors: `32`
- Learning rate: `0.02`
- Regularization: `0.01`
- Epochs: `18`
- Batch size: `4096`
- Random seed: `42`

## 결과를 쉽게 해석하면

- **RMSE**는 예측 rating이 실제 rating에서 평균적으로 얼마나 벗어나는지 보여준다.
- **Recall@10**은 사용자가 실제로 좋아한 게임 중 추천 Top-10이 찾아낸 비율이다.
- **NDCG@10**은 좋아한 게임을 단순히 찾는 것뿐 아니라 추천 목록 앞쪽에 배치했는지도 반영한다.
- **Coverage**는 여러 사용자에게 추천된 게임을 합쳤을 때 catalog를 얼마나 폭넓게 사용했는지 보여준다.

Validation과 test 값이 비슷하면 특정 validation sample에만 우연히 맞은 모델일 가능성이 낮다. 반대로 차이가 크면 overfitting이나 split 차이를 의심해야 한다.

## 기존 ItemCF와 비교

동일한 reduced dataset과 test protocol에서 기존 ItemCF 결과도 함께 보면 모델의 성격 차이가 더 분명하다.

| Model | RMSE ↓ | Recall@10 ↑ | NDCG@10 ↑ | Coverage ↑ |
| --- | ---: | ---: | ---: | ---: |
| ItemCF | 1.3974 | 0.1005 | 0.0735 | 0.1107 |
| Matrix Factorization | **1.2416** | 0.0367 | 0.0269 | **0.6120** |

Matrix Factorization은 rating prediction error와 catalog coverage에서 강점이 있었고, ItemCF는 Recall@10과 NDCG@10에서 더 좋았다. 따라서 이 실험에서는 “평점을 얼마나 정확히 예측하는가”가 목적이면 Matrix Factorization이 유리하지만, “좋아할 게임을 Top-10 앞쪽에 배치하는가”가 목적이면 현재 ItemCF가 더 적합하다.


## 실행 정보

- Test rating predictions: 88,677
- Relevant item이 하나 이상인 test users: 8,721
- Fit time: 6.32 seconds
- Test evaluation time: 0.84 seconds
