"""Evaluate the frozen Matrix Factorization validation winner on test once."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from src.data import load_interactions
from src.matrix_factorization import MatrixFactorization, MatrixFactorizationConfig
from src.matrix_factorization_experiment import evaluate_model, validation_targets


def load_frozen_config(path: Path) -> tuple[MatrixFactorizationConfig, dict[str, float]]:
    """Load only model parameters and validation metrics from the winner file."""

    winner = json.loads(path.read_text(encoding="utf-8"))
    config = MatrixFactorizationConfig(
        factors=int(winner["factors"]),
        learning_rate=float(winner["learning_rate"]),
        regularization=float(winner["regularization"]),
        epochs=int(winner["epochs"]),
        batch_size=int(winner["batch_size"]),
        seed=int(winner["seed"]),
        min_rating=float(winner["min_rating"]),
        max_rating=float(winner["max_rating"]),
    )
    validation = {
        metric: float(winner[metric])
        for metric in ("rmse", "recall_at_10", "ndcg_at_10", "coverage")
    }
    return config, validation


def write_final_report(
    output_dir: Path,
    config: MatrixFactorizationConfig,
    validation: dict[str, float],
    test: dict[str, float],
    *,
    fit_seconds: float,
    evaluation_seconds: float,
) -> None:
    """Save machine-readable results, comparison plot, and Korean explanation."""

    payload = {
        "protocol": "frozen validation winner evaluated once on held-out test data",
        "config": asdict(config),
        "validation": validation,
        "test": test,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
    }
    (output_dir / "final_test_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    metrics = ("rmse", "recall_at_10", "ndcg_at_10", "coverage")
    labels = ("RMSE ↓", "Recall@10 ↑", "NDCG@10 ↑", "Coverage ↑")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric, label in zip(axes.ravel(), metrics, labels):
        values = (validation[metric], test[metric])
        bars = axis.bar(("Validation", "Test"), values, color=("#4C78A8", "#F58518"))
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                      ha="center", va="bottom")
    figure.suptitle("Frozen Matrix Factorization: Validation vs Test")
    figure.tight_layout()
    figure.savefig(output_dir / "validation_vs_test.png", dpi=180)
    plt.close(figure)

    comparison = ""
    item_cf_path = output_dir.parent / "item_cf_10k" / "final_test_metrics.json"
    if item_cf_path.exists():
        item_cf = json.loads(item_cf_path.read_text(encoding="utf-8"))["test"]
        comparison = f"""
## 기존 ItemCF와 비교

동일한 reduced dataset과 test protocol에서 기존 ItemCF 결과도 함께 보면 모델의 성격 차이가 더 분명하다.

| Model | RMSE ↓ | Recall@10 ↑ | NDCG@10 ↑ | Coverage ↑ |
| --- | ---: | ---: | ---: | ---: |
| ItemCF | {float(item_cf['rmse']):.4f} | {float(item_cf['recall_at_10']):.4f} | {float(item_cf['ndcg_at_10']):.4f} | {float(item_cf['coverage']):.4f} |
| Matrix Factorization | **{test['rmse']:.4f}** | {test['recall_at_10']:.4f} | {test['ndcg_at_10']:.4f} | **{test['coverage']:.4f}** |

Matrix Factorization은 rating prediction error와 catalog coverage에서 강점이 있었고, ItemCF는 Recall@10과 NDCG@10에서 더 좋았다. 따라서 이 실험에서는 “평점을 얼마나 정확히 예측하는가”가 목적이면 Matrix Factorization이 유리하지만, “좋아할 게임을 Top-10 앞쪽에 배치하는가”가 목적이면 현재 ItemCF가 더 적합하다.
"""

    summary = f"""# Matrix Factorization 최종 test 결과

## 결론

Validation에서 선택한 hyperparameter를 변경하지 않고 held-out test에 정확히 한 번 평가했다. 따라서 아래 test metric은 tuning에 다시 사용하지 않은 최종 결과다.

| Split | RMSE ↓ | Recall@10 ↑ | NDCG@10 ↑ | Catalog coverage ↑ |
| --- | ---: | ---: | ---: | ---: |
| Validation | {validation['rmse']:.4f} | {validation['recall_at_10']:.4f} | {validation['ndcg_at_10']:.4f} | {validation['coverage']:.4f} |
| **Test (final)** | **{test['rmse']:.4f}** | **{test['recall_at_10']:.4f}** | **{test['ndcg_at_10']:.4f}** | **{test['coverage']:.4f}** |

## Frozen configuration

- Latent factors: `{config.factors}`
- Learning rate: `{config.learning_rate}`
- Regularization: `{config.regularization}`
- Epochs: `{config.epochs}`
- Batch size: `{config.batch_size}`
- Random seed: `{config.seed}`

## 결과를 쉽게 해석하면

- **RMSE**는 예측 rating이 실제 rating에서 평균적으로 얼마나 벗어나는지 보여준다.
- **Recall@10**은 사용자가 실제로 좋아한 게임 중 추천 Top-10이 찾아낸 비율이다.
- **NDCG@10**은 좋아한 게임을 단순히 찾는 것뿐 아니라 추천 목록 앞쪽에 배치했는지도 반영한다.
- **Coverage**는 여러 사용자에게 추천된 게임을 합쳤을 때 catalog를 얼마나 폭넓게 사용했는지 보여준다.

Validation과 test 값이 비슷하면 특정 validation sample에만 우연히 맞은 모델일 가능성이 낮다. 반대로 차이가 크면 overfitting이나 split 차이를 의심해야 한다.
{comparison}

## 실행 정보

- Test rating predictions: {int(test['rating_predictions']):,}
- Relevant item이 하나 이상인 test users: {int(test['ranking_users']):,}
- Fit time: {fit_seconds:.2f} seconds
- Test evaluation time: {evaluation_seconds:.2f} seconds
"""
    (output_dir / "final_test_summary.md").write_text(summary, encoding="utf-8")


def run_final_test(args: argparse.Namespace) -> dict[str, float]:
    """Fit the frozen winner on train and evaluate the untouched test split."""

    config, validation = load_frozen_config(args.best_config)
    shape = (args.total_users, args.total_items)
    train = load_interactions(args.train, shape=shape).ratings
    test = load_interactions(args.test, shape=shape).ratings
    relevant, test_users = validation_targets(test, args.relevant_threshold)
    if not relevant:
        raise ValueError("No test user has a relevant item")

    started = time.perf_counter()
    model = MatrixFactorization(config).fit(train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    metrics, _ = evaluate_model(model, test, relevant, test_users, args.top_k)
    evaluation_seconds = time.perf_counter() - started
    write_final_report(
        args.output_dir, config, validation, metrics,
        fit_seconds=fit_seconds, evaluation_seconds=evaluation_seconds,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/experiment_10k/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/experiment_10k/test.csv"))
    parser.add_argument("--best-config", type=Path,
                        default=Path("reports/matrix_factorization_10k/best_mf.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/matrix_factorization_10k"))
    parser.add_argument("--total-users", type=int, default=10_000)
    parser.add_argument("--total-items", type=int, default=1_500)
    parser.add_argument("--top-k", type=int, default=10, choices=(10,))
    parser.add_argument("--relevant-threshold", type=float, default=8.0)
    args = parser.parse_args()
    print(json.dumps(run_final_test(args), indent=2))


if __name__ == "__main__":
    main()
