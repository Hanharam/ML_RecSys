"""Tune Matrix Factorization on validation data and save reproducible results."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_interactions, sample_user_ids
from src.matrix_factorization import MatrixFactorization, MatrixFactorizationConfig
from src.metrics import catalog_coverage, ndcg_at_k, recall_at_k, rmse, validate_recommendations


def build_trial_configs(trials: int, seed: int) -> list[MatrixFactorizationConfig]:
    """Create deterministic random-search trials with one stable reference."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    reference = MatrixFactorizationConfig(
        factors=32, learning_rate=0.01, regularization=0.05,
        epochs=12, batch_size=4096, seed=seed,
    )
    configs = [reference]
    seen = {tuple(asdict(reference).items())}
    rng = np.random.default_rng(seed)
    while len(configs) < trials:
        config = MatrixFactorizationConfig(
            factors=int(rng.choice((16, 32, 64))),
            learning_rate=float(rng.choice((0.005, 0.01, 0.02))),
            regularization=float(rng.choice((0.01, 0.05, 0.10))),
            epochs=int(rng.choice((8, 12, 18))),
            batch_size=4096,
            seed=seed,
        )
        key = tuple(asdict(config).items())
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def validation_targets(validation, relevant_threshold: float) -> tuple[dict[int, set[int]], list[int]]:
    """Return liked validation games and users with validation ratings."""

    relevant: dict[int, set[int]] = {}
    users: list[int] = []
    for user_id in np.flatnonzero(validation.getnnz(axis=1)):
        row = validation.getrow(int(user_id))
        users.append(int(user_id))
        liked = row.indices[row.data >= relevant_threshold]
        if liked.size:
            relevant[int(user_id)] = set(map(int, liked))
    return relevant, users


def evaluate_model(
    model: MatrixFactorization,
    validation,
    relevant: dict[int, set[int]],
    validation_users: list[int],
    top_k: int,
) -> tuple[dict[str, float], dict[int, list[int]]]:
    """Measure rating error and Top-K quality under the same protocol as ItemCF."""

    train = model.train_
    if train is None:
        raise RuntimeError("model must be fitted before evaluation")
    held_out = validation.tocoo()
    predicted = model.predict_pairs(held_out.row, held_out.col)
    recommendations: dict[int, list[int]] = {}
    item_ids = np.arange(validation.shape[1])
    for user_id in relevant:
        scores = model.score_user(user_id)
        scores[train.getrow(user_id).indices] = -np.inf
        finite = np.flatnonzero(np.isfinite(scores))
        if finite.size <= top_k:
            candidates = finite
        else:
            candidates = finite[np.argpartition(scores[finite], -top_k)[-top_k:]]
        order = np.lexsort((item_ids[candidates], -scores[candidates]))
        recommendations[user_id] = list(map(int, candidates[order]))

    validate_recommendations(
        recommendations,
        {user_id: set(map(int, train.getrow(user_id).indices)) for user_id in recommendations},
        k=top_k,
    )
    metrics = {
        "rmse": rmse(held_out.data, predicted),
        "recall_at_10": recall_at_k(recommendations, relevant, top_k),
        "ndcg_at_10": ndcg_at_k(recommendations, relevant, top_k),
        "coverage": catalog_coverage(recommendations, range(validation.shape[1]), k=top_k),
        "rating_predictions": float(held_out.nnz),
        "ranking_users": float(len(relevant)),
    }
    return metrics, recommendations


def selection_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    """Prefer ranking quality, then recall, lower error, and coverage."""

    return (
        float(row["ndcg_at_10"]), float(row["recall_at_10"]),
        -float(row["rmse"]), float(row["coverage"]),
    )


def write_results(rows: list[dict[str, object]], output_dir: Path, *, full_validation: bool) -> None:
    """Write trial data, frozen winner, plots, and an accessible Korean report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mf_trials.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=selection_key)
    best_rmse = min(rows, key=lambda row: float(row["rmse"]))
    (output_dir / "best_mf.json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8"
    )

    labels = [f"T{row['trial']}" for row in rows]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(3, 1, figsize=(max(10, len(rows)), 10), sharex=True)
    for axis, metric, label, color in (
        (axes[0], "recall_at_10", "Recall@10", "#4C78A8"),
        (axes[1], "ndcg_at_10", "NDCG@10", "#F58518"),
        (axes[2], "coverage", "Catalog coverage", "#54A24B"),
    ):
        values = [float(row[metric]) for row in rows]
        axis.bar(positions, values, color=color)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        for x, value in zip(positions, values):
            axis.text(x, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_title("Matrix Factorization Hyperparameter Search")
    axes[2].set_xticks(positions, labels)
    figure.tight_layout()
    figure.savefig(output_dir / "ranking_metrics.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for row in rows:
        axis.scatter(float(row["rmse"]), float(row["ndcg_at_10"]), s=65)
        axis.annotate(f"T{row['trial']}", (float(row["rmse"]), float(row["ndcg_at_10"])),
                      xytext=(4, 4), textcoords="offset points")
    axis.set_title("Rating Error vs. Ranking Quality")
    axis.set_xlabel("RMSE (lower is better)")
    axis.set_ylabel("NDCG@10 (higher is better)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "rmse_vs_ndcg.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        [float(row["learning_rate"]) for row in rows],
        [float(row["ndcg_at_10"]) for row in rows],
        s=[40 + float(row["factors"]) * 2 for row in rows],
        c=[float(row["regularization"]) for row in rows],
        cmap="viridis", edgecolor="black", linewidth=0.5,
    )
    for row in rows:
        axis.annotate(
            f"T{row['trial']} ({row['epochs']} epochs)",
            (float(row["learning_rate"]), float(row["ndcg_at_10"])),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Learning rate")
    axis.set_ylabel("Validation NDCG@10")
    axis.set_title("Hyperparameter Effects: marker size = factors")
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="Regularization")
    figure.tight_layout()
    figure.savefig(output_dir / "hyperparameter_effects.png", dpi=180)
    plt.close(figure)

    scope = "10,000명 전체 reduced dataset" if full_validation else "재현 가능한 user sample"
    summary = f"""# Matrix Factorization validation 결과

## 한눈에 보는 결론

{scope}에서 총 {len(rows)}개 hyperparameter 조합을 비교했다. 최종 모델은 **validation NDCG@10**이 가장 높은 조합으로 선택했으며, 이 과정에서 test data는 읽지 않았다.

- Best trial: `T{best['trial']}`
- Latent factors: `{best['factors']}`
- Learning rate: `{best['learning_rate']}`
- Regularization: `{best['regularization']}`
- Epochs: `{best['epochs']}`
- RMSE: **{float(best['rmse']):.4f}**
- Recall@10: **{float(best['recall_at_10']):.4f}**
- NDCG@10: **{float(best['ndcg_at_10']):.4f}**
- Catalog coverage: **{float(best['coverage']):.4f}**

## 무엇을 구현했나

Matrix Factorization은 큰 user-game rating matrix를 두 개의 작은 embedding matrix로 나눈다. 각 user와 game은 `{best['factors']}`차원의 latent vector를 가지며, 두 vector의 dot product가 취향의 궁합을 나타낸다. 여기에 전체 평균, user bias, game bias를 더해 rating을 예측한다.

`predicted rating = global mean + user bias + game bias + user embedding · game embedding`

user bias는 평소 점수를 높거나 낮게 주는 성향을, game bias는 전반적으로 평가가 좋은지 나쁜지를 담당한다. embedding은 그 두 효과만으로 설명되지 않는 취향 pattern을 학습한다.

## Hyperparameter를 어떻게 선택했나

- `latent factors`: 취향을 몇 개의 숨은 축으로 표현할지 결정한다. 너무 작으면 표현력이 부족하고, 너무 크면 overfitting과 계산량이 증가할 수 있다.
- `learning rate`: 한 번의 update 크기다. 너무 작으면 학습이 느리고, 너무 크면 불안정해질 수 있다.
- `regularization`: embedding과 bias가 지나치게 커지는 것을 막아 overfitting을 줄인다.
- `epochs`: 전체 training interaction을 반복해서 학습하는 횟수다.

선택 기준은 Top-10 추천 순서를 반영하는 NDCG@10이다. RMSE가 낮아도 사용자가 좋아할 게임을 상위에 배치하지 못할 수 있으므로 RMSE만으로 모델을 고르지 않았다. NDCG@10이 같으면 Recall@10, RMSE, coverage 순서로 비교했다.

실제로 가장 낮은 RMSE는 `T{best_rmse['trial']}`의 **{float(best_rmse['rmse']):.4f}**였지만, 이 trial의 NDCG@10은 **{float(best_rmse['ndcg_at_10']):.4f}**로 `T{best['trial']}`보다 낮았다. 반대로 선택된 `T{best['trial']}`는 RMSE가 더 높지만 NDCG@10과 coverage가 높았다. 즉, rating 값을 정확히 맞히는 설정과 좋은 추천 순서를 만드는 설정이 같지 않을 수 있다는 결과다.

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
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> list[dict[str, object]]:
    if not 0 < args.sample_users <= args.total_users:
        raise ValueError("sample_users must be between 1 and total_users")
    users = sample_user_ids(args.total_users, args.sample_users, args.seed)
    shape = (args.total_users, args.total_items)
    print(f"Loading {args.sample_users:,} users...")
    train = load_interactions(args.train, shape=shape, user_filter=users).ratings
    validation = load_interactions(args.validation, shape=shape, user_filter=users).ratings
    relevant, validation_users = validation_targets(validation, args.relevant_threshold)
    if not relevant:
        raise ValueError("No validation user has a relevant item; increase sample size")

    rows: list[dict[str, object]] = []
    for trial, config in enumerate(build_trial_configs(args.trials, args.seed)):
        print(f"Trial {trial}: {config}")
        started = time.perf_counter()
        model = MatrixFactorization(config).fit(train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        metrics, _ = evaluate_model(model, validation, relevant, validation_users, args.top_k)
        rows.append({
            "trial": trial, **asdict(config), **metrics,
            "train_rmse": model.training_rmse_[-1],
            "fit_seconds": fit_seconds,
            "evaluation_seconds": time.perf_counter() - started,
        })
    write_results(rows, args.output_dir, full_validation=args.sample_users == args.total_users)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/experiment_10k/train.csv"))
    parser.add_argument("--validation", type=Path, default=Path("data/experiment_10k/validation.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/matrix_factorization_10k"))
    parser.add_argument("--total-users", type=int, default=10_000)
    parser.add_argument("--total-items", type=int, default=1_500)
    parser.add_argument("--sample-users", type=int, default=10_000)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10, choices=(10,))
    parser.add_argument("--relevant-threshold", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = run_experiment(args)
    print(json.dumps(max(rows, key=selection_key), indent=2))


if __name__ == "__main__":
    main()
