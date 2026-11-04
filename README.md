# 보드게임 추천 시스템 과제

BoardGameGeek(BGG)의 사용자 평점으로 개인화 보드게임 추천 모델을 만드는 머신러닝 과제 프로젝트입니다.

## 데이터셋 선택

`jvanelteren/boardgamegeek-reviews`의 **2022년 v3 스냅샷**을 사용합니다.

- 약 1,900만 개의 사용자-게임 평점
- 약 41만 명의 사용자와 2.2만 개의 게임
- 게임별 인원, 시간, 난이도, 카테고리, 메커니즘 메타데이터 포함
- 협업 필터링과 콘텐츠 기반 추천을 같은 데이터에서 비교 가능

최신 v4는 전체 용량이 약 5.4GB로 과제에 비해 크고, BGG 공식 rank CSV는 게임별 집계 점수만 제공해 개인화 추천에 필요한 사용자-게임 상호작용이 없습니다. 따라서 규모와 학습 목적의 균형이 좋은 v3를 고정 사용합니다.

원 출처는 [BoardGameGeek XML API](https://boardgamegeek.com/wiki/page/XML%20API)이며, 재배포 데이터는 [Kaggle BoardGameGeek Reviews](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews)입니다. 데이터 사용 시 BGG API 이용 조건을 확인해야 합니다.

## 폴더 구조

```text
data/raw/        원본 ZIP/CSV (Git 제외)
data/processed/  정제·분할 결과 (Git 제외)
notebooks/       탐색 및 모델 실험
reports/         과제 보고서/그림
src/             다운로드·전처리 코드
tests/           전처리 회귀 테스트
```

## 실행

Python 3.10 이상이 필요합니다.

```bash
# 1. 가상환경과 그래프 라이브러리 설치
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 공개 데이터 다운로드 (약 1GB)
python3 -m src.download_data

# 3. 압축 해제
python3 -m zipfile -e data/raw/boardgamegeek-reviews-v3.zip data/raw

# 4. 전처리: 메타데이터가 있는 게임 중 평점 100개 이상, 상위 3,000개
.venv/bin/python -m src.preprocess

# 5. 인기순 baseline
.venv/bin/python -m src.popularity

# 6. EDA 그래프
.venv/bin/python -m src.eda

# 7. 테스트
.venv/bin/python -m unittest discover -s tests -v
```

전처리는 메모리 폭증을 피하려고 CSV를 순차적으로 읽습니다. BGG의 유효 평점 범위인 1–10만 남기고, 메타데이터가 있으며 평점이 100개 이상인 게임 중 평가 수 상위 3,000개를 선택합니다. 선택된 게임 안에서 평점 10개 미만 사용자를 제거한 뒤 정수 ID로 인코딩합니다. 결과는 `(user, item)` 기반 고정 해시로 train/validation/test = 80/10/10에 가깝게 나눕니다.

생성 파일:

- `train.csv`, `validation.csv`, `test.csv`: `user_id,item_id,rating`
- `users.csv`, `items.csv`: 원본 ID와 정수 ID 매핑
- `games.csv`: 추천 설명 및 콘텐츠 모델용 게임 메타데이터
- `preprocessing_report.json`: 행 수, 필터 결과, 분할 통계

EDA 결과는 `reports/eda/`, 인기순 baseline 결과는 `reports/baseline/`에 저장됩니다. 인기 점수는 training 데이터만 사용한 Bayesian weighted rating이며, validation/test 정보는 사용하지 않습니다.

## 다음 모델링 순서

1. 인기순 추천을 baseline으로 둡니다.
2. Surprise SVD 또는 implicit ALS로 협업 필터링을 구현합니다.
3. 게임의 카테고리·메커니즘·난이도로 콘텐츠 기반 추천을 구현합니다.
4. RMSE뿐 아니라 Recall@10, NDCG@10, coverage로 비교합니다.

