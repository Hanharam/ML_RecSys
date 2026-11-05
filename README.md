# Board Game Recommendation System

This machine learning project aims to build a personalized board game recommendation model using user ratings from BoardGameGeek (BGG).

## Dataset Selection

This project uses the **2022 v3 snapshot** of `jvanelteren/boardgamegeek-reviews`.

- Approximately 19 million user–game ratings
- Approximately 410,000 users and 22,000 games
- Metadata including player count, playing time, complexity, categories, and mechanics
- Enables comparison of collaborative filtering and content-based recommendation using the same dataset

The latest v4 release is approximately 5.4 GB in total, which is unnecessarily large for this assignment. The official BGG ranking CSV contains only aggregated scores for each game and does not provide the user–game interactions required for personalized recommendations. Therefore, v3 is used to provide a suitable balance between dataset size and learning objectives.

The original data comes from the [BoardGameGeek XML API](https://boardgamegeek.com/wiki/page/XML%20API), and the redistributed dataset is available through [Kaggle BoardGameGeek Reviews](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews). The BGG API terms of use should be reviewed before using the data.

## Project Structure

```text
data/raw/        Original ZIP and CSV files (excluded from Git)
data/processed/  Cleaned and split datasets (excluded from Git)
notebooks/       Exploratory analysis and model experiments
reports/         Assignment reports and figures
src/             Download and preprocessing code
tests/           Preprocessing regression tests
```

## Usage

Python 3.10 or later is required.

```bash
# 1. Create a virtual environment and install visualization dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Download the public dataset (approximately 1 GB)
python3 -m src.download_data

# 3. Extract the archive
python3 -m zipfile -e data/raw/boardgamegeek-reviews-v3.zip data/raw

# 4. Preprocess: select the 3,000 most-rated games with metadata
#    and at least 100 ratings
.venv/bin/python -m src.preprocess

# 5. Run the popularity-based baseline
.venv/bin/python -m src.popularity

# 6. Generate EDA visualizations
.venv/bin/python -m src.eda

# 7. Run tests
.venv/bin/python -m unittest discover -s tests -v
```

The preprocessing pipeline reads the CSV files sequentially to avoid excessive memory usage. It retains only valid BGG ratings in the 1–10 range and selects the 3,000 most-rated games that have metadata and at least 100 ratings. Users with fewer than 10 ratings among the selected games are then removed, and the remaining user and game identifiers are encoded as integers.

The resulting data is divided into training, validation, and test sets using a deterministic hash of each `(user, item)` pair, producing a split of approximately 80/10/10.

Generated files:

- `train.csv`, `validation.csv`, and `test.csv`: `user_id,item_id,rating`
- `users.csv` and `items.csv`: mappings between original and encoded integer IDs
- `games.csv`: game metadata used for recommendation explanations and content-based models
- `preprocessing_report.json`: row counts, filtering results, and split statistics

EDA results are saved under `reports/eda/`, while popularity-based baseline results are saved under `reports/baseline/`. The popularity score is calculated as a Bayesian weighted rating using only the training data. No validation or test information is used.
