# Chess Win Probability MVP

This MVP builds a matchup-aware win-probability model using Lichess data. Instead of relying solely on Elo, it incorporates how players performed versus shared opponents, recent form, and direct head-to-head history.

## Project layout

- `requirements.txt` – Python dependencies.
- `src/chess_metric/data_fetch.py` – pulls small batches of rated games from the public Lichess API.
- `src/chess_metric/model_pipeline.py` – loads the dataset, engineers matchup features, trains a logistic-regression model, and prints evaluation metrics against the baseline Elo expectation.
- `data/raw/lichess_sample.ndjson` – cached sample games (generated locally).

## Getting started

```bash
cd /Users/akashspacesky/Documents/chess_ws
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Launch the on-demand GUI

```bash
source .venv/bin/activate
streamlit run app.py
```

Enter two Lichess grandmasters (e.g., `AnishGiri` vs `fabichess`). The app will fetch their recent rated games directly from Lichess, learn matchup features on the fly, and show the predicted win probability alongside the Elo baseline and feature snapshot.

### 2. (Optional) Download a reusable GM dataset

```bash
source .venv/bin/activate
python -m src.chess_metric.data_fetch
```

Edit `TARGET_USERS` in `data_fetch.py` to control who gets cached locally.

### 3. (Optional) Train and evaluate from a cached dataset

```bash
source .venv/bin/activate
python -m src.chess_metric.model_pipeline --gm-only
```

This prints accuracy, ROC-AUC, log-loss, and Brier scores for both the learned model and the Elo baseline using the cached NDJSON file. Drop `--gm-only` if you want every rated game in the file.

## Next ideas

- Persist trained weights and scalers so the model can serve predictions via CLI or API.
- Swap logistic regression for gradient boosting or Bayesian Bradley–Terry variations.
- Enrich features with clock information, opening families, or board-evaluation deltas.
- Pull significantly larger datasets (e.g., monthly Lichess dumps) and add recency decay.

