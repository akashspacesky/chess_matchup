# Chess Matchup Probability

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://akashspacesky-chess-matchup-app-wsyf1d.streamlit.app/)

**Live Demo:** [akashspacesky-chess-matchup-app-wsyf1d.streamlit.app](https://akashspacesky-chess-matchup-app-wsyf1d.streamlit.app/)

Interactive Streamlit app that estimates the win probability between two Lichess grandmasters based on how they perform against shared opponents, head-to-head history, and recent form. Enter any two GM usernames; the app fetches their latest rated games directly from Lichess, engineers matchup features, and trains a lightweight logistic model on the fly.

## Features
- Live data fetch from the public Lichess API (no API key required).
- Built-in GM search: type a prefix, hit “Search GM suggestions,” and pick from verified GM handles fetched from Lichess autocomplete (with live title validation).
- Common-opponent, head-to-head, and recency features to capture “relative wins”.
- Rich matchup dashboard: shared opponents, opponent-rating buckets (with charts), head-to-head log, and recent-form streaks.
- Clear cached downloads button for cases when you want to refresh or test another user.
- Optional CLI utilities for downloading reusable datasets and training offline.

## Quick start
```bash
git clone https://github.com/<you>/chess-matchup-probability.git
cd chess-matchup-probability
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`, type two GM usernames (e.g., `AnishGiri` vs `fabichess`), adjust the “Games per player” slider if desired, and hit **Compare match-up**.

## Deploying on Streamlit Cloud
1. Push this repo to GitHub (public).
2. In [Streamlit Community Cloud](https://share.streamlit.io/), click **New app**, select the repo/branch, set the main file to `app.py`.
3. Use the default build command (`pip install -r requirements.txt`). That’s it—the app will be reachable at `https://<your-app>.streamlit.app`.

## Optional CLI utilities
- `python -m src.chess_metric.data_fetch` – download batches of games for a curated user list into `data/raw/lichess_sample.ndjson`.
- `python -m src.chess_metric.model_pipeline --gm-only` – train and print evaluation metrics against the Elo baseline using the cached dataset.

These scripts are handy if you prefer working with a fixed snapshot, but the Streamlit app does not depend on them.

## How it works
1. Fetches rated blitz/rapid/classical games for both players from Lichess.
2. Builds player profiles (head-to-head stats, common opponents, recent scores).
3. Trains a logistic regression using matchup features:
   - `elo_diff`
   - `common_score_diff` and `common_game_count`
   - `head_score_diff` and `head_games`
   - `recent_form_diff`
4. Predicts `P(playerA wins | playerB)` and compares it to the standard Elo expectation.

## Contributions & Licensing
- Issues and pull requests are welcome—ideas include richer features, more model choices, or alternative visualizations.
- Add your preferred license (e.g., MIT/Apache-2.0) before publishing publicly.

