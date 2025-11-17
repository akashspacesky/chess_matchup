from __future__ import annotations

import json
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def _score_from_result(winner: str | None, perspective: str) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner == perspective else 0.0


def _iter_game_records(games: Sequence[dict]) -> Iterable[dict]:
    for game in games:
        if not game:
            continue
        if game.get("variant") != "standard":
            continue
        white = game.get("players", {}).get("white", {}).get("user", {})
        black = game.get("players", {}).get("black", {}).get("user", {})
        if not white or not black:
            continue
        white_rating = game.get("players", {}).get("white", {}).get("rating")
        black_rating = game.get("players", {}).get("black", {}).get("rating")
        if white_rating is None or black_rating is None:
            continue
        yield {
            "game_id": game.get("id"),
            "created_at": pd.to_datetime(game.get("createdAt"), unit="ms"),
            "speed": game.get("speed"),
            "white": white.get("name"),
            "black": black.get("name"),
            "white_title": white.get("title"),
            "black_title": black.get("title"),
            "white_rating": white_rating,
            "black_rating": black_rating,
            "winner": game.get("winner"),  # "white", "black", or None for draw
            "status": game.get("status"),
        }


def games_from_records(
    games: Sequence[dict],
    required_titles: Optional[Set[str]] = None,
) -> pd.DataFrame:
    records = list(_iter_game_records(games))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    if required_titles:
        df = df[
            df["white_title"].isin(required_titles) & df["black_title"].isin(required_titles)
        ]
    if df.empty:
        return df
    df = df.drop_duplicates(subset="game_id")
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def load_games(ndjson_path: Path, required_titles: Optional[Set[str]] = None) -> pd.DataFrame:
    records: List[dict] = []
    with ndjson_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            game = json.loads(line)
            records.append(game)
    return games_from_records(records, required_titles=required_titles)


@dataclass
class FeatureRow:
    white: str
    black: str
    created_at: pd.Timestamp
    features: Dict[str, float]
    result: int
    elo_expectation: float


def build_feature_rows(games: pd.DataFrame) -> List[FeatureRow]:
    opponent_stats: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"score": 0.0, "games": 0})
    )
    recent_form: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=20))

    feature_rows: List[FeatureRow] = []

    for _, row in games.iterrows():
        white = row.white
        black = row.black

        if white == black:
            continue

        # Snapshot stats before current game
        white_opponents = opponent_stats[white]
        black_opponents = opponent_stats[black]

        common = set(white_opponents.keys()) & set(black_opponents.keys())
        if common:
            white_common_scores = [white_opponents[o]["score"] / white_opponents[o]["games"] for o in common]
            black_common_scores = [black_opponents[o]["score"] / black_opponents[o]["games"] for o in common]
            common_score_diff = float(np.mean(white_common_scores) - np.mean(black_common_scores))
            common_game_count = int(
                sum(min(white_opponents[o]["games"], black_opponents[o]["games"]) for o in common)
            )
        else:
            common_score_diff = 0.0
            common_game_count = 0

        head_to_head_white = white_opponents.get(black, {"score": 0.0, "games": 0})
        head_to_head_black = black_opponents.get(white, {"score": 0.0, "games": 0})
        head_score_diff = 0.0
        head_games = 0
        if head_to_head_white["games"] > 0 or head_to_head_black["games"] > 0:
            head_score_diff = (
                head_to_head_white["score"] / max(head_to_head_white["games"], 1)
                - head_to_head_black["score"] / max(head_to_head_black["games"], 1)
            )
            head_games = head_to_head_white["games"] + head_to_head_black["games"]

        white_form = recent_form[white]
        black_form = recent_form[black]
        white_recent = float(np.mean(white_form)) if white_form else 0.0
        black_recent = float(np.mean(black_form)) if black_form else 0.0

        elo_diff = row.white_rating - row.black_rating
        elo_expectation = 1 / (1 + 10 ** (-elo_diff / 400))

        result = None
        if row.winner == "white":
            result = 1
        elif row.winner == "black":
            result = 0

        if result is not None:
            feature_rows.append(
                FeatureRow(
                    white=white,
                    black=black,
                    created_at=row.created_at,
                    features={
                        "elo_diff": elo_diff,
                        "common_score_diff": common_score_diff,
                        "common_game_count": common_game_count,
                        "head_score_diff": head_score_diff,
                        "head_games": head_games,
                        "recent_form_diff": white_recent - black_recent,
                    },
                    result=result,
                    elo_expectation=elo_expectation,
                )
            )

        # Update stats with current result (including draws)
        white_score = _score_from_result(row.winner, "white")
        black_score = _score_from_result(row.winner, "black")

        opponent_stats[white][black]["games"] += 1
        opponent_stats[white][black]["score"] += white_score

        opponent_stats[black][white]["games"] += 1
        opponent_stats[black][white]["score"] += black_score

        recent_form[white].append(white_score)
        recent_form[black].append(black_score)

    return feature_rows


def train_model(feature_rows: Iterable[FeatureRow]):
    df = pd.DataFrame(
        [
            {
                **row.features,
                "result": row.result,
                "elo_expectation": row.elo_expectation,
                "created_at": row.created_at,
                "white": row.white,
                "black": row.black,
            }
            for row in feature_rows
        ]
    )
    feature_cols = [
        "elo_diff",
        "common_score_diff",
        "common_game_count",
        "head_score_diff",
        "head_games",
        "recent_form_diff",
    ]
    df = df.dropna(subset=feature_cols)
    if df.empty:
        raise ValueError("No decisive games available to train the model.")
    if df["result"].nunique() < 2:
        raise ValueError("Need at least one win and one loss to train the classifier.")
    model = LogisticRegression(max_iter=500)
    model.fit(df[feature_cols].values, df["result"].values)
    preds = model.predict_proba(df[feature_cols].values)[:, 1]
    baseline = df["elo_expectation"].values
    metrics = {
        "samples": int(len(df)),
        "accuracy": float(accuracy_score(df["result"].values, preds >= 0.5)),
        "roc_auc": float(roc_auc_score(df["result"].values, preds)),
        "log_loss": float(log_loss(df["result"].values, preds)),
        "brier": float(brier_score_loss(df["result"].values, preds)),
        "baseline_log_loss": float(log_loss(df["result"].values, baseline)),
        "baseline_brier": float(brier_score_loss(df["result"].values, baseline)),
    }
    return model, metrics, feature_cols, df


def run_pipeline(ndjson_path: Path, required_titles: Optional[Set[str]] = None):
    games = load_games(ndjson_path, required_titles=required_titles)
    feature_rows = build_feature_rows(games)
    model, metrics, feature_cols, df = train_model(feature_rows)
    print("Model metrics vs Elo baseline:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    return {
        "model": model,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "training_frame": df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train matchup-aware chess model")
    parser.add_argument(
        "--ndjson",
        type=Path,
        default=Path("data/raw/lichess_sample.ndjson"),
        help="Path to NDJSON games file",
    )
    parser.add_argument(
        "--gm-only",
        action="store_true",
        help="Restrict to games where both players have GM titles",
    )
    args = parser.parse_args()
    titles = {"GM"} if args.gm_only else None
    run_pipeline(args.ndjson, required_titles=titles)

