from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict

import numpy as np
import pandas as pd


def _score_from_result(winner: str | None, perspective: str) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner == perspective else 0.0


@dataclass
class OpponentRecord:
    score: float = 0.0
    games: int = 0


@dataclass
class PlayerProfile:
    rating_sum: float = 0.0
    games_played: int = 0
    opponents: Dict[str, OpponentRecord] = field(default_factory=dict)
    recent_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    @property
    def mean_rating(self) -> float:
        if self.games_played == 0:
            return 1500.0
        return self.rating_sum / self.games_played

    @property
    def recent_mean(self) -> float:
        if not self.recent_scores:
            return 0.0
        return float(np.mean(self.recent_scores))


def build_player_profiles(
    games: pd.DataFrame,
    recent_window: int = 20,
) -> Dict[str, dict]:
    games = games.sort_values("created_at").reset_index(drop=True)
    profiles: Dict[str, PlayerProfile] = defaultdict(
        lambda: PlayerProfile(recent_scores=deque(maxlen=recent_window))
    )

    for _, row in games.iterrows():
        for color, opp_color in (("white", "black"), ("black", "white")):
            player = getattr(row, color)
            opponent = getattr(row, opp_color)
            rating = getattr(row, f"{color}_rating")
            score = _score_from_result(row.winner, color)

            profile = profiles[player]
            profile.rating_sum += rating
            profile.games_played += 1

            opp_record = profile.opponents.get(opponent)
            if opp_record is None:
                opp_record = OpponentRecord()
                profile.opponents[opponent] = opp_record
            opp_record.games += 1
            opp_record.score += score

            profile.recent_scores.append(score)

    serializable_profiles: Dict[str, dict] = {}
    for player, profile in profiles.items():
        serializable_profiles[player] = {
            "mean_rating": profile.mean_rating,
            "games_played": profile.games_played,
            "recent_mean": profile.recent_mean,
            "opponents": {
                opp: {"score": rec.score, "games": rec.games}
                for opp, rec in profile.opponents.items()
            },
        }

    return serializable_profiles


def compute_pair_features(player_a: str, player_b: str, profiles: Dict[str, dict]) -> Dict[str, float]:
    if player_a == player_b:
        raise ValueError("Choose two distinct players")
    if player_a not in profiles or player_b not in profiles:
        missing = [p for p in (player_a, player_b) if p not in profiles]
        raise KeyError(f"Missing profiles for: {', '.join(missing)}")

    prof_a = profiles[player_a]
    prof_b = profiles[player_b]

    opp_a = prof_a["opponents"]
    opp_b = prof_b["opponents"]

    common = set(opp_a.keys()) & set(opp_b.keys())
    common.discard(player_a)
    common.discard(player_b)

    if common:
        a_scores = [opp_a[o]["score"] / max(opp_a[o]["games"], 1) for o in common]
        b_scores = [opp_b[o]["score"] / max(opp_b[o]["games"], 1) for o in common]
        common_score_diff = float(np.mean(a_scores) - np.mean(b_scores))
        common_game_count = int(sum(min(opp_a[o]["games"], opp_b[o]["games"]) for o in common))
    else:
        common_score_diff = 0.0
        common_game_count = 0

    head_a = opp_a.get(player_b, {"score": 0.0, "games": 0})
    head_b = opp_b.get(player_a, {"score": 0.0, "games": 0})
    head_games = head_a["games"]
    if head_games > 0:
        head_score_diff = (
            head_a["score"] / max(head_a["games"], 1)
            - head_b["score"] / max(head_b["games"], 1)
        )
    else:
        head_score_diff = 0.0

    features = {
        "elo_diff": prof_a["mean_rating"] - prof_b["mean_rating"],
        "common_score_diff": common_score_diff,
        "common_game_count": common_game_count,
        "head_score_diff": head_score_diff,
        "head_games": head_games,
        "recent_form_diff": prof_a["recent_mean"] - prof_b["recent_mean"],
    }
    return features

