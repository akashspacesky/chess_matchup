from __future__ import annotations

import numpy as np
import streamlit as st

from src.chess_metric.data_fetch import fetch_user_games
from src.chess_metric.model_pipeline import (
    build_feature_rows,
    games_from_records,
    train_model,
)
from src.chess_metric.profiles import build_player_profiles, compute_pair_features


@st.cache_data(show_spinner=False)
def cached_games(username: str, max_games: int):
    return fetch_user_games(username, max_games=max_games)


def canonical_name(games_df, username: str) -> str:
    lower = username.lower()
    mask_white = games_df["white"].str.lower() == lower
    if mask_white.any():
        return games_df.loc[mask_white, "white"].iloc[0]
    mask_black = games_df["black"].str.lower() == lower
    if mask_black.any():
        return games_df.loc[mask_black, "black"].iloc[0]
    raise ValueError(f"{username} not found in downloaded games.")


def ensure_gm(games_df, player: str) -> None:
    rows = games_df[(games_df["white"] == player) | (games_df["black"] == player)]
    titles = set(rows.loc[rows["white"] == player, "white_title"].dropna().tolist())
    titles.update(rows.loc[rows["black"] == player, "black_title"].dropna().tolist())
    if "GM" not in titles:
        raise ValueError(f"{player} does not have a GM title on Lichess.")


def main():
    st.set_page_config(page_title="On-demand GM Matchup", page_icon="♟️")
    st.title("On-demand GM Matchup Probability")
    st.write(
        "Enter two Lichess usernames (both must be grandmasters). We'll fetch their recent "
        "rated games live, learn from their shared opponents, and display the matchup win chance."
    )

    with st.sidebar:
        st.header("Settings")
        max_games = st.slider("Games per player", min_value=50, max_value=400, value=200, step=50)
        st.caption("Games are fetched fresh from Lichess each run (rated classical/rapid/blitz).")
        if st.button("Clear cached downloads"):
            cached_games.clear()
            st.success("Cleared cached Lichess responses.")

    col1, col2 = st.columns(2)
    default_a = "AnishGiri"
    default_b = "fabichess"
    player_a_input = col1.text_input("Player A (White username)", value=default_a).strip()
    player_b_input = col2.text_input("Player B (Black username)", value=default_b).strip()

    if not player_a_input or not player_b_input:
        st.stop()

    if player_a_input.lower() == player_b_input.lower():
        st.info("Pick two different players.")
        st.stop()

    if st.button("Compare match-up", type="primary"):
        player_a_key = player_a_input.lower()
        player_b_key = player_b_input.lower()
        with st.spinner("Fetching and processing games..."):
            try:
                games_a = cached_games(player_a_key, max_games)
                games_b = cached_games(player_b_key, max_games)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to download games: {exc}")
                st.stop()

            combined = games_a + games_b
            games_df = games_from_records(combined)
            if games_df.empty:
                st.error("No standard rated games found for the chosen players.")
                st.stop()

            targets = {player_a_key, player_b_key}
            mask = games_df["white"].str.lower().isin(targets) | games_df["black"].str.lower().isin(targets)
            games_df = games_df[mask].copy()

            if games_df.empty:
                st.error("Fetched games did not include the selected players. Double-check usernames.")
                st.stop()

            player_a = canonical_name(games_df, player_a_input)
            player_b = canonical_name(games_df, player_b_input)

            try:
                ensure_gm(games_df, player_a)
                ensure_gm(games_df, player_b)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            decisive_mask = games_df["winner"].isin(["white", "black"])
            if decisive_mask.sum() < 5:
                st.error("Need more decisive games to learn anything meaningful. Increase the fetch window.")
                st.stop()

            feature_rows = build_feature_rows(games_df)
            if not feature_rows:
                st.error("Not enough matchup history to build features.")
                st.stop()

            try:
                model, metrics, feature_cols, _ = train_model(feature_rows)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            profiles = build_player_profiles(games_df)
            try:
                features = compute_pair_features(player_a, player_b, profiles)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not compute matchup features: {exc}")
                st.stop()

            feature_vector = np.array([[features[col] for col in feature_cols]])
            probability = float(model.predict_proba(feature_vector)[0][1])
            elo_baseline = 1 / (1 + 10 ** (-features["elo_diff"] / 400))

        st.success("Done!")
        st.subheader("Prediction")
        st.metric(
            label=f"P({player_a} wins vs {player_b})",
            value=f"{probability*100:.1f}%",
            delta=f"{(probability-elo_baseline)*100:.1f} pts vs Elo",
        )
        st.caption(f"Elo baseline win chance: {elo_baseline*100:.1f}%")

        st.subheader("Feature snapshot")
        st.dataframe(
            {
                "feature": list(features.keys()),
                "value": [round(v, 4) for v in features.values()],
            }
        )

        st.subheader("Model diagnostics (training-set)")
        st.write(
            {
                "samples": metrics["samples"],
                "accuracy": round(metrics["accuracy"], 3),
                "roc_auc": round(metrics["roc_auc"], 3),
                "log_loss": round(metrics["log_loss"], 3),
                "brier": round(metrics["brier"], 3),
            }
        )
        st.caption(
            "Metrics are computed on the fetched games only (no hold-out set). "
            "Fetching more history generally improves stability."
        )


if __name__ == "__main__":
    main()

