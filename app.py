from __future__ import annotations

from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.chess_metric.data_fetch import fetch_user_games
from src.chess_metric.model_pipeline import (
    build_feature_rows,
    games_from_records,
    train_model,
)
from src.chess_metric.profiles import build_player_profiles, compute_pair_features

AUTOCOMPLETE_ENDPOINT = "https://lichess.org/api/player/autocomplete"
USER_ENDPOINT_TEMPLATE = "https://lichess.org/api/user/{username}"


@st.cache_data(show_spinner=False)
def cached_games(username: str, max_games: int):
    return fetch_user_games(username, max_games=max_games)


@st.cache_data(show_spinner=False)
def autocomplete_usernames(term: str, limit: int = 20) -> List[str]:
    resp = requests.get(
        AUTOCOMPLETE_ENDPOINT,
        params={"term": term},
        timeout=15,
    )
    resp.raise_for_status()
    suggestions = resp.json()
    return suggestions[:limit]


@st.cache_data(show_spinner=False)
def fetch_user_profile(username: str) -> dict:
    resp = requests.get(
        USER_ENDPOINT_TEMPLATE.format(username=username),
        headers={"Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_gm_suggestions(term: str, max_results: int = 10) -> List[str]:
    term = term.strip()
    if len(term) < 2:
        return []
    candidates = autocomplete_usernames(term, limit=20)
    gm_names: List[str] = []
    for name in candidates:
        try:
            profile = fetch_user_profile(name)
        except Exception:
            continue
        if profile.get("title") == "GM":
            gm_names.append(profile.get("username", name))
        if len(gm_names) >= max_results:
            break
    # Deduplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for name in gm_names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(name)
    return ordered


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


def _score_from_result(winner: str | None, perspective: str) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner == perspective else 0.0


def shared_opponent_rows(
    player_a: str,
    player_b: str,
    profiles: Dict[str, dict],
    top_n: int = 8,
) -> List[dict]:
    opp_a = profiles[player_a]["opponents"]
    opp_b = profiles[player_b]["opponents"]
    common = set(opp_a) & set(opp_b)
    rows: List[dict] = []
    for opp in common:
        rec_a = opp_a[opp]
        rec_b = opp_b[opp]
        if rec_a["games"] == 0 or rec_b["games"] == 0:
            continue
        opponent_rating = profiles.get(opp, {}).get("mean_rating")
        rows.append(
            {
                "opponent": opp,
                f"{player_a} score": rec_a["score"] / rec_a["games"],
                f"{player_b} score": rec_b["score"] / rec_b["games"],
                "shared_games": min(rec_a["games"], rec_b["games"]),
                "opp_rating": opponent_rating,
            }
        )
    rows.sort(key=lambda r: (r["shared_games"], r["opp_rating"] or 0), reverse=True)
    return rows[:top_n]


RATING_BUCKETS: List[Tuple[int, int, str]] = [
    (0, 2400, "<2400"),
    (2400, 2500, "2400-2499"),
    (2500, 2600, "2500-2599"),
    (2600, 2700, "2600-2699"),
    (2700, 3000, "2700+"),
]


def _bucket_label(rating: float | int | None) -> str:
    if rating is None:
        return "unknown"
    for low, high, label in RATING_BUCKETS:
        if low <= rating < high:
            return label
    return "unknown"


def bucket_performance(games_df: pd.DataFrame, player: str) -> pd.DataFrame:
    mask = (games_df["white"] == player) | (games_df["black"] == player)
    df = games_df[mask].sort_values("created_at")
    if df.empty:
        return pd.DataFrame()
    stats: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        if row.white == player:
            perspective = "white"
            opponent_rating = row.black_rating
        else:
            perspective = "black"
            opponent_rating = row.white_rating
        label = _bucket_label(opponent_rating)
        bucket = stats.setdefault(label, {"games": 0, "wins": 0, "draws": 0, "losses": 0})
        bucket["games"] += 1
        score = _score_from_result(row.winner, perspective)
        if score == 1.0:
            bucket["wins"] += 1
        elif score == 0.5:
            bucket["draws"] += 1
        else:
            bucket["losses"] += 1
    rows = []
    for label, values in stats.items():
        games = values["games"]
        if games == 0:
            continue
        rows.append(
            {
                "bucket": label,
                "games": games,
                "win%": round(values["wins"] / games * 100, 1),
                "draw%": round(values["draws"] / games * 100, 1),
                "loss%": round(values["losses"] / games * 100, 1),
            }
        )
    rows.sort(key=lambda r: r["bucket"])
    return pd.DataFrame(rows)


def head_to_head_history(games_df: pd.DataFrame, player_a: str, player_b: str) -> pd.DataFrame:
    mask = (
        ((games_df["white"] == player_a) & (games_df["black"] == player_b))
        | ((games_df["white"] == player_b) & (games_df["black"] == player_a))
    )
    df = games_df[mask].sort_values("created_at", ascending=False)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in df.head(10).iterrows():
        if row.white == player_a:
            perspective = "white"
        elif row.black == player_a:
            perspective = "black"
        else:
            perspective = "white"
        score = _score_from_result(row.winner, perspective)
        result = "Win" if score == 1 else "Draw" if score == 0.5 else "Loss"
        rows.append(
            {
                "date": row.created_at.date(),
                "white": row.white,
                "black": row.black,
                "winner": row.winner or "draw",
                f"{player_a} result": result,
            }
        )
    return pd.DataFrame(rows)


def recent_result_icons(games_df: pd.DataFrame, player: str, window: int = 12) -> str:
    mask = (games_df["white"] == player) | (games_df["black"] == player)
    df = games_df[mask].sort_values("created_at", ascending=False).head(window)
    if df.empty:
        return "No games"
    icons = []
    for _, row in df.iterrows():
        perspective = "white" if row.white == player else "black"
        score = _score_from_result(row.winner, perspective)
        if score == 1.0:
            icons.append("✅")
        elif score == 0.5:
            icons.append("➖")
        else:
            icons.append("❌")
    return " ".join(icons)


def bucket_chart(profile_a: pd.DataFrame, profile_b: pd.DataFrame, player_a: str, player_b: str):
    frames = []
    for profile, player in ((profile_a, player_a), (profile_b, player_b)):
        if profile.empty:
            continue
        long_df = profile.melt(
            id_vars=["bucket"],
            value_vars=["win%", "draw%", "loss%"],
            var_name="result",
            value_name="percent",
        )
        long_df["player"] = player
        long_df["result"] = (
            long_df["result"].str.replace("%", "", regex=False).str.capitalize()
        )
        frames.append(long_df)
    if not frames:
        return None
    chart_data = pd.concat(frames, ignore_index=True)
    bucket_order = [label for _, _, label in RATING_BUCKETS] + ["unknown"]
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("bucket:N", sort=bucket_order, title="Opponent rating bucket"),
            y=alt.Y("percent:Q", title="Result %"),
            color=alt.Color(
                "result:N",
                scale=alt.Scale(
                    domain=["Win", "Draw", "Loss"],
                    range=["#34c759", "#a0a4a8", "#ff5b5b"],
                ),
            ),
            column=alt.Column("player:N", title=None),
            tooltip=["player", "bucket", "result", "percent"],
        )
        .properties(height=260)
    )
    return chart.configure_axis(labelColor="#666", titleColor="#666").configure_view(
        strokeWidth=0
    )


def handle_gm_search(slot: str, query: str):
    if "gm_suggestions" not in st.session_state:
        st.session_state["gm_suggestions"] = {"A": [], "B": []}
    if "gm_suggestion_msgs" not in st.session_state:
        st.session_state["gm_suggestion_msgs"] = {"A": "", "B": ""}
    query = (query or "").strip()
    if len(query) < 2:
        st.session_state["gm_suggestions"][slot] = []
        st.session_state["gm_suggestion_msgs"][slot] = "Enter at least two characters before searching."
        return
    try:
        suggestions = fetch_gm_suggestions(query)
        message = (
            f"Found {len(suggestions)} GM handle(s)." if suggestions else "No GM handles found for that query."
        )
    except Exception as exc:  # noqa: BLE001
        suggestions = []
        message = f"Lookup failed: {exc}"
    st.session_state["gm_suggestions"][slot] = suggestions
    st.session_state["gm_suggestion_msgs"][slot] = message


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
    default_b = "Illia_Nyzhnyk"

    if "gm_suggestions" not in st.session_state:
        st.session_state["gm_suggestions"] = {"A": [], "B": []}
    if "gm_suggestion_msgs" not in st.session_state:
        st.session_state["gm_suggestion_msgs"] = {"A": "", "B": ""}

    with col1:
        player_a_input = st.text_input(
            "Player A (White username)",
            value=default_a,
            key="player_a_text",
        ).strip()
        if st.button("Search GM suggestions", key="search_a"):
            with st.spinner("Searching Lichess..."):
                handle_gm_search("A", player_a_input)
        msg_a = st.session_state["gm_suggestion_msgs"]["A"]
        if msg_a:
            st.caption(msg_a)
        suggestions_a = st.session_state["gm_suggestions"]["A"]
        if suggestions_a:
            selection_a = st.selectbox(
                "Suggestions for Player A",
                ["(keep typed value)"] + suggestions_a,
                key="suggestions_box_a",
            )
            if selection_a != "(keep typed value)":
                player_a_input = selection_a
                st.session_state["player_a_text"] = selection_a

    with col2:
        player_b_input = st.text_input(
            "Player B (Black username)",
            value=default_b,
            key="player_b_text",
        ).strip()
        if st.button("Search GM suggestions", key="search_b"):
            with st.spinner("Searching Lichess..."):
                handle_gm_search("B", player_b_input)
        msg_b = st.session_state["gm_suggestion_msgs"]["B"]
        if msg_b:
            st.caption(msg_b)
        suggestions_b = st.session_state["gm_suggestions"]["B"]
        if suggestions_b:
            selection_b = st.selectbox(
                "Suggestions for Player B",
                ["(keep typed value)"] + suggestions_b,
                key="suggestions_box_b",
            )
            if selection_b != "(keep typed value)":
                player_b_input = selection_b
                st.session_state["player_b_text"] = selection_b

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

        shared_rows = shared_opponent_rows(player_a, player_b, profiles)
        profile_a = bucket_performance(games_df, player_a)
        profile_b = bucket_performance(games_df, player_b)
        h2h_df = head_to_head_history(games_df, player_a, player_b)
        recent_a = recent_result_icons(games_df, player_a)
        recent_b = recent_result_icons(games_df, player_b)

        tabs = st.tabs(
            ["Matchup insight", "Shared opponents", "Rating tiers", "Head-to-head"]
        )

        with tabs[0]:
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric(
                    label=f"P({player_a} wins vs {player_b})",
                    value=f"{probability*100:.1f}%",
                    delta=f"{(probability-elo_baseline)*100:.1f} pts vs Elo",
                )
            with metric_cols[1]:
                st.metric(
                    label="Elo baseline",
                    value=f"{elo_baseline*100:.1f}%",
                    delta=None,
                    help="Classic Elo expectation for the same rating gap",
                )
            st.progress(probability)
            st.caption(f"Elo baseline win chance: {elo_baseline*100:.1f}%")
            with st.expander("Feature vector details", expanded=False):
                st.dataframe(
                    {
                        "feature": list(features.keys()),
                        "value": [round(v, 4) for v in features.values()],
                    }
                )
            with st.expander("Model diagnostics (training set)", expanded=False):
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
                    "Diagnostics are computed on the fetched games only (no hold-out set). "
                    "Fetching more history generally improves stability."
                )

        with tabs[1]:
            if shared_rows:
                shared_df = pd.DataFrame(shared_rows)
                pretty_df = shared_df.copy()
                for col in [f"{player_a} score", f"{player_b} score"]:
                    pretty_df[col] = (pretty_df[col] * 100).round(1)
                pretty_df.rename(
                    columns={
                        f"{player_a} score": f"{player_a} score %",
                        f"{player_b} score": f"{player_b} score %",
                        "opp_rating": "opp avg rating",
                    },
                    inplace=True,
                )
                st.dataframe(pretty_df, use_container_width=True)
            else:
                st.info(
                    "No overlapping opponents in this sample yet—fetch more games to unlock this view."
                )

        with tabs[2]:
            chart = bucket_chart(profile_a, profile_b, player_a, player_b)
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Not enough games to build rating-bucket stats.")
            bucket_col_a, bucket_col_b = st.columns(2)
            if not profile_a.empty:
                with bucket_col_a:
                    st.markdown(f"**{player_a}**")
                    st.dataframe(profile_a)
            if not profile_b.empty:
                with bucket_col_b:
                    st.markdown(f"**{player_b}**")
                    st.dataframe(profile_b)

        with tabs[3]:
            if h2h_df.empty:
                st.caption("These players have not faced each other in the fetched window.")
            else:
                st.dataframe(h2h_df, use_container_width=True)
            form_col_a, form_col_b = st.columns(2)
            with form_col_a:
                st.markdown(f"**{player_a} last games**")
                st.write(recent_a)
            with form_col_b:
                st.markdown(f"**{player_b} last games**")
                st.write(recent_b)


if __name__ == "__main__":
    main()

