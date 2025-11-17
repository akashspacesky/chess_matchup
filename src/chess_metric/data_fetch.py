import json
import time
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from tqdm import tqdm


LICHESS_GAMES_ENDPOINT = (
    "https://lichess.org/api/games/user/{username}"
    "?max={max_games}&rated=true&perfType=blitz,rapid,classical"
    "&moves=false&evals=false&clocks=false&opening=false&lore=false&format=ndjson"
)


def fetch_user_games(
    username: str,
    max_games: int = 200,
    pause_seconds: float = 1.0,
    session: Optional[requests.Session] = None,
) -> List[dict]:
    """Fetch recent rated games for a Lichess username via the public API."""
    sess = session or requests.Session()
    url = LICHESS_GAMES_ENDPOINT.format(username=username, max_games=max_games)
    headers = {"Accept": "application/x-ndjson"}
    response = sess.get(url, headers=headers, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch games for {username}: {response.status_code} {response.text[:200]}"
        )
    games = [json.loads(line) for line in response.text.strip().splitlines() if line.strip()]
    time.sleep(pause_seconds)
    return games


def download_games_for_players(
    usernames: Iterable[str],
    output_path: Path,
    max_games_per_user: int = 200,
) -> None:
    """Download games for a list of usernames and write into NDJSON."""
    sess = requests.Session()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for username in tqdm(list(usernames), desc="Downloading games"):
            try:
                games = fetch_user_games(username, max_games=max_games_per_user, session=sess)
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {username}: {exc}")
                continue
            for game in games:
                f_out.write(json.dumps(game) + "\n")


if __name__ == "__main__":
    TARGET_USERS = [
        "AnishGiri",
        "fabichess",
        "Illia_Nyzhnyk",
        "rasulovvugar",
        "Demchenko_Anton",
        "AbasovN",
        "UnVieuxMonsieur",
        "retired93",
        "Nichega",
        "Rockstar83",
        "Khalilmousavii",
        "Grandelicious",
        "Gabix94",
        "Feokl1995",
        "BabaRamdev",
    ]
    download_games_for_players(
        TARGET_USERS,
        output_path=Path("data/raw/lichess_sample.ndjson"),
        max_games_per_user=200,
    )

