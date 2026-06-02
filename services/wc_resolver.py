"""
services/wc_resolver.py — Résolution automatique des prédictions CdM 2026.

Interroge football-data.org pour les matchs WC terminés, les mappe sur nos
fixtures par noms d'équipes, et met à jour la DB via resolve_prediction().

Appelé toutes les heures par bot.py. Ne lève jamais d'exception — les erreurs
sont loggées et la tâche se poursuit au prochain cycle.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

import database

BASE_URL        = "https://api.football-data.org/v4"
FIXTURES_PATH   = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"
MATCH_ID_OFFSET = 200_000

# football-data.org utilise souvent les noms anglais courants, pas les noms FIFA officiels.
# Format : {nom_football_data_org: nom_dans_wc2026_fixtures.csv}
_FDORG_TO_FIXTURE: dict[str, str] = {
    "United States":      "USA",
    "South Korea":        "Korea Republic",
    "Iran":               "IR Iran",
    "Cape Verde":         "Cabo Verde",
    "DR Congo":           "Congo DR",
    "Ivory Coast":        "Côte d'Ivoire",
    "Czech Republic":     "Czechia",
    "Turkey":             "Türkiye",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curacao":            "Curaçao",
}


def _resolve_name(fdorg_name: str) -> str:
    """Traduit un nom football-data.org vers le nom utilisé dans wc2026_fixtures.csv."""
    return _FDORG_TO_FIXTURE.get(fdorg_name, fdorg_name)


def _build_fixture_lookup() -> dict[tuple[str, str], int]:
    """
    Construit un dict (home_team, away_team) → match_id depuis wc2026_fixtures.csv.
    Seuls les matchs du groupe stage ont des équipes connues.
    """
    df = pd.read_csv(FIXTURES_PATH)
    group_stage = df[df["stage"] == "Group Stage"]
    return {
        (row["home_team"], row["away_team"]): MATCH_ID_OFFSET + int(row["match_number"])
        for _, row in group_stage.iterrows()
    }


def _fetch_wc_matches() -> list[dict]:
    """Récupère tous les matchs WC2026 depuis football-data.org. Retourne [] en cas d'erreur."""
    key = os.getenv("FOOTBALL_DATA_KEY")
    if not key:
        print("[WC RESOLVER] FOOTBALL_DATA_KEY manquant — résolution impossible.")
        return []

    try:
        resp = requests.get(
            f"{BASE_URL}/competitions/WC/matches",
            headers={"X-Auth-Token": key},
            params={"season": "2026"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("matches", [])
    except requests.HTTPError as e:
        print(f"[WC RESOLVER] Erreur HTTP {e.response.status_code} : {e}")
        return []
    except requests.RequestException as e:
        print(f"[WC RESOLVER] Erreur réseau : {e}")
        return []


def resolve_wc_predictions() -> None:
    """
    Point d'entrée principal — appelé hourly par bot.py.

    Pour chaque match WC terminé dans l'API :
    1. Traduit les noms d'équipes vers nos noms de fixtures
    2. Retrouve le match_id dans notre table de fixtures
    3. Si une prédiction DB est en attente pour ce match, la résout
    """
    pending = {p["match_id"] for p in database.get_pending_predictions()}
    if not pending:
        print("[WC RESOLVER] Aucune prédiction en attente.")
        return

    fixture_lookup = _build_fixture_lookup()
    matches        = _fetch_wc_matches()
    finished       = [m for m in matches if m.get("status") == "FINISHED"]

    print(f"[WC RESOLVER] {len(pending)} en attente | {len(finished)} matchs terminés dans l'API")

    resolved = 0
    for match in finished:
        home_fdorg = match.get("homeTeam", {}).get("name", "")
        away_fdorg = match.get("awayTeam", {}).get("name", "")

        home_fixture = _resolve_name(home_fdorg)
        away_fixture = _resolve_name(away_fdorg)

        match_id = fixture_lookup.get((home_fixture, away_fixture))

        if match_id is None:
            # Peut arriver si le nom n'est pas dans _FDORG_TO_FIXTURE
            # ou si c'est un match de phase éliminatoire (TBD)
            if home_fdorg and away_fdorg:
                print(
                    f"[WC RESOLVER] Mapping manquant : "
                    f"'{home_fdorg}' → '{home_fixture}' | "
                    f"'{away_fdorg}' → '{away_fixture}' "
                    f"— ajouter à _FDORG_TO_FIXTURE si nécessaire"
                )
            continue

        if match_id not in pending:
            continue  # Déjà résolu

        score    = match.get("score", {}).get("fullTime", {})
        actual_h = score.get("home")
        actual_a = score.get("away")

        if actual_h is None or actual_a is None:
            print(f"[WC RESOLVER] Score fullTime manquant pour {home_fixture} vs {away_fixture}")
            continue

        database.resolve_prediction(match_id, actual_h, actual_a)

        result_str = "H" if actual_h > actual_a else ("D" if actual_h == actual_a else "A")
        print(
            f"[WC RESOLVER] ✅ Résolu : {home_fixture} {actual_h}-{actual_a} {away_fixture} "
            f"({result_str})"
        )
        resolved += 1

    print(f"[WC RESOLVER] {resolved} prédiction(s) résolue(s) ce cycle.")
