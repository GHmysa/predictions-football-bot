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


def _build_fixture_lookup() -> dict[tuple[str, str], dict]:
    """
    Construit un dict (home_team, away_team) → {match_id, date} depuis wc2026_fixtures.csv.
    Seuls les matchs du groupe stage ont des équipes connues.
    """
    df = pd.read_csv(FIXTURES_PATH)
    group_stage = df[df["stage"] == "Group Stage"]
    return {
        (row["home_team"], row["away_team"]): {
            "match_id":  MATCH_ID_OFFSET + int(row["match_number"]),
            "match_date": row["date"],
        }
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
    3. Si une prédiction DB est en attente, la résout
    4. Met à jour l'ELO des deux équipes pour les prochaines prédictions
    """
    from services.elo_updater import update_elo_with_match

    pending = {p["match_id"] for p in database.get_pending_predictions()}
    if not pending:
        print("[WC RESOLVER] Aucune prédiction en attente.")
        return

    fixture_lookup = _build_fixture_lookup()
    matches        = _fetch_wc_matches()

    # Trier par date pour que les mises à jour ELO soient chronologiquement correctes
    finished = sorted(
        [m for m in matches if m.get("status") == "FINISHED"],
        key=lambda m: m.get("utcDate", ""),
    )

    print(f"[WC RESOLVER] {len(pending)} en attente | {len(finished)} matchs terminés dans l'API")

    resolved = 0
    for match in finished:
        home_fdorg = match.get("homeTeam", {}).get("name", "")
        away_fdorg = match.get("awayTeam", {}).get("name", "")

        home_fixture = _resolve_name(home_fdorg)
        away_fixture = _resolve_name(away_fdorg)

        fixture_info = fixture_lookup.get((home_fixture, away_fixture))

        if fixture_info is None:
            if home_fdorg and away_fdorg:
                print(
                    f"[WC RESOLVER] Mapping manquant : "
                    f"'{home_fdorg}' → '{home_fixture}' | "
                    f"'{away_fdorg}' → '{away_fixture}' "
                    f"— ajouter à _FDORG_TO_FIXTURE si nécessaire"
                )
            continue

        match_id   = fixture_info["match_id"]
        match_date = fixture_info["match_date"]

        score    = match.get("score", {}).get("fullTime", {})
        actual_h = score.get("home")
        actual_a = score.get("away")

        if actual_h is None or actual_a is None:
            print(f"[WC RESOLVER] Score fullTime manquant pour {home_fixture} vs {away_fixture}")
            continue

        # Résoudre la prédiction DB si elle est en attente
        if match_id in pending:
            database.resolve_prediction(match_id, actual_h, actual_a)
            result_str = "H" if actual_h > actual_a else ("D" if actual_h == actual_a else "A")
            print(
                f"[WC RESOLVER] ✅ Résolu : {home_fixture} {actual_h}-{actual_a} {away_fixture} "
                f"({result_str})"
            )
            resolved += 1

        # Mettre à jour l'ELO dans tous les cas (même si la prédiction était déjà résolue)
        # pour que wc_elo_updates.csv reste complet en cas de redémarrage du bot
        wc_elo_path = Path(__file__).parent.parent / "ml" / "data" / "wc_elo_updates.csv"
        already_updated = False
        if wc_elo_path.exists() and wc_elo_path.stat().st_size > 0:
            existing = pd.read_csv(wc_elo_path)
            already_updated = (
                (existing["team"].isin([home_fixture, away_fixture])) &
                (existing["date"] == match_date)
            ).any()

        if not already_updated:
            update_elo_with_match(home_fixture, away_fixture, actual_h, actual_a, match_date)

    print(f"[WC RESOLVER] {resolved} prédiction(s) résolue(s) ce cycle.")
