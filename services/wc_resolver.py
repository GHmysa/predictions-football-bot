"""
services/wc_resolver.py — Résolution automatique des prédictions CdM 2026.

Source : openfootball (GitHub raw) — pas de clé API, mis à jour quotidiennement.
Fallback manuel : commande Discord /score (admin uniquement).

Appelé toutes les heures par bot.py. Ne lève jamais d'exception.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

import database

OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
FIXTURES_PATH    = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"
MATCH_ID_OFFSET  = 200_000

_API_TO_FIXTURE: dict[str, str] = {
    "United States":        "USA",
    "South Korea":          "Korea Republic",
    "Iran":                 "IR Iran",
    "Cape Verde":           "Cabo Verde",
    "DR Congo":             "Congo DR",
    "Ivory Coast":          "Côte d'Ivoire",
    "Czech Republic":       "Czechia",
    "Turkey":               "Türkiye",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Curacao":              "Curaçao",
}


def _resolve_name(name: str) -> str:
    return _API_TO_FIXTURE.get(name, name)


def _build_fixture_lookup() -> dict[tuple[str, str], dict]:
    df = pd.read_csv(FIXTURES_PATH)
    group_stage = df[df["stage"] == "Group Stage"]
    return {
        (row["home_team"], row["away_team"]): {
            "match_id":   MATCH_ID_OFFSET + int(row["match_number"]),
            "match_date": row["date"],
            "group":      row["group"],
        }
        for _, row in group_stage.iterrows()
    }


def _fetch_finished_matches() -> list[dict]:
    """
    Récupère les matchs terminés depuis openfootball (GitHub raw).
    Les matchs à venir n'ont pas de clé "score" — filtrage naturel.
    Retourne [] si l'API est indisponible.
    """
    try:
        resp = requests.get(OPENFOOTBALL_URL, timeout=10)
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
    except requests.RequestException as e:
        print(f"[WC RESOLVER] openfootball indisponible : {e}")
        return []

    result = []
    for m in matches:
        score = m.get("score", {}).get("ft")
        if not score or len(score) < 2:
            continue
        try:
            result.append({
                "home":       m["team1"],
                "away":       m["team2"],
                "home_score": int(score[0]),
                "away_score": int(score[1]),
            })
        except (KeyError, ValueError):
            continue
    return result


def resolve_wc_predictions() -> None:
    """
    Point d'entrée principal — appelé hourly par bot.py.

    Pour chaque match terminé dans openfootball :
    1. Mappe les noms d'équipes vers les noms de fixtures
    2. Résout la prédiction DB si en attente
    3. Enregistre le score pour /standings
    4. Met à jour l'ELO pour les prochaines prédictions
    """
    from services.elo_updater import update_elo_with_match

    pending = {p["match_id"] for p in database.get_pending_predictions()}
    if not pending:
        print("[WC RESOLVER] Aucune prédiction en attente.")
        return

    fixture_lookup = _build_fixture_lookup()
    finished       = _fetch_finished_matches()

    to_process = []
    for match in finished:
        home = _resolve_name(match["home"])
        away = _resolve_name(match["away"])
        info = fixture_lookup.get((home, away))
        if info is None:
            if match["home"] and match["away"]:
                print(f"[WC RESOLVER] Mapping manquant : '{match['home']}' | '{match['away']}' — ajouter à _API_TO_FIXTURE")
            continue
        to_process.append((info, home, away, match))

    to_process.sort(key=lambda x: x[0]["match_id"])
    print(f"[WC RESOLVER] {len(pending)} en attente | {len(to_process)} matchs terminés (openfootball)")

    resolved    = 0
    wc_elo_path = Path(__file__).parent.parent / "ml" / "data" / "wc_elo_updates.csv"

    for info, home, away, match in to_process:
        match_id    = info["match_id"]
        match_date  = info["match_date"]
        match_group = info["group"]
        actual_h    = match["home_score"]
        actual_a    = match["away_score"]

        database.save_match_result(match_id, home, away, actual_h, actual_a, match_group, match_date)

        if match_id in pending:
            database.resolve_prediction(match_id, actual_h, actual_a)
            result_str = "H" if actual_h > actual_a else ("D" if actual_h == actual_a else "A")
            print(f"[WC RESOLVER] ✅ {home} {actual_h}-{actual_a} {away} ({result_str})")
            resolved += 1

        already_updated = False
        if wc_elo_path.exists() and wc_elo_path.stat().st_size > 0:
            existing = pd.read_csv(wc_elo_path)
            already_updated = (
                (existing["team"].isin([home, away])) &
                (existing["date"] == match_date)
            ).any()

        if not already_updated:
            update_elo_with_match(home, away, actual_h, actual_a, match_date)

    print(f"[WC RESOLVER] {resolved} prédiction(s) résolue(s) ce cycle.")
