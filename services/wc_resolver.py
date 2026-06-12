"""
services/wc_resolver.py — Résolution automatique des prédictions CdM 2026.

Source primaire : worldcup26.ir/get/games (gratuit, pas de clé API)
Source fallback : football-data.org (clé API requise — peut retourner 403)

Format worldcup26.ir observé :
  {"games": [{"id": "1", "finished": "TRUE", "home_score": "2", "away_score": "0",
              "home_team_name_en": "Mexico", "away_team_name_en": "South Africa", ...}]}
  Attention : finished est une string "TRUE"/"FALSE", pas un booléen.
  Les matchs non joués ont home_score/away_score = "0" — filtrer sur finished == "TRUE".

Appelé toutes les heures par bot.py. Ne lève jamais d'exception.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

import database

FDORG_URL       = "https://api.football-data.org/v4"
WC26_URL        = "https://worldcup26.ir/get/games"
FIXTURES_PATH   = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"
MATCH_ID_OFFSET = 200_000

# Noms d'équipes renvoyés par les APIs (fdorg et worldcup26.ir utilisent les mêmes conventions)
# vers les noms utilisés dans wc2026_fixtures.csv.
_API_TO_FIXTURE: dict[str, str] = {
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


# ---------------------------------------------------------------------------
# Sources de données — retournent toutes deux le même format normalisé
# {"home": str, "away": str, "home_score": int, "away_score": int}
# ou None en cas d'échec réseau/parsing.
# ---------------------------------------------------------------------------

def _fetch_worldcup26ir() -> list[dict] | None:
    """Source primaire : worldcup26.ir — gratuit, pas de clé."""
    try:
        resp = requests.get(WC26_URL, timeout=10)
        resp.raise_for_status()
        games = resp.json().get("games", [])
    except requests.RequestException as e:
        print(f"[WC RESOLVER] worldcup26.ir indisponible : {e}")
        return None

    result = []
    for g in games:
        # finished est une string "TRUE"/"FALSE" — ne pas se fier aux scores qui valent "0" par défaut
        if g.get("finished") != "TRUE":
            continue
        try:
            result.append({
                "home":       g["home_team_name_en"],
                "away":       g["away_team_name_en"],
                "home_score": int(g["home_score"]),
                "away_score": int(g["away_score"]),
            })
        except (KeyError, ValueError):
            continue
    return result


def _fetch_fdorg() -> list[dict] | None:
    """Source fallback : football-data.org — clé API requise."""
    key = os.getenv("FOOTBALL_DATA_KEY")
    if not key:
        print("[WC RESOLVER] FOOTBALL_DATA_KEY manquant — fallback football-data.org impossible.")
        return None

    try:
        resp = requests.get(
            f"{FDORG_URL}/competitions/WC/matches",
            headers={"X-Auth-Token": key},
            params={"season": "2026"},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"[WC RESOLVER] football-data.org HTTP {e.response.status_code} : {e}")
        return None
    except requests.RequestException as e:
        print(f"[WC RESOLVER] football-data.org réseau : {e}")
        return None

    result = []
    for m in resp.json().get("matches", []):
        if m.get("status") != "FINISHED":
            continue
        score = m.get("score", {}).get("fullTime", {})
        h, a  = score.get("home"), score.get("away")
        if h is None or a is None:
            continue
        result.append({
            "home":       m.get("homeTeam", {}).get("name", ""),
            "away":       m.get("awayTeam", {}).get("name", ""),
            "home_score": int(h),
            "away_score": int(a),
        })
    return result


def _fetch_finished_matches() -> tuple[list[dict], str]:
    """
    Tente worldcup26.ir en premier, football-data.org en fallback.
    Retourne (liste normalisée des matchs terminés, nom de la source utilisée).
    """
    matches = _fetch_worldcup26ir()
    if matches is not None:
        print(f"[WC RESOLVER] Source : worldcup26.ir — {len(matches)} match(s) terminé(s)")
        return matches, "worldcup26.ir"

    print("[WC RESOLVER] Fallback sur football-data.org…")
    matches = _fetch_fdorg()
    if matches is not None:
        print(f"[WC RESOLVER] Source : football-data.org — {len(matches)} match(s) terminé(s)")
        return matches, "football-data.org"

    print("[WC RESOLVER] Les deux sources sont indisponibles.")
    return [], "none"


# ---------------------------------------------------------------------------
# Résolution principale
# ---------------------------------------------------------------------------

def resolve_wc_predictions() -> None:
    """
    Point d'entrée principal — appelé hourly par bot.py.

    Pour chaque match WC terminé :
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

    fixture_lookup          = _build_fixture_lookup()
    finished_raw, source    = _fetch_finished_matches()

    # Résoudre les noms et filtrer les matchs trouvés dans notre lookup
    to_process = []
    for match in finished_raw:
        home_fixture = _resolve_name(match["home"])
        away_fixture = _resolve_name(match["away"])
        fixture_info = fixture_lookup.get((home_fixture, away_fixture))

        if fixture_info is None:
            if match["home"] and match["away"]:
                print(
                    f"[WC RESOLVER] Mapping manquant ({source}) : "
                    f"'{match['home']}' → '{home_fixture}' | "
                    f"'{match['away']}' → '{away_fixture}' "
                    f"— ajouter à _API_TO_FIXTURE si nécessaire"
                )
            continue

        to_process.append((fixture_info, home_fixture, away_fixture, match))

    # Traiter dans l'ordre chronologique pour que les mises à jour ELO soient correctes
    to_process.sort(key=lambda x: x[0]["match_id"])

    print(f"[WC RESOLVER] {len(pending)} en attente | {len(to_process)} matchs à traiter ({source})")

    resolved   = 0
    wc_elo_path = Path(__file__).parent.parent / "ml" / "data" / "wc_elo_updates.csv"

    for fixture_info, home_fixture, away_fixture, match in to_process:
        match_id    = fixture_info["match_id"]
        match_date  = fixture_info["match_date"]
        match_group = fixture_info["group"]
        actual_h    = match["home_score"]
        actual_a    = match["away_score"]

        # Enregistrer le score réel pour /standings (indépendant des prédictions)
        database.save_match_result(
            match_id, home_fixture, away_fixture,
            actual_h, actual_a, match_group, match_date,
        )

        # Résoudre la prédiction DB si elle est en attente
        if match_id in pending:
            database.resolve_prediction(match_id, actual_h, actual_a)
            result_str = "H" if actual_h > actual_a else ("D" if actual_h == actual_a else "A")
            print(
                f"[WC RESOLVER] ✅ Résolu : {home_fixture} {actual_h}-{actual_a} {away_fixture} "
                f"({result_str})"
            )
            resolved += 1

        # Mettre à jour l'ELO dans tous les cas pour que wc_elo_updates.csv reste complet
        # après un redémarrage du bot (auto_resolve retraite tous les matchs terminés)
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
