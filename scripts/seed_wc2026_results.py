"""
scripts/seed_wc2026_results.py — Import en masse des résultats WC 2026 déjà joués.

Remplis les scores ci-dessous puis lance :
    python scripts/seed_wc2026_results.py

Le script insère tout en DB, met à jour l'ELO dans l'ordre chronologique,
et fait UN SEUL refit Poisson à la fin (au lieu de 1 par match via /score).
"""
import sys
from pathlib import Path

# Remonter à la racine du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

import database
import pandas as pd
from services.elo_updater import update_elo_with_match
from ml.poisson import fit, save, fit_or_load
from ml.predict import _data as _predict_data

# ---------------------------------------------------------------------------
# REMPLIS LES SCORES ICI
# Format : (match_number, home_goals, away_goals)
# None = match pas encore joué ou score inconnu → ignoré
# ---------------------------------------------------------------------------

RESULTS = [
    # --- Journée 1 (11-12 juin) ---
    (1,  None, None),  # Mexico         vs South Africa
    (2,  None, None),  # Korea Republic vs Czechia
    (3,  None, None),  # Canada         vs Bosnia and Herzegovina

    # --- Journée 1 suite (13-14 juin) ---
    (4,  None, None),  # USA            vs Paraguay
    (8,  None, None),  # Qatar          vs Switzerland
    (7,  None, None),  # Brazil         vs Morocco
    (5,  None, None),  # Haiti          vs Scotland
    (6,  None, None),  # Australia      vs Türkiye
    (10, None, None),  # Germany        vs Curaçao
    (11, None, None),  # Netherlands    vs Japan
    (9,  None, None),  # Côte d'Ivoire  vs Ecuador

    # --- Journée 1 suite (15-16 juin) ---
    (12, None, None),  # Sweden         vs Tunisia
    (14, None, None),  # Spain          vs Cabo Verde
    (16, None, None),  # Belgium        vs Egypt
    (13, None, None),  # Saudi Arabia   vs Uruguay
    (15, None, None),  # IR Iran        vs New Zealand
    (17, None, None),  # France         vs Senegal
    (18, None, None),  # Iraq           vs Norway

    # --- Journée 2 (17 juin) ---
    (19, None, None),  # Argentina      vs Algeria
    (20, None, None),  # Austria        vs Jordan
    (23, None, None),  # Portugal       vs Congo DR
    (22, None, None),  # England        vs Croatia
    (21, None, None),  # Ghana          vs Panama

    # --- Journée 2 suite (18 juin — déjà joués) ---
    (24, None, None),  # Uzbekistan     vs Colombia
]

# ---------------------------------------------------------------------------
# NE PAS MODIFIER EN DESSOUS
# ---------------------------------------------------------------------------

def main():
    fixtures = pd.read_csv(
        Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"
    )
    fixture_map = {int(r["match_number"]): r for _, r in fixtures.iterrows()}

    MATCH_ID_OFFSET = 200_000

    played = [(mn, h, a) for mn, h, a in RESULTS if h is not None and a is not None]
    if not played:
        print("Aucun score rempli. Modifie RESULTS dans ce fichier et relance.")
        return

    print(f"{len(played)} résultat(s) à importer...\n")

    for match_number, home_score, away_score in played:
        r = fixture_map.get(match_number)
        if r is None:
            print(f"  SKIP  match #{match_number} — introuvable dans fixtures")
            continue

        match_id   = MATCH_ID_OFFSET + match_number
        home_team  = r["home_team"]
        away_team  = r["away_team"]
        match_date = r["date"]
        group      = r["group"]

        database.save_match_result(match_id, home_team, away_team, home_score, away_score, group, match_date)
        database.resolve_prediction(match_id, home_score, away_score)

        # ELO en ordre chronologique — vérification anti-doublon
        wc_elo_path = Path(__file__).parent.parent / "ml" / "data" / "wc_elo_updates.csv"
        already = False
        if wc_elo_path.exists() and wc_elo_path.stat().st_size > 0:
            existing = pd.read_csv(wc_elo_path)
            already = bool(
                ((existing["team"].isin([home_team, away_team])) &
                 (existing["date"] == match_date)).any()
            )

        if not already:
            update_elo_with_match(home_team, away_team, home_score, away_score, match_date)

        result_str = "V dom" if home_score > away_score else ("Nul" if home_score == away_score else "V ext")
        print(f"  OK  #{match_number:>3}  {home_team} {home_score}-{away_score} {away_team}  [{result_str}]")

    # Un seul refit Poisson à la fin
    print("\nRefit Poisson avec tous les résultats...")
    params = fit()
    save(params)
    fit_or_load.cache_clear()
    _predict_data.cache_clear()
    print(f"Done — {params['n_matches']} matchs utilisés pour le fit.")
    print(f"Params sauvegardés dans : {Path(__file__).parent.parent / 'ml' / 'data' / 'poisson_params.json'}")


if __name__ == "__main__":
    main()
