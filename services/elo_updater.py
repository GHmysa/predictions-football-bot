"""
services/elo_updater.py — Mise à jour incrémentale de l'ELO pendant la CdM 2026.

Après chaque match résolu, calcule les nouveaux ELO et les appende dans
ml/data/wc_elo_updates.csv (delta léger — 2 lignes par match).

elo_history.csv n'est jamais modifié : il reste la source historique immuable.
predict.py fusionne les deux fichiers à la volée lors de l'inférence.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_PERSISTENT_DIR   = Path(os.environ.get("PERSISTENT_DIR", Path(__file__).parent.parent / "ml" / "data"))
ELO_HISTORY_PATH  = Path(__file__).parent.parent / "ml" / "data" / "elo_history.csv"
WC_ELO_PATH       = _PERSISTENT_DIR / "wc_elo_updates.csv"
WC_TEAMS_PATH     = Path(__file__).parent.parent / "ml" / "data" / "wc2026_teams.csv"
ELO_INIT          = 1500.0
K_WC              = 40.0  # Facteur K pour les matchs de Coupe du Monde


def _fixture_to_dataset(fixture_name: str) -> str:
    """Traduit un nom FIFA (wc2026_fixtures) en nom dataset (elo_history)."""
    wc = pd.read_csv(WC_TEAMS_PATH)
    mapping = dict(zip(wc["fifa_name"], wc["dataset_name"]))
    return mapping.get(fixture_name, fixture_name)


def _current_elo(team_dataset: str) -> float:
    """
    Retourne l'ELO le plus récent d'une équipe en fusionnant elo_history.csv
    et wc_elo_updates.csv (si existant).
    C'est toujours l'ELO après le dernier match joué, même pendant le tournoi.
    """
    elo = pd.read_csv(ELO_HISTORY_PATH, parse_dates=["date"])

    if WC_ELO_PATH.exists() and WC_ELO_PATH.stat().st_size > 0:
        wc_elo = pd.read_csv(WC_ELO_PATH, parse_dates=["date"])
        elo = pd.concat([elo, wc_elo], ignore_index=True)

    elo = elo.sort_values("date")
    subset = elo[elo["team"] == team_dataset]
    return float(subset.iloc[-1]["elo"]) if not subset.empty else ELO_INIT


def update_elo_with_match(
    home_fixture: str,
    away_fixture: str,
    home_score: int,
    away_score: int,
    match_date: str,
) -> None:
    """
    Calcule les nouveaux ELO après un match WC et les appende à wc_elo_updates.csv.

    Paramètres
    ----------
    home_fixture / away_fixture : noms FIFA tels qu'ils apparaissent dans wc2026_fixtures.csv
    match_date                  : YYYY-MM-DD
    """
    home_ds = _fixture_to_dataset(home_fixture)
    away_ds = _fixture_to_dataset(away_fixture)

    elo_home = _current_elo(home_ds)
    elo_away = _current_elo(away_ds)

    # Formule ELO standard
    expected_home = 1.0 / (1.0 + 10.0 ** ((elo_away - elo_home) / 400.0))
    score = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)

    new_elo_home = elo_home + K_WC * (score - expected_home)
    new_elo_away = elo_away + K_WC * ((1.0 - score) - (1.0 - expected_home))

    new_rows = pd.DataFrame([
        {"date": pd.Timestamp(match_date), "team": home_ds, "elo": new_elo_home},
        {"date": pd.Timestamp(match_date), "team": away_ds, "elo": new_elo_away},
    ])

    # Créer ou appender — pas de chargement du fichier complet en mémoire
    write_header = not WC_ELO_PATH.exists() or WC_ELO_PATH.stat().st_size == 0
    new_rows.to_csv(WC_ELO_PATH, mode="a", index=False, header=write_header)

    print(
        f"[ELO UPDATE] {home_fixture} ({home_ds}) : {elo_home:.1f} → {new_elo_home:.1f}  |  "
        f"{away_fixture} ({away_ds}) : {elo_away:.1f} → {new_elo_away:.1f}"
    )

    # Invalider le cache de predict.py pour que les prochaines prédictions
    # relisent wc_elo_updates.csv avec les valeurs fraîches
    try:
        from ml.predict import _data
        _data.cache_clear()
    except ImportError:
        pass

def remove_elo_for_match(home_fixture: str, away_fixture: str, match_date: str) -> None:
    """
    Supprime les mises à jour ELO pour un match donné (home/away/date) dans wc_elo_updates.csv.
    Utilisé pour corriger un score saisi par erreur.
    """
    if not WC_ELO_PATH.exists() or WC_ELO_PATH.stat().st_size == 0:
        print(f"[ELO REMOVE] Aucun fichier wc_elo_updates.csv trouvé — rien à supprimer")
        return

    home_ds = _fixture_to_dataset(home_fixture)
    away_ds = _fixture_to_dataset(away_fixture)

    elo = pd.read_csv(WC_ELO_PATH, parse_dates=["date"])
    before_count = len(elo)
    elo = elo[~(
        ((elo["team"] == home_ds) | (elo["team"] == away_ds)) &
        (elo["date"] == pd.Timestamp(match_date))
    )]
    after_count = len(elo)

    elo.to_csv(WC_ELO_PATH, index=False)
    print(f"[ELO REMOVE] {before_count - after_count} lignes supprimées pour {home_fixture}/{away_fixture} le {match_date}")

    # Invalider le cache de predict.py pour que les prochaines prédictions
    # relisent wc_elo_updates.csv avec les valeurs fraîches
    try:
        from ml.predict import _data
        _data.cache_clear()
    except ImportError:
        pass