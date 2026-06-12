"""
ml/wc_features.py — Features WC-spécifiques pour le modèle XGBoost WC.

Enrichit les features de base (ELO, form, H2H) avec des données
par édition de CdM : ranking FIFA, valeur marchande, palmarès.

Données source :
  wc_teams_train.csv  — 192 équipes × 6 éditions (2002-2022)
  wc_teams_test.csv   — 48 équipes WC 2026
  features.csv        — features match-level de base (généré par features.py)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import DATA_DIR, FEATURE_COLS, build_features

# Features supplémentaires issues de wc_teams
WC_EXTRA_COLS = [
    "rank_diff",          # away_rank - home_rank  (positif = home mieux classé)
    "log_market_ratio",   # log(home_value / away_value)
    "wc_titles_diff",     # home_titles - away_titles
    "wc_participations_diff",
]

WC_FEATURE_COLS = FEATURE_COLS + WC_EXTRA_COLS

_NAME_MAP = {"Serbia and Montenegro": "Serbia"}


def _load_team_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["team"] = df["team"].replace(_NAME_MAP)
    return df


@lru_cache(maxsize=1)
def _train_lookup() -> dict[tuple[str, int], dict]:
    """(team, wc_year) -> feature dict  pour wc_teams_train (2002-2022)."""
    df = _load_team_data(DATA_DIR / "wc_teams_train.csv")
    result = {}
    for _, row in df.iterrows():
        result[(row["team"], int(row["version"]))] = row.to_dict()
    return result


@lru_cache(maxsize=1)
def _test_lookup() -> dict[str, dict]:
    """team -> feature dict pour wc_teams_test (2026)."""
    df = _load_team_data(DATA_DIR / "wc_teams_test.csv")
    return {row["team"]: row.to_dict() for _, row in df.iterrows()}


def _extra_features(home: dict | None, away: dict | None) -> dict:
    """Calcule les 4 features WC différentielles. Retourne 0 si données manquantes."""
    if home is None or away is None:
        return dict.fromkeys(WC_EXTRA_COLS, 0.0)

    home_val = home.get("squad_total_market_value_eur") or 0
    away_val = away.get("squad_total_market_value_eur") or 0

    if home_val > 0 and away_val > 0:
        log_market = float(np.log(home_val / away_val))
    else:
        log_market = 0.0

    return {
        "rank_diff":               float(away["fifa_rank_pre_tournament"] - home["fifa_rank_pre_tournament"]),
        "log_market_ratio":        log_market,
        "wc_titles_diff":          float(home["world_cup_titles_before"] - away["world_cup_titles_before"]),
        "wc_participations_diff":  float(home["world_cup_participations_before"] - away["world_cup_participations_before"]),
    }


# ---------------------------------------------------------------------------
# Construction du jeu d'entraînement WC
# ---------------------------------------------------------------------------

def build_wc_features(features_path: str | Path | None = None) -> pd.DataFrame:
    """
    Retourne un DataFrame avec WC_FEATURE_COLS + date + result.
    Filtre features.csv aux matchs WC 2002-2022 et joint avec wc_teams_train.csv.
    """
    features_path = Path(features_path or DATA_DIR / "features.csv")

    if not features_path.exists():
        print("features.csv introuvable — génération...")
        df = build_features()
        df.to_csv(features_path, index=False)
    else:
        df = pd.read_csv(features_path, parse_dates=["date"])

    # Filtrer aux matchs WC 2002-2022
    df = df[
        (df["tournament_tier"] == 4) &
        (df["date"].dt.year >= 2002) &
        (df["date"].dt.year <= 2022)
    ].copy()

    lookup = _train_lookup()
    extra_rows = []

    for _, row in df.iterrows():
        year = row["date"].year
        home = lookup.get((row["home_team"], year))
        away = lookup.get((row["away_team"], year))
        extra_rows.append(_extra_features(home, away))

    extra_df = pd.DataFrame(extra_rows, index=df.index)
    result = pd.concat([df, extra_df], axis=1)

    return result[WC_FEATURE_COLS + ["date", "result", "home_team", "away_team"]]


# ---------------------------------------------------------------------------
# Features pour l'inférence (WC 2026)
# ---------------------------------------------------------------------------

def wc_extra_for_match(home_team: str, away_team: str) -> dict:
    """
    Retourne les 4 features WC pour un match WC 2026.
    Utilisé par predict.py lors de l'inférence.
    """
    lookup = _test_lookup()
    home = lookup.get(home_team)
    away = lookup.get(away_team)
    return _extra_features(home, away)
