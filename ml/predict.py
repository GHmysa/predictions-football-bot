"""
ml/predict.py — Inférence sur un match unique.

Accepte les noms officiels FIFA (ex. "IR Iran") ou les noms dataset (ex. "Iran").
Les données et le modèle sont mis en cache après le premier appel.

Usage :
    from ml.predict import predict_match
    print(predict_match("France", "Argentina", date="2026-06-20"))

Exécution directe (exemples) :
    python -m ml.predict
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import DATA_DIR, ELO_INIT, FEATURE_COLS, FORM_WINDOW, H2H_WINDOW, WC_HOSTS
from ml.poisson import fit_or_load as _poisson_params
from ml.poisson import predict_score as _poisson_score
from ml.train import MODEL_PATH

LABEL_MAP = {2: "home", 1: "draw", 0: "away"}
LABEL_FR  = {2: "Victoire domicile", 1: "Match nul", 0: "Victoire extérieur"}


# ---------------------------------------------------------------------------
# Chargement paresseux (une seule fois par process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _name_map() -> dict[str, str]:
    """FIFA name → dataset name."""
    wc = pd.read_csv(DATA_DIR / "wc2026_teams.csv")
    return dict(zip(wc["fifa_name"], wc["dataset_name"]))


@lru_cache(maxsize=1)
def _model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne (results, elo_history) triés par date.

    Fusionne elo_history.csv (données historiques immuables) avec
    wc_elo_updates.csv (delta des matchs CdM joués) si ce fichier existe.
    Appelé une fois par process ; le cache est invalidé par elo_updater.py
    après chaque mise à jour pour forcer le rechargement.
    """
    results = (
        pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
        .dropna(subset=["home_score", "away_score"])  # exclure les matchs futurs sans score
        .sort_values("date")
        .reset_index(drop=True)
    )
    elo = pd.read_csv(DATA_DIR / "elo_history.csv", parse_dates=["date"])

    wc_elo_path = DATA_DIR / "wc_elo_updates.csv"
    if wc_elo_path.exists() and wc_elo_path.stat().st_size > 0:
        wc_elo = pd.read_csv(wc_elo_path, parse_dates=["date"])
        elo = pd.concat([elo, wc_elo], ignore_index=True).sort_values(["team", "date"])

    return results, elo


def _resolve(name: str) -> str:
    """Traduit un nom FIFA en nom dataset si nécessaire."""
    return _name_map().get(name, name)


# ---------------------------------------------------------------------------
# Calcul des features pour un seul match
# ---------------------------------------------------------------------------

def _team_elo(team: str, date: pd.Timestamp, elo: pd.DataFrame) -> float:
    """Dernier ELO connu de l'équipe strictement avant `date`."""
    subset = elo[(elo["team"] == team) & (elo["date"] < date)]
    return float(subset.iloc[-1]["elo"]) if not subset.empty else ELO_INIT


def _form(team: str, date: pd.Timestamp, results: pd.DataFrame, n: int = FORM_WINDOW) -> dict:
    """Stats de forme sur les n derniers matchs avant `date`."""
    mask = (
        ((results["home_team"] == team) | (results["away_team"] == team)) &
        (results["date"] < date)
    )
    past = results[mask].tail(n)
    m    = len(past)

    if m == 0:
        return {"form_pts": 0.0, "form_gf": 0.0, "form_ga": 0.0}

    is_home = (past["home_team"] == team).values
    gf = np.where(is_home, past["home_score"].values, past["away_score"].values).astype(float)
    ga = np.where(is_home, past["away_score"].values, past["home_score"].values).astype(float)
    pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))

    return {
        "form_pts": float(pts.sum()) / (3 * m),
        "form_gf":  float(gf.mean()),
        "form_ga":  float(ga.mean()),
    }


def _wc_form(team: str, date: pd.Timestamp, results: pd.DataFrame, n: int = FORM_WINDOW) -> float:
    """Win rate dans les n derniers matchs de tournoi majeur (tier >= 3) avant `date`."""
    from ml.features import get_tournament_tier
    mask = (
        ((results["home_team"] == team) | (results["away_team"] == team)) &
        (results["date"] < date) &
        (results["tournament"].map(get_tournament_tier) >= 3)
    )
    past = results[mask].tail(n)
    m    = len(past)

    if m == 0:
        return 0.0

    is_home = (past["home_team"] == team).values
    gf = np.where(is_home, past["home_score"].values, past["away_score"].values).astype(float)
    ga = np.where(is_home, past["away_score"].values, past["home_score"].values).astype(float)
    pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
    return float(pts.sum()) / (3 * m)


def _rest_days(team: str, date: pd.Timestamp, results: pd.DataFrame, cap: int = 30) -> int:
    """Jours depuis le dernier match de l'équipe, plafonné à cap."""
    mask = (
        ((results["home_team"] == team) | (results["away_team"] == team)) &
        (results["date"] < date)
    )
    past = results[mask]
    if past.empty:
        return cap
    return min(int((date - past.iloc[-1]["date"]).days), cap)


def _h2h(home: str, away: str, date: pd.Timestamp, results: pd.DataFrame, n: int = H2H_WINDOW) -> dict:
    """Stats H2H entre home et away sur les n dernières confrontations avant `date`."""
    mask = (
        (results["date"] < date) &
        (
            ((results["home_team"] == home) & (results["away_team"] == away)) |
            ((results["home_team"] == away) & (results["away_team"] == home))
        )
    )
    past = results[mask].tail(n)
    m    = len(past)

    if m == 0:
        return {"h2h_home_pts": 0.0, "h2h_gd": 0.0, "h2h_n": 0}

    is_home = (past["home_team"] == home).values
    gf = np.where(is_home, past["home_score"].values, past["away_score"].values).astype(float)
    ga = np.where(is_home, past["away_score"].values, past["home_score"].values).astype(float)
    pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))

    return {
        "h2h_home_pts": float(pts.sum()) / (3 * m),
        "h2h_gd":       float((gf - ga).mean()),
        "h2h_n":        m,
    }


def build_match_features(
    home_team: str,
    away_team: str,
    date: pd.Timestamp,
    is_neutral: bool = True,
    tournament_tier: int = 4,
) -> pd.DataFrame:
    """
    Construit un DataFrame d'une ligne avec toutes les features d'un match.
    Les noms passés doivent être les noms *dataset* (déjà résolus).
    """
    results, elo = _data()

    elo_home = _team_elo(home_team, date, elo)
    elo_away = _team_elo(away_team, date, elo)
    hf       = _form(home_team, date, results)
    af       = _form(away_team, date, results)
    h        = _h2h(home_team, away_team, date, results)
    hosts    = WC_HOSTS.get(date.year, set()) if tournament_tier == 4 else set()

    row = {
        "elo_home":          elo_home,
        "elo_away":          elo_away,
        "elo_diff":          elo_home - elo_away,
        "home_form_pts":     hf["form_pts"],
        "home_form_gf":      hf["form_gf"],
        "home_form_ga":      hf["form_ga"],
        "away_form_pts":     af["form_pts"],
        "away_form_gf":      af["form_gf"],
        "away_form_ga":      af["form_ga"],
        "h2h_home_pts":      h["h2h_home_pts"],
        "h2h_gd":            h["h2h_gd"],
        "h2h_n":             h["h2h_n"],
        "is_neutral":        int(is_neutral),
        "tournament_tier":   tournament_tier,
        "home_is_host":      int(home_team in hosts),
        "away_is_host":      int(away_team in hosts),
        "home_wc_form_pts":  _wc_form(home_team, date, results),
        "away_wc_form_pts":  _wc_form(away_team, date, results),
        "home_rest_days":    _rest_days(home_team, date, results),
        "away_rest_days":    _rest_days(away_team, date, results),
    }

    return pd.DataFrame([row])[FEATURE_COLS]


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def predict_match(
    home_team: str,
    away_team: str,
    date: str = "2026-06-11",
    is_neutral: bool = True,
    tournament_tier: int = 4,
) -> dict:
    """
    Prédit le résultat d'un match.

    Paramètres
    ----------
    home_team, away_team : noms FIFA officiels ou noms dataset
    date                 : date du match (YYYY-MM-DD)
    is_neutral           : True pour tous les matchs de CdM (terrain neutre)
    tournament_tier      : 4=CdM, 3=Continental, 2=Qualif, 1=Amical

    Retourne
    --------
    {
        "home_team": str,
        "away_team": str,
        "date": str,
        "prediction": "home" | "draw" | "away",
        "prediction_fr": str,
        "confidence": float,          # proba de l'issue prédite
        "probabilities": {
            "home": float,
            "draw": float,
            "away": float,
        },
        "elo_home": float,
        "elo_away": float,
    }
    """
    home_ds = _resolve(home_team)
    away_ds = _resolve(away_team)
    dt      = pd.Timestamp(date)

    features  = build_match_features(home_ds, away_ds, dt, is_neutral, tournament_tier)
    proba     = _model().predict_proba(features)[0]  # [P(away), P(draw), P(home)]
    pred_idx  = int(np.argmax(proba))
    prediction = LABEL_MAP[pred_idx]

    score = _poisson_score(home_ds, away_ds, _poisson_params(), is_neutral=is_neutral)
    matrix = score["score_matrix"]

    # Most likely score consistent with the XGBoost prediction
    if prediction == "home":
        masked = np.tril(matrix, -1)           # home goals > away goals
    elif prediction == "away":
        masked = np.triu(matrix, 1)            # away goals > home goals
    else:
        diag = np.diag(np.diag(matrix))        # home goals == away goals
        masked = diag

    if masked.sum() > 0:
        best = np.unravel_index(np.argmax(masked), masked.shape)
        pred_score_home, pred_score_away = int(best[0]), int(best[1])
    else:
        pred_score_home, pred_score_away = score["most_likely_score"]

    return {
        "home_team":          home_team,
        "away_team":          away_team,
        "date":               date,
        "prediction":         prediction,
        "prediction_fr":      LABEL_FR[pred_idx],
        "confidence":         round(float(proba[pred_idx]), 3),
        "probabilities": {
            "home": round(float(proba[2]), 3),
            "draw": round(float(proba[1]), 3),
            "away": round(float(proba[0]), 3),
        },
        "elo_home":           round(float(features["elo_home"].iloc[0]), 1),
        "elo_away":           round(float(features["elo_away"].iloc[0]), 1),
        "predicted_score_home": pred_score_home,
        "predicted_score_away": pred_score_away,
    }


if __name__ == "__main__":
    test_matches = [
        ("France",    "Argentina"),
        ("Brazil",    "England"),
        ("Spain",     "Germany"),
        ("Morocco",   "Portugal"),
        ("IR Iran",   "United States"),   # test résolution nom FIFA
        ("Japan",     "Korea Republic"),  # test résolution nom FIFA
    ]

    print("-" * 60)
    for home, away in test_matches:
        r = predict_match(home, away, date="2026-06-20")
        p = r["probabilities"]
        print(f"{home:25s} vs {away}")
        print(f"  -> {r['prediction_fr']} (confiance {r['confidence']:.0%})")
        print(f"     domicile {p['home']:.0%}  nul {p['draw']:.0%}  extérieur {p['away']:.0%}")
        print(f"     Score : {r['predicted_score_home']}-{r['predicted_score_away']}")
        print(f"     ELO : {r['elo_home']} vs {r['elo_away']}")
        print("-" * 60)
