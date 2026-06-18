"""
ml/poisson.py — Modèle Dixon-Coles pour la prédiction de score.

Estime les paramètres attack/defense de chaque équipe par maximum de vraisemblance
sur les matchs compétitifs depuis 2018, avec pondération temporelle exponentielle.

API publique :
    params = fit_or_load()
    result = predict_score("France", "Argentina", params)

Exécution directe (refit + exemples) :
    python -m ml.poisson
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special
from scipy.optimize import minimize
from scipy.stats import poisson

from ml.features import DATA_DIR

# Params mis à jour pendant le tournoi → stockés dans PERSISTENT_DIR
_PERSISTENT_DIR  = Path(os.environ.get("PERSISTENT_DIR", Path(__file__).parent / "data"))
PARAMS_PATH      = _PERSISTENT_DIR / "poisson_params.json"
PARAMS_PATH_BUNDLED = Path(__file__).parent / "data" / "poisson_params.json"

# Matchs compétitifs depuis cette date pour l'estimation
FIT_SINCE = "2018-01-01"

# Decay temporel : ~50% de poids pour un match vieux de ~1 an
XI = 0.002  # ξ, en jours⁻¹

# Tournois compétitifs (tier >= 2) — exclure les amicaux
_TIER2_KEYWORDS = {
    "FIFA World Cup",
    "Copa América",
    "UEFA Euro",
    "Africa Cup of Nations",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Gold Cup",
    "UEFA Nations League",
    "CONCACAF Nations League",
    "OFC Nations Cup",
    "AFF Championship",
    "ASEAN Championship",
    "EAFF Championship",
    "Gulf Cup",
    "Arab Cup",
    "COSAFA Cup",
    "SAFF Cup",
}

_QUAL_KEYWORDS = ("qualification", "qualifying")


def _is_competitive(tournament: str) -> bool:
    if tournament in _TIER2_KEYWORDS:
        return True
    t_lower = tournament.lower()
    return any(k in t_lower for k in _QUAL_KEYWORDS)


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def _wc2026_teams() -> set[str]:
    """Noms dataset des 48 équipes WC 2026."""
    return set(
        pd.read_csv(DATA_DIR / "wc2026_teams.csv")["dataset_name"]
    )


def _wc2026_results_from_db() -> pd.DataFrame:
    """
    Lit les résultats WC 2026 déjà joués depuis la table match_results (SQLite).
    Mappe les noms FIFA → noms dataset pour cohérence avec results.csv.
    Retourne un DataFrame vide si la DB est inaccessible ou vide.
    """
    try:
        import database
        with database.get_connection() as conn:
            rows = conn.execute(
                "SELECT home_team, away_team, home_score, away_score, match_date "
                "FROM match_results"
            ).fetchall()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["home_team", "away_team", "home_score", "away_score", "date"])
        df["date"] = pd.to_datetime(df["date"])
        df["tournament"] = "FIFA World Cup"
        df["neutral"] = True

        wc_teams = pd.read_csv(DATA_DIR / "wc2026_teams.csv")
        name_map = dict(zip(wc_teams["fifa_name"], wc_teams["dataset_name"]))
        df["home_team"] = df["home_team"].map(lambda x: name_map.get(x, x))
        df["away_team"] = df["away_team"].map(lambda x: name_map.get(x, x))
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        return df
    except Exception as e:
        print(f"[POISSON] Impossible de lire les resultats WC 2026 depuis la DB : {e}")
        return pd.DataFrame()


def _load_matches() -> pd.DataFrame:
    """
    Matchs compétitifs depuis FIT_SINCE où au moins une équipe joue en WC 2026,
    augmentés des résultats WC 2026 déjà joués (depuis la DB SQLite).
    """
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["date"] >= FIT_SINCE]
    df = df[df["tournament"].map(_is_competitive)]

    wc = _wc2026_teams()
    mask = df["home_team"].isin(wc) | df["away_team"].isin(wc)
    df = df[mask]

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    wc2026 = _wc2026_results_from_db()
    if not wc2026.empty:
        df = pd.concat([df, wc2026], ignore_index=True).sort_values("date").reset_index(drop=True)
        print(f"  + {len(wc2026)} resultats WC 2026 integres depuis la DB")

    return df


# ---------------------------------------------------------------------------
# Log-vraisemblance vectorisée Dixon-Coles
# ---------------------------------------------------------------------------

def _neg_log_likelihood(
    params: np.ndarray,
    n_free: int,         # n_teams - 1 (ref team fixed at 0)
    h_idx: np.ndarray,
    a_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    is_neutral: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Objective function (minimisée).

    Paramétrage — équipe 0 fixée comme référence (log_att=0, log_def=0) :
      params[:n_free]        = log(attack_1..n)
      params[n_free:2*n_free] = log(defense_1..n)
      params[2*n_free]       = log(home_adv)
      params[2*n_free+1]     = rho  (correction Dixon-Coles, dans [-1, 1])
    """
    log_att = np.concatenate([[0.0], params[:n_free]])
    log_def = np.concatenate([[0.0], params[n_free : 2 * n_free]])
    home_adv = np.exp(params[2 * n_free])
    rho = params[2 * n_free + 1]

    adv = np.where(is_neutral, 1.0, home_adv)
    lam_h = np.exp(log_att[h_idx] + log_def[a_idx]) * adv
    lam_a = np.exp(log_att[a_idx] + log_def[h_idx])

    # Clamp pour éviter overflow/underflow numérique
    lam_h = np.clip(lam_h, 1e-6, 20.0)
    lam_a = np.clip(lam_a, 1e-6, 20.0)

    log_p = (
        x * np.log(lam_h) - lam_h - scipy.special.gammaln(x + 1)
        + y * np.log(lam_a) - lam_a - scipy.special.gammaln(y + 1)
    )

    # Correction Dixon-Coles sur les faibles scores (0-0, 1-0, 0-1, 1-1)
    tau = np.ones(len(x))
    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
    tau[m10] = 1.0 + lam_a[m10] * rho
    tau[m01] = 1.0 + lam_h[m01] * rho
    tau[m11] = 1.0 - rho

    valid = tau > 1e-10
    log_tau = np.where(valid, np.log(np.maximum(tau, 1e-10)), -1e9)

    ll = (weights * (log_p + log_tau))[valid]
    return float(-ll.sum())


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit(ref_date: pd.Timestamp | None = None) -> dict:
    """
    Estime les paramètres Dixon-Coles par MLE.

    Identification : première équipe (ordre alphabétique) fixée à attack=1,
    defense=1. Tous les paramètres sont relatifs à cette référence.

    Retourne un dict avec :
      attack   : {team: float}  (1.0 = niveau moyen)
      defense  : {team: float}  (1.0 = niveau moyen ; plus bas = meilleure défense)
      home_adv : float
      rho      : float
    """
    ref_date = ref_date or pd.Timestamp.now().normalize()
    matches = _load_matches()

    all_teams = pd.concat([matches["home_team"], matches["away_team"]])
    team_counts = all_teams.value_counts()
    valid_teams = set(team_counts[team_counts >= 3].index)

    mask = matches["home_team"].isin(valid_teams) & matches["away_team"].isin(valid_teams)
    matches = matches[mask].reset_index(drop=True)

    teams = sorted(valid_teams)
    n = len(teams)
    t_idx = {t: i for i, t in enumerate(teams)}
    n_free = n - 1  # équipe 0 = référence fixe

    print(f"Estimation Dixon-Coles : {len(matches):,} matchs, {n} equipes")
    print(f"  Reference : '{teams[0]}'")

    h_idx = matches["home_team"].map(t_idx).values.astype(int)
    a_idx = matches["away_team"].map(t_idx).values.astype(int)
    x = matches["home_score"].values.astype(int)
    y = matches["away_score"].values.astype(int)
    is_neutral = matches.get("neutral", pd.Series(False, index=matches.index)).values.astype(bool)
    days_old = (ref_date - matches["date"]).dt.days.values
    weights = np.exp(-XI * days_old)

    x0 = np.zeros(2 * n_free + 2)
    x0[2 * n_free] = np.log(1.08)
    x0[2 * n_free + 1] = -0.08

    # Bornes pour stabilité numérique
    bounds = (
        [(-3.0, 3.0)] * n_free          # log_attack  [e^-3 ≈ 0.05 … e^3 ≈ 20]
        + [(-3.0, 3.0)] * n_free        # log_defense
        + [(np.log(0.8), np.log(1.6))]  # log_home_adv
        + [(-0.99, 0.5)]                # rho
    )

    print("Optimisation en cours...")
    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(n_free, h_idx, a_idx, x, y, is_neutral, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 15000, "ftol": 1e-10, "gtol": 1e-6},
    )
    if not result.success:
        print(f"  Avertissement optimiseur : {result.message}")

    log_att_free = result.x[:n_free]
    log_def_free = result.x[n_free : 2 * n_free]
    log_att = np.concatenate([[0.0], log_att_free])
    log_def = np.concatenate([[0.0], log_def_free])
    home_adv = float(np.exp(result.x[2 * n_free]))
    rho = float(result.x[2 * n_free + 1])

    attacks  = {t: float(np.exp(log_att[i])) for t, i in t_idx.items()}
    defenses = {t: float(np.exp(log_def[i])) for t, i in t_idx.items()}

    # Fallback: median rating over WC 2026 teams present in the model
    wc = _wc2026_teams()
    wc_present = [t for t in wc if t in attacks]
    fallback_attack  = float(np.median([attacks[t]  for t in wc_present])) if wc_present else 1.0
    fallback_defense = float(np.median([defenses[t] for t in wc_present])) if wc_present else 1.0

    return {
        "attack":           attacks,
        "defense":          defenses,
        "home_adv":         home_adv,
        "rho":              rho,
        "fallback_attack":  fallback_attack,
        "fallback_defense": fallback_defense,
        "n_matches":        int(len(matches)),
        "ref_date":         str(ref_date.date()),
        "ref_team":         teams[0],
    }


# ---------------------------------------------------------------------------
# Sauvegarde / chargement
# ---------------------------------------------------------------------------

def save(params: dict) -> None:
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"Parametres Poisson sauvegardes -> {PARAMS_PATH}")


def load() -> dict | None:
    """Charge depuis PERSISTENT_DIR en priorité, sinon depuis le fichier bundled."""
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, encoding="utf-8") as f:
            return json.load(f)
    if PARAMS_PATH_BUNDLED.exists():
        with open(PARAMS_PATH_BUNDLED, encoding="utf-8") as f:
            return json.load(f)
    return None


@lru_cache(maxsize=1)
def fit_or_load() -> dict:
    """
    Charge les paramètres depuis le disque si disponibles, sinon refit.
    Mis en cache pour ne pas relire/recalculer à chaque prédiction.
    """
    params = load()
    if params is None:
        print("poisson_params.json introuvable — ajustement du modele...")
        params = fit()
        save(params)
    return params


def refit_with_new_results() -> None:
    """
    Refit Dixon-Coles en intégrant les résultats WC 2026 depuis la DB.
    Sauvegarde dans PERSISTENT_DIR et invalide le cache.
    Appelé automatiquement après chaque /score admin.
    """
    print("[POISSON] Refit en cours avec les nouveaux resultats WC 2026...")
    params = fit()
    save(params)
    fit_or_load.cache_clear()
    print(f"[POISSON] Cache invalide — prochain /prono utilisera {params['n_matches']} matchs")


# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------

def predict_score(
    home_team: str,
    away_team: str,
    params: dict | None = None,
    is_neutral: bool = True,
    max_goals: int = 8,
) -> dict:
    """
    Calcule la distribution de scores et les probabilités de résultat.

    Retourne :
      most_likely_score : (home_goals, away_goals)
      p_home, p_draw, p_away : float
      lambda_home, lambda_away : float  (buts attendus)
      score_matrix : np.ndarray (max_goals+1, max_goals+1)
    """
    if params is None:
        params = fit_or_load()

    attacks  = params["attack"]
    defenses = params["defense"]
    home_adv = params["home_adv"]
    rho      = params["rho"]

    fb_att = params.get("fallback_attack", 1.0)
    fb_def = params.get("fallback_defense", 1.0)
    att_h = attacks.get(home_team, fb_att)
    def_h = defenses.get(home_team, fb_def)
    att_a = attacks.get(away_team, fb_att)
    def_a = defenses.get(away_team, fb_def)

    adv = 1.0 if is_neutral else home_adv
    lam_h = att_h * def_a * adv
    lam_a = att_a * def_h

    g = np.arange(max_goals + 1)
    p_h = poisson.pmf(g, lam_h)
    p_a = poisson.pmf(g, lam_a)
    matrix = np.outer(p_h, p_a)

    # Correction Dixon-Coles sur les 4 premiers scores
    for xi, yi, fn in [
        (0, 0, lambda lh, la: 1.0 - lh * la * rho),
        (1, 0, lambda lh, la: 1.0 + la * rho),
        (0, 1, lambda lh, la: 1.0 + lh * rho),
        (1, 1, lambda lh, la: 1.0 - rho),
    ]:
        matrix[xi, yi] *= fn(lam_h, lam_a)

    matrix = np.clip(matrix, 0, None)
    matrix /= matrix.sum()

    p_home = float(np.tril(matrix, -1).sum())   # home goals > away goals
    p_away = float(np.triu(matrix, 1).sum())    # away goals > home goals
    p_draw = float(np.diag(matrix).sum())

    best = np.unravel_index(np.argmax(matrix), matrix.shape)

    return {
        "most_likely_score": (int(best[0]), int(best[1])),
        "p_home":            round(p_home, 4),
        "p_draw":            round(p_draw, 4),
        "p_away":            round(p_away, 4),
        "lambda_home":       round(lam_h, 3),
        "lambda_away":       round(lam_a, 3),
        "score_matrix":      matrix,
    }


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = fit()
    save(params)

    print(f"\nhome_adv = {params['home_adv']:.3f}  rho = {params['rho']:.4f}")
    print(f"Matchs utilises : {params['n_matches']}")

    print("\nTop 10 attaques :")
    top_att = sorted(params["attack"].items(), key=lambda x: x[1], reverse=True)[:10]
    for team, val in top_att:
        print(f"  {team:<30} {val:.3f}")

    print("\nTop 10 defenses (plus bas = meilleur) :")
    top_def = sorted(params["defense"].items(), key=lambda x: x[1])[:10]
    for team, val in top_def:
        print(f"  {team:<30} {val:.3f}")

    print("\nExemples de predictions :")
    tests = [
        ("France", "Argentina"),
        ("Brazil", "Germany"),
        ("Morocco", "Spain"),
    ]
    for home, away in tests:
        r = predict_score(home, away, params)
        s = r["most_likely_score"]
        print(
            f"  {home} vs {away} -> {s[0]}-{s[1]}  "
            f"[dom {r['p_home']:.0%} / nul {r['p_draw']:.0%} / ext {r['p_away']:.0%}]"
            f"  lambda {r['lambda_home']:.2f} - {r['lambda_away']:.2f}"
        )
