"""
ml/features.py — Feature engineering pipeline.

Entrées  : ml/data/results.csv, ml/data/elo_history.csv
Sortie   : DataFrame (+ ml/data/features.csv si __main__)

Features (20) :
  elo_home, elo_away, elo_diff
  home/away_form_pts, home/away_form_gf, home/away_form_ga   (× 2 équipes, sans _n)
  h2h_home_pts, h2h_gd, h2h_n
  is_neutral, tournament_tier
  home_is_host, away_is_host
  home/away_wc_form_pts                                       (forme en tournois majeurs)
  home/away_rest_days                                         (jours depuis dernier match)
  result  (target: 0=away, 1=draw, 2=home)

Exécution : python -m ml.features   (depuis predictions-football-bot/)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path(__file__).parent / "data"
FORM_WINDOW = 5
H2H_WINDOW  = 5
ELO_INIT    = 1500.0
REST_CAP    = 30   # jours max pour rest_days (au-delà : même signal "bien reposé")

FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "home_form_pts", "home_form_gf", "home_form_ga",
    "away_form_pts", "away_form_gf", "away_form_ga",
    "h2h_home_pts", "h2h_gd", "h2h_n",
    "is_neutral", "tournament_tier",
    "home_is_host", "away_is_host",
    "home_wc_form_pts", "away_wc_form_pts",
    "home_rest_days", "away_rest_days",
]

_TIER_MAP: dict[int, list[str]] = {
    4: ["FIFA World Cup"],
    3: [
        "UEFA Euro", "Copa América",
        "Africa Cup of Nations", "African Cup of Nations",
        "AFC Asian Cup", "Asian Cup",
        "CONCACAF Gold Cup", "Gold Cup",
        "Oceania Nations Cup",
    ],
}
_QUAL_KEYWORDS = ("qualification", "qualifier", "Qualification", "Qualifier")

# Noms d'équipes hôtes tels qu'ils apparaissent dans results.csv, par édition CdM.
WC_HOSTS: dict[int, set[str]] = {
    1930: {"Uruguay"},
    1934: {"Italy"},
    1938: {"France"},
    1950: {"Brazil"},
    1954: {"Switzerland"},
    1958: {"Sweden"},
    1962: {"Chile"},
    1966: {"England"},
    1970: {"Mexico"},
    1974: {"West Germany"},
    1978: {"Argentina"},
    1982: {"Spain"},
    1986: {"Mexico"},
    1990: {"Italy"},
    1994: {"United States"},
    1998: {"France"},
    2002: {"South Korea", "Japan"},
    2006: {"Germany"},
    2010: {"South Africa"},
    2014: {"Brazil"},
    2018: {"Russia"},
    2022: {"Qatar"},
    2026: {"United States", "Canada", "Mexico"},
}


def get_tournament_tier(tournament: str) -> int:
    """4=World Cup, 3=Continental, 2=Qualification, 1=Other."""
    if any(kw in tournament for kw in _QUAL_KEYWORDS):
        return 2
    for tier, keywords in _TIER_MAP.items():
        if any(kw in tournament for kw in keywords):
            return tier
    return 1


def _elo_before_match(df: pd.DataFrame, elo_history: pd.DataFrame) -> pd.DataFrame:
    """
    Adds elo_home, elo_away, elo_diff.

    elo_history stores ELO *after* each match. We shift by 1 within each team
    group to get the ELO *before* the current match. Teams with no prior history
    start at ELO_INIT (1500).
    """
    elo = elo_history.sort_values(["team", "date"]).copy()
    elo["elo_pre"] = elo.groupby("team")["elo"].shift(1).fillna(ELO_INIT)

    elo_pre = (
        elo.groupby(["team", "date"], as_index=False)
        .first()[["team", "date", "elo_pre"]]
    )

    df = df.merge(
        elo_pre.rename(columns={"team": "home_team", "elo_pre": "elo_home"}),
        on=["date", "home_team"], how="left",
    ).merge(
        elo_pre.rename(columns={"team": "away_team", "elo_pre": "elo_away"}),
        on=["date", "away_team"], how="left",
    )
    df["elo_home"] = df["elo_home"].fillna(ELO_INIT)
    df["elo_away"] = df["elo_away"].fillna(ELO_INIT)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df


def _form_features(df: pd.DataFrame, n: int = FORM_WINDOW) -> pd.DataFrame:
    """
    Form over last n matches before each match: pts (normalized), gf, ga.
    form_n supprimé de FEATURE_COLS — gardé temporairement pour compatibilité interne.

    Trick anti-leakage : shift(1) exclut le match courant du rolling window.
    """
    home = df[["date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "gf", "away_score": "ga"}
    )
    away = df[["date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "gf", "home_score": "ga"}
    )
    combined = pd.concat([home, away], ignore_index=True)
    combined["pts"] = np.select(
        [combined["gf"] > combined["ga"], combined["gf"] == combined["ga"]],
        [3, 1], default=0,
    )
    combined = combined.sort_values(["team", "date"]).reset_index(drop=True)

    grp = combined.groupby("team")

    def _roll(s: pd.Series):
        return s.shift(1).rolling(n, min_periods=1)

    combined["form_pts_raw"] = grp["pts"].transform(lambda s: _roll(s).sum().fillna(0))
    combined["form_n"]       = grp["pts"].transform(lambda s: _roll(s).count().fillna(0)).astype(int)
    combined["form_gf"]      = grp["gf"].transform(lambda s: _roll(s).mean().fillna(0))
    combined["form_ga"]      = grp["ga"].transform(lambda s: _roll(s).mean().fillna(0))
    combined["form_pts"]     = np.where(
        combined["form_n"] > 0,
        combined["form_pts_raw"] / (3 * combined["form_n"]),
        0.0,
    )

    form = (
        combined[["team", "date", "form_pts", "form_gf", "form_ga"]]
        .groupby(["team", "date"], as_index=False)
        .first()
    )

    home_form = (
        df[["date", "home_team"]]
        .merge(form.rename(columns={"team": "home_team"}), on=["date", "home_team"], how="left")
        .rename(columns={"form_pts": "home_form_pts", "form_gf": "home_form_gf",
                         "form_ga": "home_form_ga"})
        [["home_form_pts", "home_form_gf", "home_form_ga"]]
        .fillna(0)
        .reset_index(drop=True)
    )
    away_form = (
        df[["date", "away_team"]]
        .merge(form.rename(columns={"team": "away_team"}), on=["date", "away_team"], how="left")
        .rename(columns={"form_pts": "away_form_pts", "form_gf": "away_form_gf",
                         "form_ga": "away_form_ga"})
        [["away_form_pts", "away_form_gf", "away_form_ga"]]
        .fillna(0)
        .reset_index(drop=True)
    )
    return pd.concat([home_form, away_form], axis=1)


def _h2h_features(df: pd.DataFrame, n: int = H2H_WINDOW) -> pd.DataFrame:
    """
    H2H features using a pre-indexed dict for O(log k) lookup per match pair.
    """
    raw: dict[tuple, list] = {}
    for dt, home, away, hs, as_ in zip(
        df["date"], df["home_team"], df["away_team"], df["home_score"], df["away_score"]
    ):
        key = tuple(sorted([home, away]))
        raw.setdefault(key, []).append(
            {"date": dt, "home_team": home, "home_score": hs, "away_score": as_}
        )
    index = {
        k: pd.DataFrame(v).sort_values("date").reset_index(drop=True)
        for k, v in raw.items()
    }

    rows = []
    for dt, home, away in zip(df["date"], df["home_team"], df["away_team"]):
        key = tuple(sorted([home, away]))
        grp = index.get(key)

        if grp is None:
            rows.append({"h2h_home_pts": 0.0, "h2h_gd": 0.0, "h2h_n": 0})
            continue

        idx  = int(grp["date"].searchsorted(dt, side="left"))
        past = grp.iloc[max(0, idx - n): idx]
        m    = len(past)

        if m == 0:
            rows.append({"h2h_home_pts": 0.0, "h2h_gd": 0.0, "h2h_n": 0})
            continue

        is_home = (past["home_team"] == home).values
        gf = np.where(is_home, past["home_score"].values, past["away_score"].values).astype(float)
        ga = np.where(is_home, past["away_score"].values, past["home_score"].values).astype(float)
        pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))

        rows.append({
            "h2h_home_pts": float(pts.sum()) / (3 * m),
            "h2h_gd":       float((gf - ga).mean()),
            "h2h_n":        m,
        })

    return pd.DataFrame(rows)


def _host_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    home_is_host / away_is_host : 1 si l'équipe est la nation hôte de cette CdM.
    Lookup dict vectorisé — aucun fichier externe.
    """
    host_lookup: dict[tuple[int, str], int] = {}
    for yr, teams in WC_HOSTS.items():
        for t in teams:
            host_lookup[(yr, t)] = 1

    year = df["date"].dt.year
    home_is_host = [host_lookup.get((y, h), 0) for y, h in zip(year, df["home_team"])]
    away_is_host = [host_lookup.get((y, a), 0) for y, a in zip(year, df["away_team"])]

    return pd.DataFrame({
        "home_is_host": np.array(home_is_host, dtype=np.int8),
        "away_is_host": np.array(away_is_host, dtype=np.int8),
    })


def _wc_form_features(df: pd.DataFrame, n: int = FORM_WINDOW) -> pd.DataFrame:
    """
    Win rate dans les n derniers matchs de tournoi majeur (tier >= 3) avant chaque match.

    Différent de form_pts (toutes compétitions) : un 4-0 en amical et un 1-0 en finale
    de Copa América ont le même poids dans form_pts, pas ici.

    Implémentation : merge_asof (jointure temporelle "as-of") pour récupérer la
    wc_form la plus récente avant chaque date de match, même si le match courant
    n'est pas lui-même un tournoi majeur.
    """
    tier = df["tournament"].map(get_tournament_tier)
    major_df = df[tier >= 3]

    home_m = major_df[["date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "gf", "away_score": "ga"})
    away_m = major_df[["date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "gf", "home_score": "ga"})

    major = pd.concat([home_m, away_m], ignore_index=True)
    major["pts"] = np.select(
        [major["gf"] > major["ga"], major["gf"] == major["ga"]], [3, 1], default=0)
    major = major.sort_values(["team", "date"]).reset_index(drop=True)

    grp = major.groupby("team")

    def _roll(s: pd.Series):
        return s.shift(1).rolling(n, min_periods=1)

    wc_n       = grp["pts"].transform(lambda s: _roll(s).count().fillna(0)).astype(int)
    wc_pts_raw = grp["pts"].transform(lambda s: _roll(s).sum().fillna(0))
    major["wc_form_pts"] = np.where(wc_n > 0, wc_pts_raw / (3 * wc_n), 0.0)

    # Une entrée par (team, date) pour les tournois majeurs seulement
    wc_form = (
        major[["team", "date", "wc_form_pts"]]
        .groupby(["team", "date"], as_index=False)
        .first()
        .sort_values("date")   # merge_asof exige le tri par la clé temporelle
    )

    # merge_asof : pour chaque match dans df, trouve la wc_form la plus récente ≤ date
    base = df[["date", "home_team", "away_team"]].copy()
    base["_idx"] = np.arange(len(base))
    base = base.sort_values("date")

    home_j = pd.merge_asof(
        base[["date", "home_team", "_idx"]],
        wc_form.rename(columns={"team": "home_team", "wc_form_pts": "home_wc_form_pts"}),
        on="date", by="home_team", direction="backward",
    ).set_index("_idx").sort_index()["home_wc_form_pts"].fillna(0)

    away_j = pd.merge_asof(
        base[["date", "away_team", "_idx"]],
        wc_form.rename(columns={"team": "away_team", "wc_form_pts": "away_wc_form_pts"}),
        on="date", by="away_team", direction="backward",
    ).set_index("_idx").sort_index()["away_wc_form_pts"].fillna(0)

    return pd.DataFrame({
        "home_wc_form_pts": home_j.values,
        "away_wc_form_pts": away_j.values,
    })


def _rest_days_features(df: pd.DataFrame, cap: int = REST_CAP) -> pd.DataFrame:
    """
    Jours depuis le dernier match pour chaque équipe, plafonné à `cap` jours.

    Au-delà de cap jours, le signal "reposé" est le même qu'on soit à 31 ou 180 jours.
    Valeur initiale (premier match d'une équipe) : cap (neutre).

    Même implémentation merge_asof que wc_form : on calcule les rest_days pour chaque
    apparition d'une équipe, puis on joint en temporel sur le DataFrame complet.
    """
    home_app = df[["date", "home_team"]].rename(columns={"home_team": "team"})
    away_app = df[["date", "away_team"]].rename(columns={"away_team": "team"})
    apps = (
        pd.concat([home_app, away_app])
        .drop_duplicates()
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )

    apps["prev_date"] = apps.groupby("team")["date"].shift(1)
    apps["rest_days"] = (
        (apps["date"] - apps["prev_date"]).dt.days
        .clip(upper=cap)
        .fillna(cap)
        .astype(int)
    )

    rest = (
        apps.groupby(["team", "date"], as_index=False)["rest_days"]
        .first()
        .sort_values("date")
    )

    base = df[["date", "home_team", "away_team"]].copy()
    base["_idx"] = np.arange(len(base))
    base = base.sort_values("date")

    home_r = pd.merge_asof(
        base[["date", "home_team", "_idx"]],
        rest.rename(columns={"team": "home_team", "rest_days": "home_rest_days"}),
        on="date", by="home_team", direction="backward",
    ).set_index("_idx").sort_index()["home_rest_days"].fillna(cap).astype(int)

    away_r = pd.merge_asof(
        base[["date", "away_team", "_idx"]],
        rest.rename(columns={"team": "away_team", "rest_days": "away_rest_days"}),
        on="date", by="away_team", direction="backward",
    ).set_index("_idx").sort_index()["away_rest_days"].fillna(cap).astype(int)

    return pd.DataFrame({
        "home_rest_days": home_r.values,
        "away_rest_days": away_r.values,
    })


def build_features(
    results_path: str | Path | None = None,
    elo_history_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Construit la matrice de features complète.

    Paramètres
    ----------
    results_path     : chemin vers results.csv  (défaut : ml/data/results.csv)
    elo_history_path : chemin vers elo_history.csv (défaut : ml/data/elo_history.csv)
    """
    results_path     = Path(results_path     or DATA_DIR / "results.csv")
    elo_history_path = Path(elo_history_path or DATA_DIR / "elo_history.csv")

    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    elo = pd.read_csv(elo_history_path, parse_dates=["date"])

    print(f"Loaded {len(df):,} matches")

    df = _elo_before_match(df, elo)
    print("ELO done")

    form_df = _form_features(df)
    print("Form done")

    print("H2H (one pass)...")
    h2h_df = _h2h_features(df)
    print("H2H done")

    host_df = _host_features(df)
    print(f"Host done — {host_df['home_is_host'].sum() + host_df['away_is_host'].sum()} host matches")

    wc_form_df = _wc_form_features(df)
    print("WC form done")

    rest_df = _rest_days_features(df)
    print("Rest days done")

    df = pd.concat(
        [df.reset_index(drop=True), form_df, h2h_df, host_df, wc_form_df, rest_df],
        axis=1,
    )

    df["is_neutral"]      = (df["neutral"].astype(str).str.upper() == "TRUE").astype(int)
    df["tournament_tier"] = df["tournament"].map(get_tournament_tier)
    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        [2, 1], default=0,
    )

    out_cols = [
        "date", "home_team", "away_team", "tournament",
        *FEATURE_COLS,
        "result",
    ]
    return df[out_cols]


if __name__ == "__main__":
    import time

    t0 = time.time()
    features = build_features()
    elapsed  = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s — {len(features):,} rows × {features.shape[1]} cols")
    print(features[FEATURE_COLS].describe().round(3).to_string())

    out = DATA_DIR / "features.csv"
    features.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
