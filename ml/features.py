"""
ml/features.py — Feature engineering pipeline.

Entrées  : ml/data/results.csv, ml/data/elo_history.csv
Sortie   : DataFrame (+ ml/data/features.csv si __main__)
Colonnes : elo_home, elo_away, elo_diff,
           home/away_form_pts/gf/ga/n,
           h2h_home_pts/gd/n,
           is_neutral, tournament_tier,
           home_is_host, away_is_host
           result (target: 0=away, 1=draw, 2=home)

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

FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "home_form_pts", "home_form_gf", "home_form_ga", "home_form_n",
    "away_form_pts", "away_form_gf", "away_form_ga", "away_form_n",
    "h2h_home_pts", "h2h_gd", "h2h_n",
    "is_neutral", "tournament_tier",
    "home_is_host", "away_is_host",
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
# Source unique et auditables — pas de jointure externe.
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

    # If a team plays twice on the same date, keep the first entry (= pre-day ELO)
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
    Vectorized form features: stats over the last n matches *before* each match.

    Trick: shift(1) within each team group excludes the current match from
    the rolling window, so the window is strictly in the past.
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

    # Normalize points to [0, 1] — win rate equivalent
    combined["form_pts"] = np.where(
        combined["form_n"] > 0,
        combined["form_pts_raw"] / (3 * combined["form_n"]),
        0.0,
    )

    form = (
        combined[["team", "date", "form_pts", "form_gf", "form_ga", "form_n"]]
        .groupby(["team", "date"], as_index=False)
        .first()
    )

    home_form = (
        df[["date", "home_team"]]
        .merge(form.rename(columns={"team": "home_team"}), on=["date", "home_team"], how="left")
        .rename(columns={"form_pts": "home_form_pts", "form_gf": "home_form_gf",
                         "form_ga": "home_form_ga", "form_n": "home_form_n"})
        [["home_form_pts", "home_form_gf", "home_form_ga", "home_form_n"]]
        .fillna(0)
        .reset_index(drop=True)
    )
    away_form = (
        df[["date", "away_team"]]
        .merge(form.rename(columns={"team": "away_team"}), on=["date", "away_team"], how="left")
        .rename(columns={"form_pts": "away_form_pts", "form_gf": "away_form_gf",
                         "form_ga": "away_form_ga", "form_n": "away_form_n"})
        [["away_form_pts", "away_form_gf", "away_form_ga", "away_form_n"]]
        .fillna(0)
        .reset_index(drop=True)
    )
    return pd.concat([home_form, away_form], axis=1)


def _h2h_features(df: pd.DataFrame, n: int = H2H_WINDOW) -> pd.DataFrame:
    """
    H2H features using a pre-indexed dict for O(log k) lookup per match pair.
    One iteration to build the index, one to compute features.
    """
    # Build index: frozenset({team1, team2}) → sorted DataFrame of their meetings
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

        # searchsorted gives first position >= dt → all positions before are strictly < dt
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
    home_is_host : 1 si le home_team est la nation hôte de cette CdM, 0 sinon.
    away_is_host : 1 si le away_team est la nation hôte de cette CdM, 0 sinon.

    Implémentation : lookup dict (year, team) → 1, vectorisé sur tout le DataFrame.
    Aucun fichier externe — auditez WC_HOSTS directement dans ce module.
    """
    # Précalcule un dict plat (year, team) → 1 pour éviter les appels de set par ligne
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


def build_features(
    results_path: str | Path | None = None,
    elo_history_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix.

    Parameters
    ----------
    results_path     : path to results.csv  (default: ml/data/results.csv)
    elo_history_path : path to elo_history.csv (default: ml/data/elo_history.csv)

    Returns
    -------
    DataFrame, one row per historical match, ready for train.py.
    """
    results_path     = Path(results_path     or DATA_DIR / "results.csv")
    elo_history_path = Path(elo_history_path or DATA_DIR / "elo_history.csv")

    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    elo = pd.read_csv(elo_history_path, parse_dates=["date"])

    print(f"Loaded {len(df):,} matches (scores disponibles)")

    df = _elo_before_match(df, elo)
    print("ELO features done")

    form_df = _form_features(df)
    print("Form features done")

    print("Computing H2H features (one loop pass)...")
    h2h_df = _h2h_features(df)
    print("H2H features done")

    host_df = _host_features(df)
    n_host = host_df["home_is_host"].sum() + host_df["away_is_host"].sum()
    print(f"Host features done — {n_host} matchs avec un hôte identifié")

    df = pd.concat([df.reset_index(drop=True), form_df, h2h_df, host_df], axis=1)

    df["is_neutral"]      = (df["neutral"].astype(str).str.upper() == "TRUE").astype(int)
    df["tournament_tier"] = df["tournament"].map(get_tournament_tier)

    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        [2, 1], default=0,
    )

    feature_cols = [
        "date", "home_team", "away_team", "tournament",
        "elo_home", "elo_away", "elo_diff",
        "home_form_pts", "home_form_gf", "home_form_ga", "home_form_n",
        "away_form_pts", "away_form_gf", "away_form_ga", "away_form_n",
        "h2h_home_pts", "h2h_gd", "h2h_n",
        "is_neutral", "tournament_tier",
        "home_is_host", "away_is_host",
        "result",
    ]
    return df[feature_cols]


if __name__ == "__main__":
    import time

    t0 = time.time()
    features = build_features()
    elapsed  = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s — {len(features):,} rows × {features.shape[1]} columns")
    print(features.dtypes.to_string())
    print(features.head(3).to_string())

    out = DATA_DIR / "features.csv"
    features.to_csv(out, index=False)
    print(f"\nSaved → {out}")
