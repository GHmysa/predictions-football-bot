from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import database
from ml.predict import predict_match

FIXTURES_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"
ELO_HISTORY   = Path(__file__).parent.parent / "ml" / "data" / "elo_history.csv"
WC_TEAMS_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_teams.csv"

import os
_PERSISTENT_DIR = Path(os.environ.get("PERSISTENT_DIR", Path(__file__).parent.parent / "ml" / "data"))
WC_ELO_PATH = _PERSISTENT_DIR / "wc_elo_updates.csv"

router    = APIRouter()
templates = Jinja2Templates(directory="templates")

KO_STAGES = ["Round of 32", "Round of 16", "Quarter Finals", "Semi Finals", "Finals"]
KO_LABEL  = {
    "Round of 32":    "1/32 de finale",
    "Round of 16":    "1/16 de finale",
    "Quarter Finals": "Quarts de finale",
    "Semi Finals":    "Demi-finales",
    "Finals":         "Finale",
}


def _fmt_date(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d %b")
    except Exception:
        return date_str


def _upcoming_matches(n: int = 8) -> list[dict]:
    df = pd.read_csv(FIXTURES_PATH)
    today = pd.Timestamp.today().normalize().isoformat()[:10]
    mask = (
        (df["date"] >= today) &
        (df["home_team"] != "To be announced") &
        (df["away_team"] != "To be announced")
    )
    rows = df[mask].sort_values("date").head(n)
    out = []
    for _, m in rows.iterrows():
        try:
            r  = predict_match(str(m["home_team"]), str(m["away_team"]), str(m["date"]), True, 4)
            p  = r["probabilities"]
            ph = round(p["home"] * 100)
            pd_ = round(p["draw"] * 100)
            pa = 100 - ph - pd_
            out.append({
                "match_number": int(m["match_number"]),
                "date":         _fmt_date(str(m["date"])),
                "home":         str(m["home_team"]),
                "away":         str(m["away_team"]),
                "stage":        KO_LABEL.get(str(m["stage"]), str(m["stage"])),
                "ph": ph, "pd": pd_, "pa": pa,
                "score_home": r.get("predicted_score_home", "?"),
                "score_away": r.get("predicted_score_away", "?"),
                "prediction": r["prediction"],
            })
        except Exception:
            pass
    return out


def _recent_results(n: int = 6) -> list[dict]:
    with database.get_connection() as conn:
        rows = conn.execute("""
            SELECT home_team, away_team, home_score, away_score, match_date, match_group
            FROM match_results ORDER BY match_date DESC LIMIT ?
        """, (n,)).fetchall()
    return [
        {"home": r[0], "away": r[1], "hs": r[2], "as_": r[3],
         "date": _fmt_date(r[4]), "stage": r[5] or ""}
        for r in rows
    ]


def _ko_results() -> dict[int, dict]:
    with database.get_connection() as conn:
        rows = conn.execute(
            "SELECT match_id, home_score, away_score FROM match_results WHERE match_id > 200072"
        ).fetchall()
    return {r[0]: {"hs": r[1], "as_": r[2]} for r in rows}


def _bracket_data() -> list[dict]:
    df      = pd.read_csv(FIXTURES_PATH)
    ko      = df[df["stage"] != "Group Stage"].sort_values("match_number")
    results = _ko_results()
    stages  = []
    for stage in KO_STAGES:
        sdf = ko[ko["stage"] == stage]
        if sdf.empty:
            continue
        matches = []
        for _, m in sdf.iterrows():
            mid  = 200_000 + int(m["match_number"])
            home = str(m["home_team"])
            away = str(m["away_team"])
            dt   = _fmt_date(str(m["date"]))
            tba  = home == "To be announced" or away == "To be announced"
            if mid in results:
                r = results[mid]
                draw = r["hs"] == r["as_"]
                matches.append({
                    "home": home, "away": away, "date": dt,
                    "played": True, "tba": False,
                    "hs": r["hs"], "as_": r["as_"],
                    "draw": draw,
                    "home_won": not draw and r["hs"] > r["as_"],
                })
            else:
                matches.append({
                    "home": home, "away": away, "date": dt,
                    "played": False, "tba": tba,
                })
        stages.append({"label": KO_LABEL.get(stage, stage), "matches": matches})
    return stages


def _elo_data() -> list[dict]:
    wc_teams = pd.read_csv(WC_TEAMS_PATH)
    elo      = pd.read_csv(ELO_HISTORY, parse_dates=["date"])
    if WC_ELO_PATH.exists() and WC_ELO_PATH.stat().st_size > 0:
        wc_elo = pd.read_csv(WC_ELO_PATH, parse_dates=["date"])
        elo = pd.concat([elo, wc_elo], ignore_index=True).sort_values("date")
    out = []
    for _, t in wc_teams.iterrows():
        sub = elo[elo["team"] == t["dataset_name"]]
        if not sub.empty:
            out.append({
                "name": t["fifa_name"],
                "elo":  round(float(sub.iloc[-1]["elo"]), 0),
            })
    return sorted(out, key=lambda x: -x["elo"])


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    matches = _upcoming_matches(8)
    recent  = _recent_results(6)
    stats   = database.get_stats()
    return templates.TemplateResponse("public/home.html", {
        "request": request,
        "matches": matches,
        "recent":  recent,
        "stats":   stats,
    })


@router.get("/bracket", response_class=HTMLResponse)
def bracket(request: Request):
    stages = _bracket_data()
    return templates.TemplateResponse("public/bracket.html", {
        "request": request,
        "stages":  stages,
    })


@router.get("/elo", response_class=HTMLResponse)
def elo(request: Request):
    teams = _elo_data()
    return templates.TemplateResponse("public/elo.html", {
        "request": request,
        "teams":   teams,
    })


@router.get("/api/elo")
def api_elo():
    return JSONResponse(_elo_data())


@router.get("/api/match/{match_number}")
def api_match(match_number: int):
    df  = pd.read_csv(FIXTURES_PATH)
    row = df[df["match_number"] == match_number]
    if row.empty:
        return JSONResponse({"error": "not found"}, status_code=404)
    m = row.iloc[0]
    return JSONResponse({
        "home":  str(m["home_team"]),
        "away":  str(m["away_team"]),
        "date":  str(m["date"]),
        "stage": KO_LABEL.get(str(m["stage"]), str(m["stage"])),
    })
