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


STAGE_SHORT = {
    "Group Stage":    "Groupes",
    "Round of 32":    "R32",
    "Round of 16":    "R16",
    "Quarter Finals": "QF",
    "Semi Finals":    "SF",
    "Finals":         "Finale",
}
STAGE_ORDER = ["Group Stage", "Round of 32", "Round of 16", "Quarter Finals", "Semi Finals", "Finals"]
CM_LABELS   = {"H": "Dom.", "D": "Nul", "A": "Ext."}


def _perf_data() -> dict:
    # Accuracy curve: one point per resolved match ordered by date
    with database.get_connection() as conn:
        rows = conn.execute("""
            SELECT r.match_date, p.is_correct_result, p.predicted_result, p.actual_result
            FROM predictions p
            JOIN match_results r ON r.match_id = p.match_id
            WHERE p.actual_result IS NOT NULL
            ORDER BY r.match_date, p.match_id
        """).fetchall()

    dates, cum_acc, correct = [], [], 0
    for i, (date, is_ok, _pred, _act) in enumerate(rows):
        correct += (is_ok or 0)
        dates.append(date[:10])
        cum_acc.append(round(correct / (i + 1) * 100, 1))

    # Confusion matrix [predicted][actual] with labels H/D/A
    cm: dict[tuple[str, str], int] = {}
    for _, _, pred, actual in rows:
        key = (pred, actual)
        cm[key] = cm.get(key, 0) + 1

    lbl = list(CM_LABELS.keys())
    matrix = [
        {"label": CM_LABELS[p], "vals": [cm.get((p, a), 0) for a in lbl]}
        for p in lbl
    ]
    col_labels = list(CM_LABELS.values())
    max_val = max((v for row in matrix for v in row["vals"]), default=1) or 1

    # Per-stage accuracy
    df = pd.read_csv(FIXTURES_PATH)
    stage_data = []
    for stage in STAGE_ORDER:
        sub = df[df["stage"] == stage]
        if sub.empty:
            continue
        ids = tuple(200_000 + int(mn) for mn in sub["match_number"])
        placeholders = ",".join("?" * len(ids))
        with database.get_connection() as conn:
            r = conn.execute(
                f"SELECT COUNT(*), SUM(is_correct_result) FROM predictions "
                f"WHERE match_id IN ({placeholders}) AND actual_result IS NOT NULL",
                ids,
            ).fetchone()
        if r[0] > 0:
            stage_data.append({
                "stage": STAGE_SHORT.get(stage, stage),
                "total": r[0],
                "correct": r[1] or 0,
                "rate": round((r[1] or 0) / r[0] * 100, 1),
            })

    return {
        "stats":      database.get_stats(),
        "dates":      dates,
        "cum_acc":    cum_acc,
        "matrix":     matrix,
        "col_labels": col_labels,
        "max_val":    max_val,
        "stage_data": stage_data,
    }


RESULT_FR = {"H": "Dom.", "D": "Nul", "A": "Ext."}


def _standings_data() -> list[dict]:
    df    = pd.read_csv(FIXTURES_PATH)
    gs    = df[df["stage"] == "Group Stage"]

    # Init all teams from fixtures
    groups: dict[str, dict[str, dict]] = {}
    for grp in sorted(gs["group"].dropna().unique()):
        sub   = gs[gs["group"] == grp]
        teams = sorted({
            t for t in sub["home_team"].tolist() + sub["away_team"].tolist()
            if t != "To be announced"
        })
        groups[grp] = {
            t: {"team": t, "p": 0, "w": 0, "d": 0, "l": 0,
                "gf": 0, "ga": 0, "pts": 0}
            for t in teams
        }

    # Fill from DB results
    with database.get_connection() as conn:
        rows = conn.execute(
            "SELECT home_team, away_team, home_score, away_score, match_group "
            "FROM match_results WHERE match_group IS NOT NULL"
        ).fetchall()

    for home, away, hs, as_, grp in rows:
        if grp not in groups:
            continue
        for t in (home, away):
            if t not in groups[grp]:
                groups[grp][t] = {"team": t, "p": 0, "w": 0, "d": 0, "l": 0,
                                   "gf": 0, "ga": 0, "pts": 0}
        h, a = groups[grp][home], groups[grp][away]
        h["p"] += 1;  a["p"] += 1
        h["gf"] += hs; h["ga"] += as_
        a["gf"] += as_; a["ga"] += hs
        if hs > as_:
            h["w"] += 1; h["pts"] += 3; a["l"] += 1
        elif hs < as_:
            a["w"] += 1; a["pts"] += 3; h["l"] += 1
        else:
            h["d"] += 1; h["pts"] += 1; a["d"] += 1; a["pts"] += 1

    result = []
    for grp in sorted(groups.keys()):
        teams = list(groups[grp].values())
        for t in teams:
            t["gd"] = t["gf"] - t["ga"]
            t["gd_str"] = f"+{t['gd']}" if t["gd"] > 0 else str(t["gd"])
        teams.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
        for i, t in enumerate(teams):
            t["rank"] = i + 1
            t["q"] = i < 2   # top 2 advance (simplified)
        result.append({"group": grp, "teams": teams})

    return result


def _history_data() -> dict:
    with database.get_connection() as conn:
        resolved = conn.execute("""
            SELECT p.home_team, p.away_team,
                   p.predicted_result, p.actual_result,
                   p.is_correct_result, p.is_correct_score,
                   r.home_score, r.away_score, r.match_date, r.match_group
            FROM predictions p
            JOIN match_results r ON r.match_id = p.match_id
            WHERE p.actual_result IS NOT NULL
            ORDER BY r.match_date DESC, r.match_id DESC
        """).fetchall()

    rows = []
    for home, away, pred, actual, ok_r, ok_s, hs, as_, date, grp in resolved:
        rows.append({
            "home":     home,
            "away":     away,
            "pred":     RESULT_FR.get(pred, pred),
            "pred_raw": pred,
            "actual":   RESULT_FR.get(actual, actual),
            "actual_raw": actual,
            "score":    f"{hs}–{as_}",
            "ok":       bool(ok_r),
            "ok_score": bool(ok_s),
            "date":     _fmt_date(date),
            "group":    grp or "",
        })
    return {"rows": rows, "stats": database.get_stats()}


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


@router.get("/standings", response_class=HTMLResponse)
def standings(request: Request):
    return templates.TemplateResponse("public/standings.html", {
        "request": request,
        "groups":  _standings_data(),
    })


@router.get("/history", response_class=HTMLResponse)
def history(request: Request):
    data = _history_data()
    return templates.TemplateResponse("public/history.html", {
        "request": request,
        **data,
    })


@router.get("/stats", response_class=HTMLResponse)
def stats(request: Request):
    data = _perf_data()
    return templates.TemplateResponse("public/stats.html", {"request": request, **data})


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
