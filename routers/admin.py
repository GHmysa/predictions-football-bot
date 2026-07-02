from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

import pandas as pd
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

import database

FIXTURES_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"
METRICS_PATH  = Path(__file__).parent.parent / "ml" / "metrics.json"

ADMIN_USER     = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

router    = APIRouter(prefix="/admin")
security  = HTTPBasic()
templates = Jinja2Templates(directory="templates")


def require_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    ok = secrets.compare_digest(
        credentials.password.encode(), ADMIN_PASSWORD.encode()
    ) and secrets.compare_digest(
        credentials.username.encode(), ADMIN_USER.encode()
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ── Overview ─────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def overview(request: Request, _: str = Depends(require_admin)):
    stats = database.get_stats()
    with database.get_connection() as conn:
        n_results = conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
        n_preds   = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    from ml.poisson import fit_or_load
    params = fit_or_load()
    return templates.TemplateResponse("admin/overview.html", {
        "request":   request,
        "stats":     stats,
        "n_results": n_results,
        "n_preds":   n_preds,
        "params":    params,
    })


# ── Scores ────────────────────────────────────────────────────────────────────

@router.get("/scores", response_class=HTMLResponse)
def scores_page(request: Request, success: str = "", error: str = "",
                _: str = Depends(require_admin)):
    fixtures = pd.read_csv(FIXTURES_PATH)
    with database.get_connection() as conn:
        entered = {
            r[0] - 200_000
            for r in conn.execute("SELECT match_id FROM match_results").fetchall()
        }
    # Missing: played matches without result
    today = pd.Timestamp.today().normalize().isoformat()[:10]
    played = fixtures[
        (fixtures["home_team"] != "To be announced") &
        (fixtures["away_team"] != "To be announced") &
        (fixtures["date"] < today)
    ]
    missing = played[~played["match_number"].isin(entered)].sort_values("date")
    return templates.TemplateResponse("admin/scores.html", {
        "request": request,
        "missing": missing.to_dict("records"),
        "success": success,
        "error":   error,
    })


@router.post("/scores", response_class=RedirectResponse)
def post_score(
    match_number: int = Form(...),
    home_score:   int = Form(...),
    away_score:   int = Form(...),
    _: str = Depends(require_admin),
):
    try:
        from commands.admin import _apply_score
        _apply_score(match_number, home_score, away_score)
        return RedirectResponse("/admin/scores?success=Score+enregistré", status_code=303)
    except Exception as e:
        return RedirectResponse(f"/admin/scores?error={quote(str(e))}", status_code=303)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@router.get("/fixtures", response_class=HTMLResponse)
def fixtures_page(request: Request, success: str = "", _: str = Depends(require_admin)):
    df  = pd.read_csv(FIXTURES_PATH)
    tba = df[
        (df["home_team"] == "To be announced") | (df["away_team"] == "To be announced")
    ].sort_values("date")
    known_teams = sorted(set(
        df[df["home_team"] != "To be announced"]["home_team"].tolist() +
        df[df["away_team"] != "To be announced"]["away_team"].tolist()
    ))
    return templates.TemplateResponse("admin/fixtures.html", {
        "request":     request,
        "tba":         tba.to_dict("records"),
        "known_teams": known_teams,
        "success":     success,
    })


@router.post("/fixtures", response_class=RedirectResponse)
def post_fixture(
    match_number: int  = Form(...),
    home_team:    str  = Form(...),
    away_team:    str  = Form(...),
    _: str = Depends(require_admin),
):
    df = pd.read_csv(FIXTURES_PATH)
    df.loc[df["match_number"] == match_number, "home_team"] = home_team
    df.loc[df["match_number"] == match_number, "away_team"] = away_team
    df.to_csv(FIXTURES_PATH, index=False)
    return RedirectResponse("/admin/fixtures?success=Fixture+mise+à+jour", status_code=303)


# ── Métriques ML ─────────────────────────────────────────────────────────────

@router.get("/metrics", response_class=HTMLResponse)
def metrics_page(request: Request, _: str = Depends(require_admin)):
    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    from ml.poisson import fit_or_load
    params = fit_or_load()
    attacks  = sorted(
        {k: v for k, v in params.items() if k.startswith("attack_")}.items(),
        key=lambda x: -x[1]
    )[:12]
    defenses = sorted(
        {k: v for k, v in params.items() if k.startswith("defense_")}.items(),
        key=lambda x: x[1]
    )[:12]
    return templates.TemplateResponse("admin/metrics.html", {
        "request":  request,
        "metrics":  metrics,
        "params":   params,
        "attacks":  [(k.replace("attack_", ""), round(v, 3)) for k, v in attacks],
        "defenses": [(k.replace("defense_", ""), round(v, 3)) for k, v in defenses],
    })
