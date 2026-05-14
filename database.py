import sqlite3
from datetime import datetime, timedelta, timezone

DB_PATH = "football.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prono_cache (
                fixture_id  INTEGER PRIMARY KEY,
                team1       TEXT NOT NULL,
                team2       TEXT NOT NULL,
                result_text TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS teams_cache (
                competition_code TEXT PRIMARY KEY,
                teams_json       TEXT NOT NULL,
                cached_at        TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id             INTEGER NOT NULL UNIQUE,
                competition          TEXT NOT NULL,
                home_team            TEXT NOT NULL,
                away_team            TEXT NOT NULL,
                predicted_home_goals INTEGER NOT NULL,
                predicted_away_goals INTEGER NOT NULL,
                predicted_result     TEXT NOT NULL,
                actual_home_goals    INTEGER,
                actual_away_goals    INTEGER,
                actual_result        TEXT,
                is_correct_result    INTEGER,
                is_correct_score     INTEGER,
                created_at           TEXT NOT NULL,
                resolved_at          TEXT
            )
        """)


def get_cached_prono(fixture_id: int) -> str | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT result_text, created_at FROM prono_cache WHERE fixture_id = ?",
            (fixture_id,),
        ).fetchone()
    if row is None:
        return None
    result_text, created_at_str = row
    created_at = datetime.fromisoformat(created_at_str)
    if datetime.now(timezone.utc) - created_at > timedelta(days=30):
        return None
    return result_text


def save_prono(fixture_id: int, team1: str, team2: str, result_text: str):
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO prono_cache (fixture_id, team1, team2, result_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(fixture_id) DO UPDATE SET
                result_text = excluded.result_text,
                created_at  = excluded.created_at
            """,
            (fixture_id, team1, team2, result_text, now),
        )


def get_cached_teams(competition_code: str) -> list | None:
    import json
    with get_connection() as conn:
        row = conn.execute(
            "SELECT teams_json, cached_at FROM teams_cache WHERE competition_code = ?",
            (competition_code,),
        ).fetchone()
    if row is None:
        return None
    teams_json, cached_at_str = row
    cached_at = datetime.fromisoformat(cached_at_str)
    if datetime.now(timezone.utc) - cached_at > timedelta(hours=24):
        return None
    return json.loads(teams_json)


def save_teams(competition_code: str, teams: list) -> None:
    import json
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO teams_cache (competition_code, teams_json, cached_at)
            VALUES (?, ?, ?)
            ON CONFLICT(competition_code) DO UPDATE SET
                teams_json = excluded.teams_json,
                cached_at  = excluded.cached_at
            """,
            (competition_code, json.dumps(teams), now),
        )


def get_pending_predictions() -> list[dict]:
    """Return all predictions not yet resolved."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT match_id, home_team, away_team,
                   predicted_home_goals, predicted_away_goals
            FROM predictions
            WHERE actual_result IS NULL
        """).fetchall()
    return [
        {
            "match_id":   r[0],
            "home_team":  r[1],
            "away_team":  r[2],
            "pred_home":  r[3],
            "pred_away":  r[4],
        }
        for r in rows
    ]


def save_prediction(
    match_id: int,
    competition: str,
    home: str,
    away: str,
    pred_home: int,
    pred_away: int,
) -> None:
    if pred_home > pred_away:
        predicted_result = "H"
    elif pred_home < pred_away:
        predicted_result = "A"
    else:
        predicted_result = "D"

    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (match_id, competition, home_team, away_team,
                 predicted_home_goals, predicted_away_goals, predicted_result, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO NOTHING
            """,
            (match_id, competition, home, away, pred_home, pred_away, predicted_result, now),
        )


def resolve_prediction(match_id: int, actual_home: int, actual_away: int) -> None:
    if actual_home > actual_away:
        actual_result = "H"
    elif actual_home < actual_away:
        actual_result = "A"
    else:
        actual_result = "D"

    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT predicted_result, predicted_home_goals, predicted_away_goals FROM predictions WHERE match_id = ?",
            (match_id,),
        ).fetchone()
        if row is None:
            return
        pred_result, pred_home, pred_away = row
        is_correct_result = int(pred_result == actual_result)
        is_correct_score  = int(pred_home == actual_home and pred_away == actual_away)

        conn.execute(
            """
            UPDATE predictions SET
                actual_home_goals = ?,
                actual_away_goals = ?,
                actual_result     = ?,
                is_correct_result = ?,
                is_correct_score  = ?,
                resolved_at       = ?
            WHERE match_id = ?
            """,
            (actual_home, actual_away, actual_result,
             is_correct_result, is_correct_score, now, match_id),
        )


def get_stats() -> dict:
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)               AS total,
                SUM(is_correct_result) AS correct_results,
                SUM(is_correct_score)  AS correct_scores
            FROM predictions
            WHERE actual_result IS NOT NULL
        """).fetchone()
        total, correct_results, correct_scores = row or (0, 0, 0)

        rows = conn.execute("""
            SELECT
                competition,
                COUNT(*)               AS total,
                SUM(is_correct_result) AS correct_results
            FROM predictions
            WHERE actual_result IS NOT NULL
            GROUP BY competition
            ORDER BY correct_results DESC
        """).fetchall()

    by_competition = {
        r[0]: {
            "total":           r[1],
            "correct_results": r[2] or 0,
            "rate":            round((r[2] or 0) / r[1] * 100, 1) if r[1] else 0.0,
        }
        for r in rows
    }

    return {
        "total":            total or 0,
        "correct_results":  correct_results or 0,
        "correct_scores":   correct_scores or 0,
        "result_rate":      round((correct_results or 0) / total * 100, 1) if total else 0.0,
        "score_rate":       round((correct_scores or 0) / total * 100, 1) if total else 0.0,
        "by_competition":   by_competition,
        "best_competition": max(by_competition, key=lambda k: by_competition[k]["rate"], default=None),
    }


init_db()
