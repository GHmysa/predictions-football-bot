import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_PERSISTENT_DIR = Path(os.environ.get("PERSISTENT_DIR", "."))
DB_PATH = str(_PERSISTENT_DIR / "football.db")


def get_connection():
    """Retourne une connexion SQLite au fichier de base de données."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """Crée les tables si elles n'existent pas encore."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS match_results (
                match_id    INTEGER PRIMARY KEY,
                home_team   TEXT    NOT NULL,
                away_team   TEXT    NOT NULL,
                home_score  INTEGER NOT NULL,
                away_score  INTEGER NOT NULL,
                match_group TEXT,
                match_date  TEXT    NOT NULL
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


def get_pending_predictions() -> list[dict]:
    """Retourne toutes les prédictions non encore résolues."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT match_id, home_team, away_team,
                   predicted_home_goals, predicted_away_goals
            FROM predictions
            WHERE actual_result IS NULL
        """).fetchall()
    return [
        {
            "match_id":  r[0],
            "home_team": r[1],
            "away_team": r[2],
            "pred_home": r[3],
            "pred_away": r[4],
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
    """Enregistre une prédiction. Ignorée si le match_id existe déjà."""
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
    """Met à jour une prédiction avec le score réel et calcule is_correct_result."""
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
    """Retourne les statistiques globales et par compétition des prédictions résolues."""
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
        "total":           total or 0,
        "correct_results": correct_results or 0,
        "correct_scores":  correct_scores or 0,
        "result_rate":     round((correct_results or 0) / total * 100, 1) if total else 0.0,
        "score_rate":      round((correct_scores or 0) / total * 100, 1) if total else 0.0,
        "by_competition":  by_competition,
        "best_competition": max(by_competition, key=lambda k: by_competition[k]["rate"], default=None),
    }


def save_match_result(
    match_id: int,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    match_group: str | None,
    match_date: str,
) -> None:
    """Enregistre le score réel d'un match. Si le match_id existe déjà, met à jour le score."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO match_results
                (match_id, home_team, away_team, home_score, away_score, match_group, match_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                home_score = excluded.home_score,
                away_score = excluded.away_score
            """,
            (match_id, home_team, away_team, home_score, away_score, match_group, match_date),
        )


def get_group_results(group: str) -> list[dict]:
    """Retourne tous les matchs joués d'un groupe, triés par date."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT match_id, home_team, away_team, home_score, away_score, match_date
            FROM match_results
            WHERE match_group = ?
            ORDER BY match_date
            """,
            (group,),
        ).fetchall()
    return [
        {
            "match_id":   r[0],
            "home_team":  r[1],
            "away_team":  r[2],
            "home_score": r[3],
            "away_score": r[4],
            "match_date": r[5],
        }
        for r in rows
    ]


init_db()
