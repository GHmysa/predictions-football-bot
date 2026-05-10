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
    if datetime.now(timezone.utc) - created_at > timedelta(hours=24):
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


init_db()
