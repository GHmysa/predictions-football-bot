import os
import httpx

BASE_URL = "https://api.football-data.org/v4"


class RateLimitError(Exception):
    pass


def _headers() -> dict:
    return {"X-Auth-Token": os.getenv("FOOTBALL_DATA_KEY")}


def _check_rate_limit(response: httpx.Response) -> None:
    available = response.headers.get("X-Requests-Available-Minute")
    if available is not None and int(available) == 0:
        raise RateLimitError("Limite de requêtes atteinte (10/min). Réessayez dans une minute.")


async def fetch_competition_teams(competition_code: str) -> list[dict]:
    """Return all teams in a competition (used to build the local search cache)."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/competitions/{competition_code}/teams",
            headers=_headers(),
        )
        response.raise_for_status()
        _check_rate_limit(response)
        data = response.json()

    return [
        {
            "id": t["id"],
            "name": t["name"],
            "country": t.get("area", {}).get("name", "—"),
        }
        for t in data.get("teams", [])
    ]


async def search_teams(name: str) -> list[dict]:
    """Kept for fallback text search — not reliable on the free plan."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/teams",
            headers=_headers(),
            params={"name": name},
        )
        response.raise_for_status()
        _check_rate_limit(response)
        data = response.json()

    return [
        {
            "id": t["id"],
            "name": t["name"],
            "country": t.get("area", {}).get("name", "—"),
        }
        for t in data.get("teams", [])
    ]


async def fetch_fixtures(team_id: int) -> list[dict]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/teams/{team_id}/matches",
            headers=_headers(),
            params={"status": "FINISHED", "limit": 5},
        )
        response.raise_for_status()
        _check_rate_limit(response)
        data = response.json()

    matches = data.get("matches", [])
    matches.sort(key=lambda m: m["utcDate"], reverse=True)

    result = []
    for match in matches[:5]:
        home_id = match["homeTeam"]["id"]
        home_name = match["homeTeam"]["name"]
        away_name = match["awayTeam"]["name"]
        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]
        winner = match["score"]["winner"]  # "HOME_TEAM" | "AWAY_TEAM" | "DRAW"

        team_is_home = home_id == team_id
        team_goals = home_goals if team_is_home else away_goals
        opponent_goals = away_goals if team_is_home else home_goals
        opponent = away_name if team_is_home else home_name

        if winner == "DRAW":
            outcome = "N"
        elif (winner == "HOME_TEAM" and team_is_home) or (winner == "AWAY_TEAM" and not team_is_home):
            outcome = "V"
        else:
            outcome = "D"

        # home_winner recalculé pour compatibilité avec stats.py (_format_fixture)
        if winner == "HOME_TEAM":
            home_winner = True
        elif winner == "AWAY_TEAM":
            home_winner = False
        else:
            home_winner = None

        result.append({
            "date": match["utcDate"],
            "home": home_name,
            "away": away_name,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_winner": home_winner,
            "team_is_home": team_is_home,
            "result": outcome,
            "opponent": opponent,
            "team_goals": team_goals,
            "opponent_goals": opponent_goals,
        })

    return result


async def fetch_upcoming_fixtures(competition_code: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/competitions/{competition_code}/matches",
            headers=_headers(),
            params={"status": "SCHEDULED"},
        )
        response.raise_for_status()
        _check_rate_limit(response)
        data = response.json()

    matches = data.get("matches", [])
    matches.sort(key=lambda m: m["utcDate"])

    return [
        {
            "fixture_id": m["id"],
            "date": m["utcDate"],
            "home_team": m["homeTeam"]["name"],
            "home_team_id": m["homeTeam"]["id"],
            "away_team": m["awayTeam"]["name"],
            "away_team_id": m["awayTeam"]["id"],
        }
        for m in matches[:10]
    ]
