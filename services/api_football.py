import os
import httpx

BASE_URL = "https://v3.football.api-sports.io"


def _headers() -> dict:
    return {"x-apisports-key": os.getenv("API_FOOTBALL_KEY")}


async def search_teams(name: str) -> list[dict]:
    """Return a list of {id, name, country} matching the search."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/teams",
            headers=_headers(),
            params={"search": name},
        )
        response.raise_for_status()
        data = response.json()

    return [
        {
            "id": item["team"]["id"],
            "name": item["team"]["name"],
            "country": item["team"]["country"],
        }
        for item in data.get("response", [])
    ]


async def fetch_fixtures(team_id: int) -> list[dict]:
    """Return up to 5 finished fixtures for the team in season 2024, sorted by date desc."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/fixtures",
            headers=_headers(),
            params={"team": team_id, "season": 2024, "status": "FT"},
        )
        response.raise_for_status()
        data = response.json()

    fixtures = data.get("response", [])
    fixtures.sort(key=lambda f: f["fixture"]["date"], reverse=True)

    return [
        {
            "date": fixture["fixture"]["date"],
            "home": fixture["teams"]["home"]["name"],
            "away": fixture["teams"]["away"]["name"],
            "home_goals": fixture["goals"]["home"],
            "away_goals": fixture["goals"]["away"],
            "home_winner": fixture["teams"]["home"]["winner"],
        }
        for fixture in fixtures[:5]
    ]
