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

    result = []
    for fixture in fixtures[:5]:
        home_id = fixture["teams"]["home"]["id"]
        home_name = fixture["teams"]["home"]["name"]
        away_name = fixture["teams"]["away"]["name"]
        home_goals = fixture["goals"]["home"]
        away_goals = fixture["goals"]["away"]
        home_winner = fixture["teams"]["home"]["winner"]

        team_is_home = home_id == team_id
        team_goals = home_goals if team_is_home else away_goals
        opponent_goals = away_goals if team_is_home else home_goals
        opponent = away_name if team_is_home else home_name

        if home_winner is None:
            outcome = "N"
        elif (home_winner and team_is_home) or (not home_winner and not team_is_home):
            outcome = "V"
        else:
            outcome = "D"

        result.append({
            "date": fixture["fixture"]["date"],
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
