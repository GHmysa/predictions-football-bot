import os
import requests


def _build_prompt(
    team1_name: str,
    team1_fixtures: list[dict],
    team2_name: str,
    team2_fixtures: list[dict],
) -> str:
    def format_fixtures(name: str, fixtures: list[dict]) -> str:
        if not fixtures:
            return f"{name} : aucun match récent disponible."
        lines = [f"{name} — 5 derniers matchs :"]
        for f in fixtures:
            lines.append(f"  {f['home']} {f['home_goals']} - {f['away_goals']} {f['away']}")
        return "\n".join(lines)

    context = (
        f"{format_fixtures(team1_name, team1_fixtures)}\n\n"
        f"{format_fixtures(team2_name, team2_fixtures)}"
    )

    return (
        f"Voici les résultats récents de deux équipes de football :\n\n"
        f"{context}\n\n"
        f"En te basant sur ces données, génère un pronostic pour le match "
        f"{team1_name} (domicile) vs {team2_name} (extérieur).\n"
        f"Fournis :\n"
        f"1. Le score prédit\n"
        f"2. 3 facteurs clés qui justifient ce pronostic\n"
        f"3. Un niveau de confiance en %\n"
        f"Réponds en français, de façon concise et formatée pour Discord."
    )


def _call_claude(prompt: str) -> str:
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


def _call_mistral(prompt: str) -> str:
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
            "content-type": "application/json",
        },
        json={
            "model": "mistral-large-latest",
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_prono(
    team1_name: str,
    team1_fixtures: list[dict],
    team2_name: str,
    team2_fixtures: list[dict],
) -> str:
    prompt = _build_prompt(team1_name, team1_fixtures, team2_name, team2_fixtures)

    provider = os.getenv("AI_PROVIDER", "mistral").lower()
    if provider == "claude":
        text = _call_claude(prompt)
    else:
        text = _call_mistral(prompt)

    return text.replace(". ", ".\n")
