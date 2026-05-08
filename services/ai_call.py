import os
import requests


def _compute_stats(fixtures: list[dict]) -> dict:
    return {
        "wins": sum(1 for f in fixtures if f["result"] == "V"),
        "draws": sum(1 for f in fixtures if f["result"] == "N"),
        "losses": sum(1 for f in fixtures if f["result"] == "D"),
        "goals_scored": sum(f["team_goals"] for f in fixtures),
        "goals_conceded": sum(f["opponent_goals"] for f in fixtures),
    }


def _format_team_block(name: str, role: str, fixtures: list[dict]) -> str:
    if not fixtures:
        return f"{name} ({role}) — aucun match récent disponible."

    s = _compute_stats(fixtures)
    bilan = f"{s['wins']}V {s['draws']}N {s['losses']}D — {s['goals_scored']} buts marqués, {s['goals_conceded']} encaissés"

    results_parts = []
    for f in fixtures:
        results_parts.append(f"{f['result']} {f['team_goals']}-{f['opponent_goals']} vs {f['opponent']}")
    results_line = ", ".join(results_parts)

    return (
        f"{name} ({role}) — Bilan 5 derniers matchs : {bilan}\n"
        f"Résultats : {results_line}"
    )


def _build_prompt(
    team1_name: str,
    team1_fixtures: list[dict],
    team2_name: str,
    team2_fixtures: list[dict],
) -> str:
    block1 = _format_team_block(team1_name, "Domicile", team1_fixtures)
    block2 = _format_team_block(team2_name, "Extérieur", team2_fixtures)

    return (
        f"Tu es un expert en analyse football. Voici les données des deux équipes.\n\n"
        f"{block1}\n\n"
        f"{block2}\n\n"
        f"Réponds UNIQUEMENT avec ce format, sans texte avant ou après :\n"
        f"🏆 **Score prédit** : X - Y\n"
        f"📊 **Analyse** :\n"
        f"• Facteur 1\n"
        f"• Facteur 2\n"
        f"• Facteur 3\n"
        f"🎯 **Confiance** : XX%\n"
        f"⚡ **À surveiller** : [un élément décisif du match]"
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
        return _call_claude(prompt)
    return _call_mistral(prompt)
