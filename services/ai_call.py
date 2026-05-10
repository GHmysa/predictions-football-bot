import os
import requests


def _compute_stats(fixtures: list[dict]) -> dict:
    n = len(fixtures)
    scored   = sum(f["team_goals"] for f in fixtures)
    conceded = sum(f["opponent_goals"] for f in fixtures)
    return {
        "wins":          sum(1 for f in fixtures if f["result"] == "V"),
        "draws":         sum(1 for f in fixtures if f["result"] == "N"),
        "losses":        sum(1 for f in fixtures if f["result"] == "D"),
        "goals_scored":  scored,
        "goals_conceded": conceded,
        "avg_scored":    round(scored / n, 1) if n else 0.0,
        "avg_conceded":  round(conceded / n, 1) if n else 0.0,
    }


def _format_data_block(name: str, fixtures: list[dict]) -> str:
    if not fixtures:
        return f"{name} — aucune donnée disponible."
    s = _compute_stats(fixtures)
    bilan = f"{s['wins']}V {s['draws']}N {s['losses']}D"
    scores = ", ".join(
        f"{f['result']} {f['team_goals']}-{f['opponent_goals']} vs {f['opponent']}"
        for f in fixtures
    )
    return f"{name} — {bilan} — {scores}"


def _build_prompt(
    team1_name: str,
    team1_fixtures: list[dict],
    team2_name: str,
    team2_fixtures: list[dict],
    competition: str = "—",
    venue: str = "Stade du domicile",
) -> str:
    s1 = _compute_stats(team1_fixtures) if team1_fixtures else {}
    s2 = _compute_stats(team2_fixtures) if team2_fixtures else {}

    data1 = _format_data_block(team1_name, team1_fixtures)
    data2 = _format_data_block(team2_name, team2_fixtures)

    # Bloc stats pré-rempli côté Python — le modèle doit le reproduire tel quel
    prefilled_stats = (
        f"📊 **Stats clés** :\n"
        f"• Attaque {team1_name} : {s1.get('avg_scored', '?')} buts/match | Défense : {s1.get('avg_conceded', '?')} encaissés/match\n"
        f"• Attaque {team2_name} : {s2.get('avg_scored', '?')} buts/match | Défense : {s2.get('avg_conceded', '?')} encaissés/match\n"
        f"• Forme {team1_name} : {s1.get('wins', '?')}V {s1.get('draws', '?')}N {s1.get('losses', '?')}D sur 5 matchs\n"
        f"• Forme {team2_name} : {s2.get('wins', '?')}V {s2.get('draws', '?')}N {s2.get('losses', '?')}D sur 5 matchs"
    )

    return (
        f"Tu es un analyste football pour parieurs professionnels.\n"
        f"INTERDIT ABSOLU : ne cite aucun nom de joueur, entraîneur, ou personne réelle.\n"
        f"Si tu cites un nom propre de personne, ta réponse est invalide.\n\n"
        f"DONNÉES SOURCE (seules données autorisées) :\n"
        f"{data1}\n"
        f"{data2}\n"
        f"Compétition : {competition} | Lieu : {venue}\n\n"
        f"Les stats ont été calculées côté serveur et sont déjà écrites ci-dessous.\n"
        f"Tu dois reproduire le bloc suivant EXACTEMENT, en remplaçant UNIQUEMENT les trois champs [COMPLÉTER] :\n\n"
        f"⚽ **Score prédit** : [COMPLÉTER]\n"
        f"{prefilled_stats}\n"
        f"🎯 **Verdict** : [COMPLÉTER — 2 phrases max. Uniquement des faits tirés des stats ci-dessus. Zéro nom de joueur.]\n"
        f"📈 **Confiance** : [COMPLÉTER — XX%]"
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
            "max_tokens": 600,
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
            "max_tokens": 600,
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
    competition: str = "—",
    venue: str = "Stade du domicile",
) -> str:
    prompt = _build_prompt(
        team1_name, team1_fixtures,
        team2_name, team2_fixtures,
        competition, venue,
    )
    provider = os.getenv("AI_PROVIDER", "mistral").lower()
    if provider == "claude":
        return _call_claude(prompt)
    return _call_mistral(prompt)
