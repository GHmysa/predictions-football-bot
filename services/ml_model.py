"""
services/ml_model.py — Wrapper entre le modèle ML et le bot Discord.

Charge le modèle une seule fois (lru_cache), puis expose format_prediction()
qui retourne un message Discord prêt à envoyer pour un match donné.
"""
from __future__ import annotations

import asyncio
from ml.predict import predict_match


def _bar(prob: float, width: int = 10) -> str:
    """Barre de progression ASCII proportionnelle à la probabilité."""
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def format_result(r: dict) -> str:
    """
    Formate le dict retourné par predict_match() en message Discord.

    Fonction pure — ne fait aucun appel réseau ni IO.
    Séparée de format_prediction() pour permettre à l'appelant de réutiliser
    le dict (ex. pour sauvegarder en DB) sans appeler predict_match() deux fois.
    """
    p      = r["probabilities"]
    p_home = p["home"]
    p_draw = p["draw"]
    p_away = p["away"]

    def _bold(label: str, key: str) -> str:
        return f"**{label}**" if r["prediction"] == key else label

    lines = [
        f"## {r['home_team']}  vs  {r['away_team']}",
        f"📅 {r['date']}  •  ELO : {r['elo_home']:.0f} vs {r['elo_away']:.0f}",
        "",
        f"{_bold('🏠 Victoire ' + r['home_team'], 'home')}",
        f"`{_bar(p_home)}` {p_home:.0%}",
        "",
        f"{_bold('🤝 Match nul', 'draw')}",
        f"`{_bar(p_draw)}` {p_draw:.0%}",
        "",
        f"{_bold('✈️ Victoire ' + r['away_team'], 'away')}",
        f"`{_bar(p_away)}` {p_away:.0%}",
        "",
        f"**Prédiction : {r['prediction_fr']}** (confiance {r['confidence']:.0%})",
    ]
    return "\n".join(lines)


def format_prediction(home_team: str, away_team: str, date: str) -> str:
    """Appelle predict_match() puis format_result(). Pratique pour un usage standalone."""
    r = predict_match(home_team=home_team, away_team=away_team, date=date,
                      is_neutral=True, tournament_tier=4)
    return format_result(r)


async def format_prediction_async(home_team: str, away_team: str, date: str) -> str:
    """
    Version async de format_prediction pour le bot Discord.

    predict_match() est synchrone (pandas + pickle) — on le délègue à un thread
    pour ne pas bloquer la boucle événementielle d'asyncio.
    """
    return await asyncio.to_thread(format_prediction, home_team, away_team, date)
