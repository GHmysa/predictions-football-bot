"""
services/ml_model.py — Formatage des prédictions ML pour Discord.
"""
from __future__ import annotations


def _bar(prob: float, width: int = 10) -> str:
    """Barre ASCII proportionnelle à la probabilité."""
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def format_result(r: dict) -> str:
    """
    Formate le dict retourné par predict_match() en message Discord.
    Fonction pure — aucun IO. Appelée par commands/prono.py.
    """
    p = r["probabilities"]

    def _bold(label: str, key: str) -> str:
        return f"**{label}**" if r["prediction"] == key else label

    lines = [
        f"## {r['home_team']}  vs  {r['away_team']}",
        f"📅 {r['date']}  •  ELO : {r['elo_home']:.0f} vs {r['elo_away']:.0f}",
        "",
        f"{_bold('🏠 Victoire ' + r['home_team'], 'home')}",
        f"`{_bar(p['home'])}` {p['home']:.0%}",
        "",
        f"{_bold('🤝 Match nul', 'draw')}",
        f"`{_bar(p['draw'])}` {p['draw']:.0%}",
        "",
        f"{_bold('✈️ Victoire ' + r['away_team'], 'away')}",
        f"`{_bar(p['away'])}` {p['away']:.0%}",
        "",
        f"**Prédiction : {r['prediction_fr']}** (confiance {r['confidence']:.0%})",
    ]
    return "\n".join(lines)
