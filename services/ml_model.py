"""
services/ml_model.py — Formatage des prédictions ML pour Discord.
"""
from __future__ import annotations


def _bar(prob: float, width: int = 10) -> str:
    """Barre Unicode proportionnelle à la probabilité."""
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def _confidence_label(max_prob: float) -> str:
    """
    Indicateur qualitatif de la certitude du modèle.
    Basé sur la probabilité de l'issue la plus probable.
    """
    if max_prob >= 0.60:
        return "Favori clair"
    elif max_prob >= 0.50:
        return "Légère faveur"
    else:
        return "Match serré"


def format_result(r: dict) -> str:
    """
    Formate le dict retourné par predict_match() en message Discord.
    Fonction pure — aucun IO. Appelée par commands/prono.py.
    """
    p        = r["probabilities"]
    max_prob = max(p["home"], p["draw"], p["away"])
    conf     = _confidence_label(max_prob)

    def _bold(label: str, key: str) -> str:
        return f"**{label}**" if r["prediction"] == key else label

    score_line = ""
    sh = r.get("predicted_score_home")
    sa = r.get("predicted_score_away")
    if sh is not None and sa is not None:
        score_line = f"\n🎯 Score prédit : **{sh} – {sa}**"

    lines = [
        f"## {r['home_team']}  vs  {r['away_team']}",
        f"📅 {r['date']}  •  ELO : {r['elo_home']:.0f} vs {r['elo_away']:.0f}  •  _{conf}_{score_line}",
        "",
        f"{_bold('🏠 Victoire ' + r['home_team'], 'home')}",
        f"`{_bar(p['home'])}` {p['home']:.0%}",
        "",
        f"{_bold('🤝 Match nul', 'draw')}",
        f"`{_bar(p['draw'])}` {p['draw']:.0%}",
        "",
        f"{_bold('✈️ Victoire ' + r['away_team'], 'away')}",
        f"`{_bar(p['away'])}` {p['away']:.0%}",
    ]
    return "\n".join(lines)
