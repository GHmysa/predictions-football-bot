"""
commands/prono.py — Commande /prono CdM 2026 avec prédictions ML.

UX : /prono groupe:A → sélecteur de match → prédiction ML avec barres de probabilité.
Remplace l'ancien comportement IA + football-data.org.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import discord
import pandas as pd
from discord import app_commands

import database
from ml.predict import predict_match
from services.ml_model import format_result

FIXTURES_PATH  = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"
WC_COMPETITION = "WC2026"

# Préfixe pour les match_id en DB : évite toute collision avec les IDs football-data.org
# Les match_numbers CdM vont de 1 à 104 → IDs DB : 200001–200104
MATCH_ID_OFFSET = 200_000


@lru_cache(maxsize=1)
def _fixtures() -> pd.DataFrame:
    """Charge les fixtures groupe stage une seule fois."""
    df = pd.read_csv(FIXTURES_PATH)
    return df[df["stage"] == "Group Stage"].copy()


def _group_matches(group: str) -> list[dict]:
    """Retourne les matchs d'un groupe triés par date."""
    return (
        _fixtures()[_fixtures()["group"] == group]
        .sort_values("date")
        .to_dict("records")
    )


class MatchSelect(discord.ui.Select):
    def __init__(self, matches: list[dict], group: str):
        self._matches = {str(m["match_number"]): m for m in matches}
        options = []
        for m in matches:
            date_fr = datetime.strptime(m["date"], "%Y-%m-%d").strftime("%d/%m")
            venue   = m["venue"].replace(" Stadium", "")
            options.append(discord.SelectOption(
                label=f"{m['home_team']} vs {m['away_team']}"[:100],
                description=f"{date_fr}  •  {venue}"[:100],
                value=str(m["match_number"]),
            ))
        super().__init__(
            placeholder=f"Groupe {group} — choisissez un match…",
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        match   = self._matches[self.values[0]]
        home    = match["home_team"]
        away    = match["away_team"]
        date    = match["date"]
        match_id = MATCH_ID_OFFSET + int(match["match_number"])

        await interaction.response.edit_message(
            content=f"⏳ Calcul de la prédiction ML pour **{home} vs {away}**…",
            view=None,
        )

        try:
            # predict_match() est synchrone (pandas + pickle) → thread pour ne pas
            # bloquer la boucle événementielle Discord pendant le calcul
            result = await asyncio.to_thread(
                predict_match, home, away, date, True, 4
            )
        except Exception as e:
            await interaction.followup.send(f"❌ Erreur ML : {e}")
            return

        message = format_result(result)

        # Encode H/D/A comme score symbolique pour la DB existante
        # (is_correct_result sera correct ; is_correct_score n'est pas pertinent ici)
        pred      = result["prediction"]
        pred_home = 1 if pred == "home" else 0
        pred_away = 1 if pred == "away" else 0
        await asyncio.to_thread(
            database.save_prediction,
            match_id, WC_COMPETITION, home, away, pred_home, pred_away,
        )

        await interaction.followup.send(message)


class GroupView(discord.ui.View):
    def __init__(self, matches: list[dict], group: str):
        super().__init__(timeout=120)
        self.add_item(MatchSelect(matches, group))


@app_commands.command(
    name="prono",
    description="Prédictions ML pour les matchs de la Coupe du Monde 2026",
)
@app_commands.describe(groupe="Groupe à consulter (A à L)")
@app_commands.choices(groupe=[
    app_commands.Choice(name=f"Groupe {g}", value=g)
    for g in "ABCDEFGHIJKL"
])
async def prono(interaction: discord.Interaction, groupe: app_commands.Choice[str]) -> None:
    """Affiche un sélecteur de match pour le groupe demandé."""
    await interaction.response.defer()

    matches = _group_matches(groupe.value)
    if not matches:
        await interaction.followup.send(
            f"Aucun match trouvé pour le Groupe {groupe.value}."
        )
        return

    await interaction.followup.send(
        f"**🏆 Coupe du Monde 2026 — Groupe {groupe.value}**\nChoisissez un match :",
        view=GroupView(matches, groupe.value),
    )


def setup(tree: app_commands.CommandTree) -> None:
    """Enregistre la commande dans le command tree Discord."""
    tree.add_command(prono)
