"""
commands/prono.py — Commande /prono CdM 2026 avec prédictions ML.

UX : /prono groupe:A → sélecteur de match → prédiction ML avec barres de probabilité.
     /prono groupe:R32 → sélecteur matchs Round of 32, etc.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
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
MATCH_ID_OFFSET = 200_000

# Traduction valeur Discord → nom de stage dans le CSV
STAGE_MAP = {
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF":  "Quarter Finals",
    "SF":  "Semi Finals",
    "F":   "Finals",
}


@lru_cache(maxsize=1)
def _fixtures() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_PATH)


def _get_matches(selection: str) -> list[dict]:
    """Retourne les matchs à venir pour un groupe (A-L) ou un tour KO (R32, R16…)."""
    today = date.today().isoformat()
    df = _fixtures()
    if selection in STAGE_MAP:
        stage = STAGE_MAP[selection]
        filtered = df[
            (df["stage"] == stage) &
            (df["date"] >= today) &
            (df["home_team"] != "To be announced") &
            (df["away_team"] != "To be announced")
        ]
    else:
        filtered = df[
            (df["group"] == selection) &
            (df["date"] >= today)
        ]
    return filtered.sort_values("date").to_dict("records")


def _display_label(selection: str) -> str:
    if selection in STAGE_MAP:
        return STAGE_MAP[selection]
    return f"Groupe {selection}"


class MatchSelect(discord.ui.Select):
    def __init__(self, matches: list[dict], selection: str):
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
            placeholder=f"{_display_label(selection)} — choisissez un match…",
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        match    = self._matches[self.values[0]]
        home     = match["home_team"]
        away     = match["away_team"]
        dt       = match["date"]
        match_id = MATCH_ID_OFFSET + int(match["match_number"])

        await interaction.response.edit_message(
            content=f"⏳ Calcul de la prédiction ML pour **{home} vs {away}**…",
            view=None,
        )

        try:
            result = await asyncio.to_thread(
                predict_match, home, away, dt, True, 4
            )
        except Exception as e:
            await interaction.followup.send(f"❌ Erreur ML : {e}")
            return

        message = format_result(result)

        pred      = result["prediction"]
        pred_home = 1 if pred == "home" else 0
        pred_away = 1 if pred == "away" else 0
        await asyncio.to_thread(
            database.save_prediction,
            match_id, WC_COMPETITION, home, away, pred_home, pred_away,
        )

        await interaction.followup.send(message)


class GroupView(discord.ui.View):
    def __init__(self, matches: list[dict], selection: str):
        super().__init__(timeout=120)
        self.add_item(MatchSelect(matches, selection))


@app_commands.command(
    name="prono",
    description="Prédictions ML pour les matchs de la Coupe du Monde 2026",
)
@app_commands.describe(groupe="Groupe (A à L) ou tour éliminatoire (Round of 32…)")
@app_commands.choices(groupe=[
    app_commands.Choice(name=f"Groupe {g}", value=g)
    for g in "ABCDEFGHIJKL"
] + [
    app_commands.Choice(name="Round of 32 (1/16)",  value="R32"),
    app_commands.Choice(name="Round of 16 (1/8)",   value="R16"),
    app_commands.Choice(name="Quarts de finale",    value="QF"),
    app_commands.Choice(name="Demi-finales",        value="SF"),
    app_commands.Choice(name="Finale / 3e place",   value="F"),
])
async def prono(interaction: discord.Interaction, groupe: app_commands.Choice[str]) -> None:
    await interaction.response.defer()

    matches = _get_matches(groupe.value)
    if not matches:
        label = _display_label(groupe.value)
        await interaction.followup.send(
            f"Tous les matchs du {label} sont terminés ou les équipes ne sont pas encore connues."
        )
        return

    label = _display_label(groupe.value)
    await interaction.followup.send(
        f"**🏆 Coupe du Monde 2026 — {label}**\nChoisissez un match :",
        view=GroupView(matches, groupe.value),
    )


def setup(tree: app_commands.CommandTree) -> None:
    tree.add_command(prono)
