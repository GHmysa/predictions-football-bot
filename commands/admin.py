"""
commands/admin.py — Commande /score (admin) pour saisir manuellement un score.

Utilité : fallback quand openfootball n'a pas encore mis à jour un résultat.
Réservé aux membres avec la permission Administrateur sur le serveur Discord.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

import os

import discord
import pandas as pd
from discord import app_commands

import database
from services.elo_updater import update_elo_with_match

FIXTURES_PATH   = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"
MATCH_ID_OFFSET = 200_000
_PERSISTENT_DIR = Path(os.environ.get("PERSISTENT_DIR", Path(__file__).parent.parent / "ml" / "data"))


@lru_cache(maxsize=1)
def _fixtures() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_PATH)


def _apply_score(match_number: int, home_score: int, away_score: int) -> str:
    df  = _fixtures()
    row = df[df["match_number"] == match_number]
    if row.empty:
        return f"❌ Match #{match_number} introuvable (numéros valides : 1–{len(df)})."

    r          = row.iloc[0]
    match_id   = MATCH_ID_OFFSET + match_number
    home_team  = r["home_team"]
    away_team  = r["away_team"]
    match_date = r["date"]
    group      = r["group"]

    database.save_match_result(match_id, home_team, away_team, home_score, away_score, group, match_date)
    database.resolve_prediction(match_id, home_score, away_score)

    wc_elo_path = _PERSISTENT_DIR / "wc_elo_updates.csv"
    already_updated = False
    if wc_elo_path.exists() and wc_elo_path.stat().st_size > 0:
        existing = pd.read_csv(wc_elo_path)
        already_updated = (
            (existing["team"].isin([home_team, away_team])) &
            (existing["date"] == match_date)
        ).any()

    if not already_updated:
        update_elo_with_match(home_team, away_team, home_score, away_score, match_date)

    result_str = "Victoire domicile" if home_score > away_score else ("Nul" if home_score == away_score else "Victoire extérieure")
    return (
        f"✅ **Match #{match_number} enregistré**\n"
        f"{home_team} **{home_score} – {away_score}** {away_team}\n"
        f"_{result_str} • ELO mis à jour • /standings et /accuracy à jour_"
    )


@app_commands.command(
    name="score",
    description="[Admin] Enregistre manuellement le score d'un match CdM",
)
@app_commands.describe(
    match_number="Numéro du match (ex: 1 = Mexico vs South Africa)",
    home_score="Buts de l'équipe domicile",
    away_score="Buts de l'équipe extérieure",
)
@app_commands.checks.has_permissions(administrator=True)
async def score_cmd(
    interaction: discord.Interaction,
    match_number: int,
    home_score: int,
    away_score: int,
) -> None:
    await interaction.response.defer(ephemeral=True)
    msg = await asyncio.to_thread(_apply_score, match_number, home_score, away_score)
    await interaction.followup.send(msg, ephemeral=True)


@score_cmd.error
async def score_cmd_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("❌ Commande réservée aux administrateurs.", ephemeral=True)


def setup(tree: app_commands.CommandTree) -> None:
    tree.add_command(score_cmd)
