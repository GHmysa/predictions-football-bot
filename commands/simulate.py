"""commands/simulate.py — Commande /simulate : Monte Carlo qualification probabilities."""
from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

import discord
import pandas as pd
from discord import app_commands

from ml.simulator import simulate_group

FIXTURES_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"


@lru_cache(maxsize=1)
def _fixtures() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES_PATH)
    return df[df["stage"] == "Group Stage"].copy()


def _bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def _format_result(group: str, results: dict[str, float]) -> str:
    lines = [
        f"## 🎲 Monte Carlo — Groupe {group}",
        "_Probabilité de qualification (top 2) sur 10 000 simulations_",
        "",
        "```",
    ]
    for i, (team, prob) in enumerate(results.items()):
        qualifier = " ✓" if i < 2 else "  "
        lines.append(f"{team:<26} {_bar(prob)} {prob:.1%}{qualifier}")
    lines += [
        "```",
        "*✓ = favoris pour la qualification · Tirs au but simulés 50-50*",
    ]
    return "\n".join(lines)


@app_commands.command(
    name="simulate",
    description="Simulation Monte Carlo des probabilités de qualification (Groupe A–L)",
)
@app_commands.describe(groupe="Groupe à simuler (A à L)")
@app_commands.choices(groupe=[
    app_commands.Choice(name=f"Groupe {g}", value=g)
    for g in "ABCDEFGHIJKL"
])
async def simulate(interaction: discord.Interaction, groupe: app_commands.Choice[str]) -> None:
    await interaction.response.defer()

    group = groupe.value
    group_matches = (
        _fixtures()[_fixtures()["group"] == group]
        .sort_values("date")
        .to_dict("records")
    )

    if not group_matches:
        await interaction.followup.send(f"Aucun match trouvé pour le Groupe {group}.")
        return

    results = await asyncio.to_thread(simulate_group, group_matches, 10_000)
    await interaction.followup.send(_format_result(group, results))


def setup(tree: app_commands.CommandTree) -> None:
    tree.add_command(simulate)
