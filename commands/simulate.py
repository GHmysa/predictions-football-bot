"""commands/simulate.py — /simulate : Monte Carlo WC 2026 winner probabilities from R16."""
from __future__ import annotations

import asyncio
from pathlib import Path

import discord
import pandas as pd
from discord import app_commands

from ml.simulator import simulate_ko_from_r16

FIXTURES_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"


def _bar(prob: float, width: int = 18) -> str:
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def _format_result(results: dict[str, float]) -> str:
    lines = [
        "## 🎲 Monte Carlo — Vainqueur CdM 2026",
        "_Probabilité de remporter le titre · 10 000 simulations depuis les 1/8_",
        "",
        "```",
    ]
    for team, prob in results.items():
        lines.append(f"{team:<26} {_bar(prob)} {prob:.1%}")
    lines += [
        "```",
        "*Tirs au but simulés 50-50 · ELO et Poisson mis à jour après chaque match*",
    ]
    return "\n".join(lines)


@app_commands.command(
    name="simulate",
    description="Simulation Monte Carlo des chances de remporter la CdM 2026 (depuis les 1/8)",
)
async def simulate(interaction: discord.Interaction) -> None:
    await interaction.response.defer()

    fixtures = pd.read_csv(FIXTURES_PATH)
    result, tba = await asyncio.to_thread(simulate_ko_from_r16, fixtures, 10_000)

    if result is None:
        tba_lines = "\n".join(f"• {t}" for t in tba)
        await interaction.followup.send(
            f"⚠️ Certains matchs de 1/8 ont des équipes non définies :\n{tba_lines}\n\n"
            "Complète les fixtures via le dashboard (Page Fixtures) puis relance."
        )
        return

    await interaction.followup.send(_format_result(result))


def setup(tree: app_commands.CommandTree) -> None:
    tree.add_command(simulate)
