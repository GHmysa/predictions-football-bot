"""
commands/standings.py — Commande /standings pour le classement d'un groupe CdM 2026.

Affiche le classement en temps réel basé sur les matchs résolus en DB.
Si aucun match n'a encore été joué, affiche le calendrier avec les prédictions ML.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import discord
import pandas as pd
from discord import app_commands

import database
from ml.predict import predict_match

FIXTURES_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"


@lru_cache(maxsize=1)
def _fixtures() -> pd.DataFrame:
    """Charge les fixtures groupe stage une seule fois."""
    df = pd.read_csv(FIXTURES_PATH)
    return df[df["stage"] == "Group Stage"].copy()


def _group_teams(group: str) -> list[str]:
    """Retourne les 4 équipes d'un groupe dans l'ordre d'apparition."""
    df = _fixtures()[_fixtures()["group"] == group]
    seen, teams = set(), []
    for team in pd.concat([df["home_team"], df["away_team"]]):
        if team not in seen:
            seen.add(team)
            teams.append(team)
    return teams


def _group_schedule(group: str) -> list[dict]:
    """Retourne les 6 matchs du groupe triés par date."""
    df = _fixtures()[_fixtures()["group"] == group].sort_values("date")
    return df.to_dict("records")


def _compute_standings(teams: list[str], results: list[dict]) -> list[dict]:
    """
    Calcule le classement à partir des résultats joués.
    Tri : points → diff de buts → buts marqués → nom (alphabétique).
    """
    table = {
        t: {"team": t, "pts": 0, "played": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0}
        for t in teams
    }

    for r in results:
        home, away = r["home_team"], r["away_team"]
        hg,   ag   = r["home_score"], r["away_score"]

        for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
            if team not in table:
                continue
            table[team]["played"] += 1
            table[team]["gf"]     += gf
            table[team]["ga"]     += ga

        if hg > ag:
            table[home]["pts"] += 3
            table[home]["w"]   += 1
            table[away]["l"]   += 1
        elif hg == ag:
            table[home]["pts"] += 1
            table[home]["d"]   += 1
            table[away]["pts"] += 1
            table[away]["d"]   += 1
        else:
            table[away]["pts"] += 3
            table[away]["w"]   += 1
            table[home]["l"]   += 1

    for t in table.values():
        t["gd"] = t["gf"] - t["ga"]

    return sorted(
        table.values(),
        key=lambda t: (-t["pts"], -t["gd"], -t["gf"], t["team"]),
    )


def _fmt_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m")


def _build_message(group: str) -> str:
    teams    = _group_teams(group)
    results  = database.get_group_results(group)
    schedule = _group_schedule(group)

    played_ids   = {r["match_id"] for r in results}
    total_matches = len(schedule)
    played_count  = len(results)

    lines = [f"## 🏆 Groupe {group}"]

    # --- Classement (si au moins 1 match joué) ---
    if results:
        standings = _compute_standings(teams, results)
        journee   = max(
            sum(1 for m in schedule
                if (200_000 + int(m["match_number"])) in played_ids
                and (m["home_team"] == s["team"] or m["away_team"] == s["team"]))
            for s in standings
        )
        lines.append(f"*Journée {journee}/3 — {played_count}/{total_matches} matchs joués*")
        lines.append("```")
        lines.append(f" #  {'Équipe':<26} Pts  J  V  N  D   GF  GA  +/-")
        lines.append(f" {'─'*58}")
        for i, s in enumerate(standings, 1):
            gd_str = f"+{s['gd']}" if s["gd"] > 0 else str(s["gd"])
            qualifier = " ✓" if i <= 2 else ""
            lines.append(
                f" {i}  {s['team']:<26} {s['pts']:>3}  {s['played']}  "
                f"{s['w']}  {s['d']}  {s['l']}  "
                f" {s['gf']:>2}  {s['ga']:>2}  {gd_str:>3}{qualifier}"
            )
        lines.append("```")
        lines.append("")

        # Résultats joués
        lines.append("**Résultats :**")
        for r in results:
            lines.append(
                f"✅ {r['home_team']} **{r['home_score']}–{r['away_score']}** "
                f"{r['away_team']}  *({_fmt_date(r['match_date'])})*"
            )
        lines.append("")

    # --- Matchs à venir avec prédictions ML ---
    upcoming = [
        m for m in schedule
        if (200_000 + int(m["match_number"])) not in played_ids
    ]

    if upcoming:
        lines.append("**Prochains matchs :**")
        for m in upcoming:
            r   = predict_match(m["home_team"], m["away_team"], m["date"])
            p   = r["probabilities"]
            dom = f"{p['home']:.0%}"
            nul = f"{p['draw']:.0%}"
            ext = f"{p['away']:.0%}"
            lines.append(
                f"📅 {_fmt_date(m['date'])} — **{m['home_team']} vs {m['away_team']}**\n"
                f"   🏠 {dom} · 🤝 {nul} · ✈️ {ext}"
            )

    if not results and not upcoming:
        lines.append("*Aucun match planifié trouvé pour ce groupe.*")

    return "\n".join(lines)


@app_commands.command(
    name="standings",
    description="Classement en temps réel d'un groupe CdM 2026",
)
@app_commands.describe(groupe="Groupe à consulter (A à L)")
@app_commands.choices(groupe=[
    app_commands.Choice(name=f"Groupe {g}", value=g)
    for g in "ABCDEFGHIJKL"
])
async def standings(interaction: discord.Interaction, groupe: app_commands.Choice[str]) -> None:
    """Affiche le classement du groupe avec les matchs joués et les prochaines prédictions."""
    await interaction.response.defer()
    import asyncio
    message = await asyncio.to_thread(_build_message, groupe.value)
    await interaction.followup.send(message)


def setup(tree: app_commands.CommandTree) -> None:
    """Enregistre la commande dans le command tree Discord."""
    tree.add_command(standings)
