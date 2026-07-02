"""
commands/standings.py — /standings : classement groupe ou bracket KO CdM 2026.
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

FIXTURES_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"

KO_STAGES = ["Round of 32", "Round of 16", "Quarter Finals", "Semi Finals", "Finals"]
KO_EMOJI  = {
    "Round of 32":    "⚫",
    "Round of 16":   "⚔️",
    "Quarter Finals": "🔥",
    "Semi Finals":    "🌟",
    "Finals":         "🏆",
}
KO_LABEL = {
    "Round of 32":    "1/32",
    "Round of 16":    "1/16",
    "Quarter Finals": "Quarts",
    "Semi Finals":    "Demies",
    "Finals":         "Finale",
}


@lru_cache(maxsize=1)
def _group_fixtures() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES_PATH)
    return df[df["stage"] == "Group Stage"].copy()


@lru_cache(maxsize=1)
def _ko_fixtures() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES_PATH)
    return df[df["stage"] != "Group Stage"].sort_values("match_number").copy()


def _ko_results() -> dict[int, dict]:
    with database.get_connection() as conn:
        rows = conn.execute("""
            SELECT match_id, home_score, away_score
            FROM match_results
            WHERE match_id > 200072
        """).fetchall()
    return {r[0]: {"home_score": r[1], "away_score": r[2]} for r in rows}


# ── Bracket KO ────────────────────────────────────────────────────────────────

def _fmt_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d %b")


def _build_bracket() -> str:
    df      = _ko_fixtures()
    results = _ko_results()
    lines   = ["## 🏆 Bracket KO — CdM 2026"]

    for stage in KO_STAGES:
        stage_df = df[df["stage"] == stage]
        if stage_df.empty:
            continue

        match_ids   = {200_000 + int(n) for n in stage_df["match_number"]}
        played_ids  = {mid for mid in match_ids if mid in results}
        all_played  = len(played_ids) == len(match_ids)
        some_played = len(played_ids) > 0

        status = " · terminé" if all_played else (" · en cours" if some_played else "")
        label  = KO_LABEL[stage]
        emoji  = KO_EMOJI[stage]
        lines.append(f"\n**{emoji} {label}**{status}")

        for _, m in stage_df.sort_values("match_number").iterrows():
            mid  = 200_000 + int(m["match_number"])
            home = str(m["home_team"])
            away = str(m["away_team"])
            dt   = _fmt_date(str(m["date"]))

            if mid in results:
                r  = results[mid]
                hs = r["home_score"]
                as_ = r["away_score"]
                # KO draw = decided by penalties — bold both (winner unknown from 90min)
                if hs == as_:
                    lines.append(f"  ✅ {home} {hs}–{as_} {away} *(pen.)*")
                elif hs > as_:
                    lines.append(f"  ✅ **{home}** {hs}–{as_} {away}")
                else:
                    lines.append(f"  ✅ {home} {hs}–{as_} **{away}**")
            elif home == "To be announced" or away == "To be announced":
                lines.append(f"  🔜 {dt} · {home} — {away}")
            else:
                lines.append(f"  📅 {dt} · **{home}** vs **{away}**")

    return "\n".join(lines)


# ── Groupe stage ──────────────────────────────────────────────────────────────

def _group_teams(group: str) -> list[str]:
    df = _group_fixtures()[_group_fixtures()["group"] == group]
    seen, teams = set(), []
    for team in pd.concat([df["home_team"], df["away_team"]]):
        if team not in seen:
            seen.add(team)
            teams.append(team)
    return teams


def _group_schedule(group: str) -> list[dict]:
    df = _group_fixtures()[_group_fixtures()["group"] == group].sort_values("date")
    return df.to_dict("records")


def _compute_standings(teams: list[str], results: list[dict]) -> list[dict]:
    table = {
        t: {"team": t, "pts": 0, "played": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0}
        for t in teams
    }
    for r in results:
        home, away = r["home_team"], r["away_team"]
        hg, ag     = r["home_score"], r["away_score"]
        for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
            if team not in table:
                continue
            table[team]["played"] += 1
            table[team]["gf"]     += gf
            table[team]["ga"]     += ga
        if hg > ag:
            table[home]["pts"] += 3; table[home]["w"] += 1; table[away]["l"] += 1
        elif hg == ag:
            table[home]["pts"] += 1; table[home]["d"] += 1
            table[away]["pts"] += 1; table[away]["d"] += 1
        else:
            table[away]["pts"] += 3; table[away]["w"] += 1; table[home]["l"] += 1
    for t in table.values():
        t["gd"] = t["gf"] - t["ga"]
    return sorted(table.values(), key=lambda t: (-t["pts"], -t["gd"], -t["gf"], t["team"]))


def _build_group_message(group: str) -> str:
    teams    = _group_teams(group)
    results  = database.get_group_results(group)
    schedule = _group_schedule(group)

    played_ids    = {r["match_id"] for r in results}
    total_matches = len(schedule)
    played_count  = len(results)

    lines = [f"## 🏆 Groupe {group}"]

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
            gd_str    = f"+{s['gd']}" if s["gd"] > 0 else str(s["gd"])
            qualifier = " ✓" if i <= 2 else ""
            lines.append(
                f" {i}  {s['team']:<26} {s['pts']:>3}  {s['played']}  "
                f"{s['w']}  {s['d']}  {s['l']}   {s['gf']:>2}  {s['ga']:>2}  {gd_str:>3}{qualifier}"
            )
        lines.append("```")
        lines.append("**Résultats :**")
        for r in results:
            lines.append(
                f"✅ {r['home_team']} **{r['home_score']}–{r['away_score']}** "
                f"{r['away_team']}  *({_fmt_date(r['match_date'])})*"
            )
        lines.append("")

    upcoming = [m for m in schedule if (200_000 + int(m["match_number"])) not in played_ids]
    if upcoming:
        lines.append("**Prochains matchs :**")
        for m in upcoming:
            r   = predict_match(m["home_team"], m["away_team"], m["date"])
            p   = r["probabilities"]
            lines.append(
                f"📅 {_fmt_date(m['date'])} — **{m['home_team']} vs {m['away_team']}**\n"
                f"   🏠 {p['home']:.0%} · 🤝 {p['draw']:.0%} · ✈️ {p['away']:.0%}"
            )

    if not results and not upcoming:
        lines.append("*Aucun match trouvé pour ce groupe.*")

    return "\n".join(lines)


# ── Commande ──────────────────────────────────────────────────────────────────

@app_commands.command(
    name="standings",
    description="Classement d'un groupe ou bracket KO — CdM 2026",
)
@app_commands.describe(groupe="Groupe (A–L) ou Bracket KO")
@app_commands.choices(groupe=[
    app_commands.Choice(name="🏆 Bracket KO", value="KO"),
    *[app_commands.Choice(name=f"Groupe {g}", value=g) for g in "ABCDEFGHIJKL"],
])
async def standings(interaction: discord.Interaction, groupe: app_commands.Choice[str]) -> None:
    await interaction.response.defer()

    if groupe.value == "KO":
        message = await asyncio.to_thread(_build_bracket)
    else:
        message = await asyncio.to_thread(_build_group_message, groupe.value)

    await interaction.followup.send(message)


def setup(tree: app_commands.CommandTree) -> None:
    tree.add_command(standings)
