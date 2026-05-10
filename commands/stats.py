import discord
from discord import app_commands
from datetime import datetime
from services.api_football import fetch_competition_teams, fetch_fixtures
import database

_RESULT_EMOJI = {"V": "✅", "N": "➖", "D": "❌"}


async def _get_teams(competition_code: str) -> list[dict]:
    cached = database.get_cached_teams(competition_code)
    if cached is not None:
        return cached
    teams = await fetch_competition_teams(competition_code)
    database.save_teams(competition_code, teams)
    return teams


def _format_fixture(f: dict) -> str:
    date = datetime.fromisoformat(f["date"]).strftime("%d/%m/%Y")
    emoji = _RESULT_EMOJI.get(f["result"], "❓")
    return f"{emoji} {date} — {f['home']} {f['home_goals']} - {f['away_goals']} {f['away']}"


def _form_summary(fixtures: list[dict]) -> str:
    wins   = sum(1 for f in fixtures if f["result"] == "V")
    draws  = sum(1 for f in fixtures if f["result"] == "N")
    losses = sum(1 for f in fixtures if f["result"] == "D")
    scored   = sum(f["team_goals"] for f in fixtures)
    conceded = sum(f["opponent_goals"] for f in fixtures)

    last3 = fixtures[:3]
    w3 = sum(1 for f in last3 if f["result"] == "V")
    l3 = sum(1 for f in last3 if f["result"] == "D")
    if w3 >= 2:
        trend = "En hausse 📈"
    elif l3 >= 2:
        trend = "En baisse 📉"
    else:
        trend = "Stable ➡️"

    return (
        f"\n📊 **Résumé de forme**\n"
        f"Bilan : {wins}V {draws}N {losses}D | {scored} buts marqués, {conceded} encaissés\n"
        f"Tendance (3 derniers) : {trend}"
    )


def _build_response(team_name: str, fixtures: list[dict]) -> str:
    lines = [f"**5 derniers matchs de {team_name}**"]
    for f in fixtures:
        lines.append(_format_fixture(f))
    lines.append(_form_summary(fixtures))
    return "\n".join(lines)


class TeamSelect(discord.ui.Select):
    def __init__(self, teams: list[dict]):
        self._teams = {str(t["id"]): t["name"] for t in teams}
        options = [
            discord.SelectOption(label=t["name"], value=str(t["id"]))
            for t in teams[:25]
        ]
        super().__init__(placeholder="Choisissez une équipe…", options=options)

    async def callback(self, interaction: discord.Interaction):
        team_id = int(self.values[0])
        team_name = self._teams[self.values[0]]
        fixtures = await fetch_fixtures(team_id)
        if not fixtures:
            await interaction.response.edit_message(
                content=f"Aucun match terminé trouvé pour **{team_name}**.", view=None
            )
            return
        await interaction.response.edit_message(
            content=_build_response(team_name, fixtures), view=None
        )


class TeamSelectView(discord.ui.View):
    def __init__(self, teams: list[dict]):
        super().__init__(timeout=60)
        self.add_item(TeamSelect(teams))


@app_commands.command(name="stats", description="Affiche les 5 derniers matchs d'une équipe")
@app_commands.describe(ligue="Ligue à consulter")
@app_commands.choices(ligue=[
    app_commands.Choice(name="Ligue 1",          value="FL1"),
    app_commands.Choice(name="Premier League",   value="PL"),
    app_commands.Choice(name="Liga",             value="PD"),
    app_commands.Choice(name="Bundesliga",       value="BL1"),
    app_commands.Choice(name="Serie A",          value="SA"),
    app_commands.Choice(name="Champions League", value="CL"),
])
async def stats(interaction: discord.Interaction, ligue: app_commands.Choice[str]):
    await interaction.response.defer()

    teams = await _get_teams(ligue.value)
    if not teams:
        await interaction.followup.send(f"Aucune équipe trouvée pour **{ligue.name}**.")
        return

    teams_sorted = sorted(teams, key=lambda t: t["name"])
    view = TeamSelectView(teams_sorted)
    await interaction.followup.send(
        f"**{ligue.name}** — Choisissez une équipe :", view=view
    )


def setup(tree: app_commands.CommandTree):
    tree.add_command(stats)
