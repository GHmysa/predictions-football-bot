import discord
from discord import app_commands
from datetime import datetime
from services.api_football import search_teams, fetch_fixtures

_RESULT_EMOJI = {"V": "✅", "N": "➖", "D": "❌"}


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


async def _send_fixtures(interaction: discord.Interaction, team_id: int, team_name: str):
    fixtures = await fetch_fixtures(team_id)
    if not fixtures:
        await interaction.response.send_message(
            f"Aucun match terminé trouvé pour **{team_name}**.", ephemeral=True
        )
        return
    await interaction.response.send_message(_build_response(team_name, fixtures))


class TeamSelect(discord.ui.Select):
    def __init__(self, teams: list[dict]):
        options = [
            discord.SelectOption(
                label=t["name"],
                description=t["country"] or "—",
                value=str(t["id"]),
            )
            for t in teams[:25]
        ]
        super().__init__(placeholder="Choisissez une équipe…", options=options)
        self._teams = {str(t["id"]): t["name"] for t in teams}

    async def callback(self, interaction: discord.Interaction):
        await _send_fixtures(interaction, int(self.values[0]), self._teams[self.values[0]])


class TeamSelectView(discord.ui.View):
    def __init__(self, teams: list[dict]):
        super().__init__(timeout=60)
        self.add_item(TeamSelect(teams))


@app_commands.command(name="stats", description="Affiche les 5 derniers matchs d'une équipe")
@app_commands.describe(equipe="Nom de l'équipe à rechercher")
async def stats(interaction: discord.Interaction, equipe: str):
    await interaction.response.defer()

    teams = await search_teams(equipe)
    if not teams:
        await interaction.followup.send(f"Aucune équipe trouvée pour **{equipe}**.")
        return

    if len(teams) == 1:
        t = teams[0]
        fixtures = await fetch_fixtures(t["id"])
        if not fixtures:
            await interaction.followup.send(f"Aucun match terminé trouvé pour **{t['name']}**.")
            return
        await interaction.followup.send(_build_response(t["name"], fixtures))
        return

    await interaction.followup.send(
        f"Plusieurs équipes trouvées pour **{equipe}**, sélectionnez-en une :",
        view=TeamSelectView(teams),
    )


def setup(tree: app_commands.CommandTree):
    tree.add_command(stats)
