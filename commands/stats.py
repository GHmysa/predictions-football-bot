import discord
from discord import app_commands
from datetime import datetime
from services.api_football import search_teams, fetch_fixtures


def _format_fixture(f: dict, team_id: int = None) -> str:
    date = datetime.fromisoformat(f["date"]).strftime("%d/%m/%Y")
    home, away = f["home"], f["away"]
    hg, ag = f["home_goals"], f["away_goals"]

    if f["home_winner"] is True:
        winning_side = "home"
    elif f["home_winner"] is False:
        winning_side = "away"
    else:
        winning_side = "draw"

    return f"✅ {date} — {home} {hg} - {ag} {away}"


async def _send_fixtures(interaction: discord.Interaction, team_id: int, team_name: str):
    fixtures = await fetch_fixtures(team_id)
    if not fixtures:
        await interaction.response.send_message(
            f"Aucun match terminé trouvé pour **{team_name}** en 2024.", ephemeral=True
        ) 
        return

    lines = [f"**5 derniers matchs de {team_name}**"]
    for f in fixtures:
        lines.append(_format_fixture(f))

    await interaction.response.send_message("\n".join(lines))


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
        team_id = int(self.values[0])
        team_name = self._teams[self.values[0]]
        await _send_fixtures(interaction, team_id, team_name)


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
            await interaction.followup.send(
                f"Aucun match terminé trouvé pour **{t['name']}** en 2024."
            )
            return
        lines = [f"**5 derniers matchs de {t['name']}**"]
        for f in fixtures:
            lines.append(_format_fixture(f))
        await interaction.followup.send("\n".join(lines))
        return

    view = TeamSelectView(teams)
    await interaction.followup.send(
        f"Plusieurs équipes trouvées pour **{equipe}**, sélectionnez-en une :",
        view=view,
    )


def setup(tree: app_commands.CommandTree):
    tree.add_command(stats)
