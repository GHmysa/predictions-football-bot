import discord
from discord import app_commands
from datetime import datetime
from services.api_football import fetch_upcoming_fixtures, fetch_fixtures
from services.ai_call import generate_prono
import database


class MatchSelect(discord.ui.Select):
    def __init__(self, fixtures: list[dict]):
        self._fixtures = {str(f["fixture_id"]): f for f in fixtures}
        options = []
        for f in fixtures:
            date = datetime.fromisoformat(f["date"]).strftime("%d/%m %H:%M")
            label = f"{f['home_team']} vs {f['away_team']}"
            options.append(
                discord.SelectOption(
                    label=label[:100],
                    description=date,
                    value=str(f["fixture_id"]),
                )
            )
        super().__init__(placeholder="Choisissez un match pour le pronostic…", options=options)

    async def callback(self, interaction: discord.Interaction):
        fixture = self._fixtures[self.values[0]]
        fixture_id = fixture["fixture_id"]
        home = fixture["home_team"]
        away = fixture["away_team"]
        header = f"**Pronostic : {home} vs {away}**\n\n"

        cached = database.get_cached_prono(fixture_id)
        if cached:
            await interaction.response.edit_message(
                content=f"**Pronostic (cache) : {home} vs {away}**\n\n{cached}", view=None
            )
            return

        await interaction.response.edit_message(content="⏳ Génération du pronostic…", view=None)

        home_fixtures = await fetch_fixtures(fixture["home_team_id"])
        away_fixtures = await fetch_fixtures(fixture["away_team_id"])

        try:
            result = generate_prono(home, home_fixtures, away, away_fixtures)
        except Exception as e:
            await interaction.followup.send(f"❌ Erreur lors de la génération du pronostic : {e}")
            return

        database.save_prono(fixture_id, home, away, result)
        await interaction.followup.send(header + result)


class MatchSelectView(discord.ui.View):
    def __init__(self, fixtures: list[dict]):
        super().__init__(timeout=120)
        self.add_item(MatchSelect(fixtures))


@app_commands.command(name="matchs", description="Affiche les prochains matchs d'une ligue et génère un pronostic")
@app_commands.describe(ligue="Ligue à consulter")
@app_commands.choices(ligue=[
    app_commands.Choice(name="Ligue 1", value="FL1"),
    app_commands.Choice(name="Premier League", value="PL"),
    app_commands.Choice(name="Liga", value="PD"),
    app_commands.Choice(name="Bundesliga", value="BL1"),
    app_commands.Choice(name="Serie A", value="SA"),
    app_commands.Choice(name="Champions League", value="CL"),
])
async def matchs(interaction: discord.Interaction, ligue: app_commands.Choice[str]):
    await interaction.response.defer()

    fixtures = await fetch_upcoming_fixtures(ligue.value)

    if not fixtures:
        await interaction.followup.send(f"Aucun match à venir trouvé pour **{ligue.name}**.")
        return

    view = MatchSelectView(fixtures)
    await interaction.followup.send(f"**{ligue.name}** — Prochains matchs :", view=view)


def setup(tree: app_commands.CommandTree):
    tree.add_command(matchs)
