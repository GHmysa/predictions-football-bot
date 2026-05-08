import discord
from discord import app_commands
from services.api_football import search_teams, fetch_fixtures
from services.ai_call import generate_prono


# — Select Menu helpers —

class TeamSelectForProno(discord.ui.Select):
    def __init__(self, teams: list[dict], label: str, state: dict, key: str):
        options = [
            discord.SelectOption(
                label=t["name"],
                description=t["country"] or "—",
                value=str(t["id"]),
            )
            for t in teams[:25]
        ]
        super().__init__(placeholder=f"Choisissez {label}…", options=options)
        self._teams = {str(t["id"]): t["name"] for t in teams}
        self._state = state
        self._key = key

    async def callback(self, interaction: discord.Interaction):
        self._state[self._key] = {
            "id": int(self.values[0]),
            "name": self._teams[self.values[0]],
        }
        self.disabled = True

        if self._state.get("home") and self._state.get("away"):
            await interaction.response.edit_message(
                content="⏳ Génération du pronostic…", view=None
            )
            await _run_prono(interaction, self._state["home"], self._state["away"])
        else:
            await interaction.response.edit_message(view=self.view)


class PronoSelectView(discord.ui.View):
    def __init__(self, home_teams: list[dict], away_teams: list[dict]):
        super().__init__(timeout=120)
        self._state: dict = {}

        if len(home_teams) == 1:
            self._state["home"] = {"id": home_teams[0]["id"], "name": home_teams[0]["name"]}
        else:
            self.add_item(
                TeamSelectForProno(home_teams, "l'équipe à domicile", self._state, "home")
            )

        if len(away_teams) == 1:
            self._state["away"] = {"id": away_teams[0]["id"], "name": away_teams[0]["name"]}
        else:
            self.add_item(
                TeamSelectForProno(away_teams, "l'équipe à l'extérieur", self._state, "away")
            )

    def both_resolved(self) -> bool:
        return bool(self._state.get("home") and self._state.get("away"))

    def home(self) -> dict:
        return self._state["home"]

    def away(self) -> dict:
        return self._state["away"]


# — Core logic —

async def _run_prono(interaction: discord.Interaction, home: dict, away: dict):
    home_fixtures, away_fixtures = (
        await fetch_fixtures(home["id"]),
        await fetch_fixtures(away["id"]),
    )

    try:
        result = generate_prono(home["name"], home_fixtures, away["name"], away_fixtures)
    except Exception as e:
        await interaction.followup.send(f"❌ Erreur lors de la génération du pronostic : {e}")
        return

    header = f"**Pronostic : {home['name']} vs {away['name']}**\n\n"
    await interaction.followup.send(header + result)


# — Slash command —

@app_commands.command(name="prono", description="Génère un pronostic pour un match via l'IA")
@app_commands.describe(
    equipe_domicile="Équipe jouant à domicile",
    equipe_exterieure="Équipe jouant à l'extérieur",
)
async def prono(
    interaction: discord.Interaction,
    equipe_domicile: str,
    equipe_exterieure: str,
):
    await interaction.response.defer()

    home_teams, away_teams = (
        await search_teams(equipe_domicile),
        await search_teams(equipe_exterieure),
    )

    if not home_teams:
        await interaction.followup.send(f"Aucune équipe trouvée pour **{equipe_domicile}**.")
        return
    if not away_teams:
        await interaction.followup.send(f"Aucune équipe trouvée pour **{equipe_exterieure}**.")
        return

    view = PronoSelectView(home_teams, away_teams)

    if view.both_resolved():
        await _run_prono(interaction, view.home(), view.away())
        return

    needs_selection = []
    if len(home_teams) > 1:
        needs_selection.append(f"**{equipe_domicile}** (domicile)")
    if len(away_teams) > 1:
        needs_selection.append(f"**{equipe_exterieure}** (extérieur)")

    msg = "Plusieurs équipes trouvées pour " + " et ".join(needs_selection) + " :"
    await interaction.followup.send(msg, view=view)


def setup(tree: app_commands.CommandTree):
    tree.add_command(prono)
