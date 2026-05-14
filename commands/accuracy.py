import discord
from discord import app_commands
import database

_LEAGUE_NAMES = {
    "FL1": "Ligue 1",
    "PL":  "Premier League",
    "PD":  "Liga",
    "BL1": "Bundesliga",
    "SA":  "Serie A",
    "CL":  "Champions League",
}


@app_commands.command(name="accuracy", description="Statistiques de précision des pronostics IA")
async def accuracy(interaction: discord.Interaction):
    await interaction.response.defer()

    stats = database.get_stats()

    if stats["total"] == 0:
        await interaction.followup.send(
            "📊 Aucun pronostic résolu pour l'instant.\n"
            "Les stats apparaîtront une fois les premiers matchs terminés."
        )
        return

    lines = [
        "## 📊 Précision des pronostics IA",
        "",
        f"**Pronos analysés** : {stats['total']}",
        f"🎯 **Résultat correct** : {stats['correct_results']}/{stats['total']} — **{stats['result_rate']}%**",
        f"⚽ **Score exact** : {stats['correct_scores']}/{stats['total']} — **{stats['score_rate']}%**",
    ]

    if stats["by_competition"]:
        lines += ["", "**Par compétition :**"]
        for code, s in stats["by_competition"].items():
            name = _LEAGUE_NAMES.get(code, code)
            star = " ⭐" if code == stats["best_competition"] else ""
            lines.append(f"• {name}{star} : {s['correct_results']}/{s['total']} ({s['rate']}%)")

    if stats["best_competition"]:
        best_name = _LEAGUE_NAMES.get(stats["best_competition"], stats["best_competition"])
        best_rate = stats["by_competition"][stats["best_competition"]]["rate"]
        lines += ["", f"🏆 **Meilleure ligue** : {best_name} ({best_rate}%)"]

    await interaction.followup.send("\n".join(lines))


def setup(tree: app_commands.CommandTree):
    tree.add_command(accuracy)
