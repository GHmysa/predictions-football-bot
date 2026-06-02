import discord
from discord import app_commands
import database

_COMPETITION_NAMES = {
    "WC2026": "Coupe du Monde 2026",
}


@app_commands.command(
    name="accuracy",
    description="Précision des prédictions ML sur les matchs résolus",
)
async def accuracy(interaction: discord.Interaction):
    """Affiche les statistiques globales et par compétition des prédictions."""
    await interaction.response.defer()

    stats = database.get_stats()

    if stats["total"] == 0:
        await interaction.followup.send(
            "📊 Aucune prédiction résolue pour l'instant.\n"
            "Les stats apparaîtront une fois les premiers matchs terminés."
        )
        return

    lines = [
        "## 📊 Précision des prédictions ML",
        "",
        f"**Matchs analysés** : {stats['total']}",
        f"🎯 **Résultat correct** : {stats['correct_results']}/{stats['total']} — **{stats['result_rate']}%**",
    ]

    if stats["by_competition"]:
        lines += ["", "**Par compétition :**"]
        for code, s in stats["by_competition"].items():
            name = _COMPETITION_NAMES.get(code, code)
            star = " ⭐" if code == stats["best_competition"] else ""
            lines.append(f"• {name}{star} : {s['correct_results']}/{s['total']} ({s['rate']}%)")

    await interaction.followup.send("\n".join(lines))


def setup(tree: app_commands.CommandTree) -> None:
    """Enregistre la commande dans le command tree Discord."""
    tree.add_command(accuracy)
