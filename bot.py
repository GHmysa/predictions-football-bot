import os
from dotenv import load_dotenv

load_dotenv()

import discord
from discord import app_commands
from commands.stats import setup as setup_stats
from commands.matchs import setup as setup_matchs
import database

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

setup_stats(tree)
setup_matchs(tree)


@client.event
async def on_ready():
    if GUILD_ID:
        guild = discord.Object(id=int(GUILD_ID))
        # Copie d'abord les commandes vers le guild
        tree.copy_global_to(guild=guild)
        synced = await tree.sync(guild=guild)
        # Puis vide les commandes globales sur Discord (supprime les doublons)
        tree.clear_commands(guild=None)
        await tree.sync()
        print(f"[SYNC] {len(synced)} commandes synced sur le guild :")
        for cmd in synced:
            print(f"  - /{cmd.name}")
    else:
        synced = await tree.sync()
        print(f"[SYNC] {len(synced)} commandes synced globalement")
    print(f"Connecté en tant que {client.user}")


client.run(DISCORD_TOKEN)
