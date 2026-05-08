import os
from dotenv import load_dotenv

load_dotenv()

import discord
from discord import app_commands
from commands.stats import setup as setup_stats
from commands.prono import setup as setup_prono
from commands.matchs import setup as setup_matchs
import database

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

setup_stats(tree)
setup_prono(tree)
setup_matchs(tree)


@client.event
async def on_ready():
    if GUILD_ID:
        guild = discord.Object(id=int(GUILD_ID))
        tree.copy_global_to(guild=guild)
        await tree.sync(guild=guild)
    else:
        await tree.sync()
    print(f"Connecté en tant que {client.user}")


client.run(DISCORD_TOKEN)
