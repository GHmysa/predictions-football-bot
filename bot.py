import os
from dotenv import load_dotenv

load_dotenv()

import discord
from discord import app_commands
from commands.stats import setup as setup_stats
from commands.prono import setup as setup_prono
import database

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

setup_stats(tree)
setup_prono(tree)


@client.event
async def on_ready():
    await tree.sync()
    print(f"Connecté en tant que {client.user}")


client.run(DISCORD_TOKEN)
