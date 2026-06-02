import os
from dotenv import load_dotenv

load_dotenv()

import discord
from discord import app_commands
from commands.prono import setup as setup_prono
from commands.accuracy import setup as setup_accuracy
import database

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID      = os.getenv("GUILD_ID")

intents = discord.Intents.default()
client  = discord.Client(intents=intents)
tree    = app_commands.CommandTree(client)

setup_prono(tree)
setup_accuracy(tree)


@client.event
async def on_ready():
    if GUILD_ID:
        guild = discord.Object(id=int(GUILD_ID))
        tree.copy_global_to(guild=guild)
        synced = await tree.sync(guild=guild)
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
