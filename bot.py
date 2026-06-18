import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

import discord
from discord import app_commands

from commands.prono import setup as setup_prono, _fixtures, MATCH_ID_OFFSET, WC_COMPETITION
from commands.accuracy import setup as setup_accuracy
from commands.standings import setup as setup_standings
from commands.simulate import setup as setup_simulate
from commands.admin import setup as setup_admin
from ml.predict import predict_match
from ml.poisson import fit_or_load as _poisson_params
import database

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID      = os.getenv("GUILD_ID")

intents = discord.Intents.default()
client  = discord.Client(intents=intents)
tree    = app_commands.CommandTree(client)

setup_prono(tree)
setup_accuracy(tree)
setup_standings(tree)
setup_simulate(tree)
setup_admin(tree)


async def _prefill_predictions() -> None:
    def _sync() -> int:
        with database.get_connection() as conn:
            existing = {
                r[0] for r in conn.execute("SELECT match_id FROM predictions").fetchall()
            }
        count = 0
        for _, row in _fixtures().iterrows():
            match_id = MATCH_ID_OFFSET + int(row["match_number"])
            if match_id in existing:
                continue
            result = predict_match(row["home_team"], row["away_team"], row["date"], True, 4)
            pred = result["prediction"]
            database.save_prediction(
                match_id, WC_COMPETITION,
                row["home_team"], row["away_team"],
                1 if pred == "home" else 0,
                1 if pred == "away" else 0,
            )
            count += 1
        return count

    count = await asyncio.to_thread(_sync)
    print(f"[PREFILL] {count} prédictions pré-remplies en DB")


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

    def _startup_log() -> None:
        # Déclenche le chargement des params Poisson et log l'état
        params = _poisson_params()
        with database.get_connection() as conn:
            n_results = conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
            n_preds   = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            n_resolved = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE actual_result IS NOT NULL"
            ).fetchone()[0]
        print(
            f"[STARTUP] DB : {n_results} matchs joues | "
            f"{n_resolved}/{n_preds} predictions resolues"
        )
        print(
            f"[STARTUP] Poisson : {params['n_matches']} matchs | "
            f"ref_date={params.get('ref_date','?')} | home_adv={params['home_adv']:.3f}"
        )

    await asyncio.to_thread(_startup_log)
    await _prefill_predictions()
    print(f"[STARTUP] Bot pret : {client.user}")


client.run(DISCORD_TOKEN)
