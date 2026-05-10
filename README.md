# ⚽ Discord Football Bot

A Discord bot that delivers real-time football statistics and AI-generated match predictions directly in your server. Powered by the football-data.org API for live data and Mistral AI (with Claude as a production fallback) for analytical predictions.

---

## Commands

| Command | Description |
|---|---|
| `/stats <league>` | Browse teams from a league via a select menu and display their last 5 results with a form summary (W/D/L, goals, trend) |
| `/prono <league>` | Browse upcoming fixtures from a league and generate an AI-powered match prediction with stats and confidence score |

### League options (both commands)
`Ligue 1` · `Premier League` · `Liga` · `Bundesliga` · `Serie A` · `Champions League`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Bot framework | [discord.py](https://github.com/Rapptz/discord.py) 2.x |
| Football data | [football-data.org](https://www.football-data.org/) v4 API |
| AI predictions | [Mistral AI](https://mistral.ai/) (`mistral-large-latest`) / [Anthropic Claude](https://www.anthropic.com/) (`claude-sonnet-4`) |
| Database | SQLite via `sqlite3` (prediction cache + team cache) |
| HTTP client | `httpx` (async) · `requests` (sync AI calls) |
| Runtime | Python 3.12+ |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/discord-football-bot.git
cd discord-football-bot
```

### 2. Install dependencies

```bash
pip install discord.py httpx requests python-dotenv
```

### 3. Configure environment variables

Copy the example below into a `.env` file at the project root and fill in your keys:

```env
DISCORD_TOKEN=your_discord_bot_token
GUILD_ID=your_discord_server_id        # optional — enables instant command sync during development

FOOTBALL_DATA_KEY=your_football_data_org_key

MISTRAL_API_KEY=your_mistral_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

AI_PROVIDER=mistral                    # set to "claude" for production
```

| Variable | Where to get it |
|---|---|
| `DISCORD_TOKEN` | [discord.com/developers/applications](https://discord.com/developers/applications) → Bot → Reset Token |
| `GUILD_ID` | Discord app → Settings → Enable Developer Mode → right-click your server → Copy Server ID |
| `FOOTBALL_DATA_KEY` | [football-data.org/client/register](https://www.football-data.org/client/register) |
| `MISTRAL_API_KEY` | [console.mistral.ai](https://console.mistral.ai/) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |

### 4. Invite the bot to your server

In the Discord Developer Portal, go to **OAuth2 → URL Generator**, select the `bot` and `applications.commands` scopes, grant **Send Messages** and **Use Slash Commands** permissions, then open the generated URL.

### 5. Run

```bash
python bot.py
```

On first start the bot syncs slash commands to your guild (instant if `GUILD_ID` is set) and initialises the SQLite database (`football.db`).

---

## Project Structure

```
discord-football-bot/
├── bot.py                  # Entry point — client setup, command registration, sync
├── database.py             # SQLite helpers — prediction cache + team cache (24h TTL)
├── commands/
│   ├── stats.py            # /stats — league picker → team select → form report
│   └── prono.py            # /prono — league picker → fixture select → AI prediction
└── services/
    ├── api_football.py     # football-data.org client (teams, fixtures, upcoming matches)
    └── ai_call.py          # AI provider abstraction (Mistral / Claude)
```

---

## Caching

The bot uses SQLite to avoid redundant API calls:

| Cache | TTL | What is stored |
|---|---|---|
| `teams_cache` | 24 hours | Team roster per competition code |
| `prono_cache` | 24 hours | AI prediction per fixture ID |

---

## Roadmap

- [ ] **Premium tier via Stripe** — limit free users to N predictions/day, unlock unlimited access with a subscription
- [ ] **Railway deployment** — one-click cloud hosting with persistent SQLite volume and environment variable management
- [ ] **Head-to-head history** — enrich predictions with historical results between the two teams
- [ ] **Multi-language support** — French / English toggle per server
- [ ] **Webhook alerts** — notify a channel automatically when a prediction is available for tonight's matches

---

## License

MIT
