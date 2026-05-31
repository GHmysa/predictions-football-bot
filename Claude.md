# predictions-football-bot — CLAUDE.md

## Project Overview

Discord bot for football match predictions with an integrated ML pipeline.
**Portfolio project** targeting a **ML Engineer internship**.
The bot generates AI-powered pronostics via slash commands, tracks prediction accuracy in SQLite, and is backed by a supervised ML pipeline trained on historical international match data.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.12.x |
| Discord | discord.py | 2.3.2 |
| HTTP (async) | httpx | 0.27.2 |
| HTTP (sync) | requests | 2.31.0 |
| Web server | Flask | 3.1.1 |
| Payments | stripe | 12.2.0 |
| Data | pandas | 2.2.0 |
| ML | scikit-learn | 1.4.0 |
| ML (boosting) | xgboost | 2.0.3 |
| Visualization | matplotlib / seaborn | 3.8.0 / 0.13.2 |
| Database | SQLite (built-in) | — |
| Env vars | python-dotenv | 1.0.0 |
| AI providers | Anthropic Claude API / Mistral AI | — |
| Football data | football-data.org API v4 | — |
| Deployment | Railway | — |
| Python pin | `.python-version` → `3.12` | — |

---

## Project Structure

```
predictions-football-bot/
│
├── bot.py                      # Entry point — Discord client, slash command tree, hourly resolver loop
├── database.py                 # SQLite layer — all DB reads/writes (prono_cache, teams_cache, predictions)
├── Procfile                    # Railway process definition (worker: python bot.py)
├── requirements.txt            # Pinned dependencies — always keep in sync
├── .python-version             # Pins Python 3.12 for Railway builds
├── .env                        # Local secrets — never committed
│
├── commands/
│   ├── __init__.py
│   ├── prono.py                # /prono command — fixture selector UI + AI prono generation + DB save
│   ├── stats.py                # /stats command — accuracy display per competition
│   └── accuracy.py             # /accuracy command — global prediction accuracy
│
├── services/
│   ├── __init__.py
│   ├── api_football.py         # football-data.org async client (httpx) — fixtures, teams, upcoming matches
│   ├── ai_call.py              # Prompt builder + Claude/Mistral API calls — generates pronostic text
│   └── resolver.py             # Sync job — polls finished matches and resolves pending predictions in DB
│
└── ml/
    ├── data/
    │   ├── results.csv         # Mart Jürisoo dataset — international results since 1872
    │   ├── goalscorers.csv     # Goal events per match
    │   ├── shootouts.csv       # Penalty shootout outcomes
    │   └── former_names.csv    # Historical country name mappings
    └── notebooks/
        └── exploration.ipynb   # EDA, feature engineering, model training (Jupyter — kernel: Python 3.12 venv)
```

---

## ML Pipeline — Target Architecture (World Cup 2026)

### Data
- **Source**: Mart Jürisoo dataset (`results.csv`) — international matches 1872–present
- **Scope for training**: post-1990 (modern football era)
- **Target variable**: match result — `H` (home win), `D` (draw), `A` (away win)

### Features (planned)
| Feature | Description |
|---|---|
| `elo_home` / `elo_away` | ELO ratings computed in-house from historical results |
| `elo_diff` | `elo_home - elo_away` — primary strength signal |
| `form_home` / `form_away` | Points per game over last 5 matches (rolling, date-sorted) |
| `h2h_home_win_rate` | Head-to-head win rate (home team perspective) |
| `is_neutral` | 1 if neutral venue, 0 otherwise |
| `tournament` | Encoded competition type (World Cup, qualifier, friendly…) |
| `goals_scored_avg` / `goals_conceded_avg` | Rolling 5-match averages |

### Model
- **Algorithm**: XGBoost classifier (`multi:softprob`, 3 classes)
- **Hyperparameter tuning**: `GridSearchCV` or `Optuna`
- **Evaluation metrics**: accuracy, log loss, confusion matrix, calibration curve

### Train/Val/Test Split
- **Always temporal**: sort by date, then cut (e.g. train ≤ 2018, val 2018–2022, test ≥ 2022)
- **Never random split on time series** — this is a hard rule, see Development Rules

---

## Development Rules

> This is the most important section. Follow these directives strictly in every interaction.

### Code explanation
- **Always explain new logic line by line** when writing or modifying code.
- Treat every explanation as if the reader is learning — assume nothing is obvious.
- If a function has a non-trivial invariant or a subtle gotcha, add a one-line comment on the WHY (not the what).

### File generation
- **Never generate multiple files in a single response without asking first.**
- Propose the plan, wait for confirmation, then generate one file at a time.

### Bad practices
- **Flag bad practices immediately** when spotted, even if not asked. Explain why it is bad and offer a corrected version.

### Python environment
- **Always use the venv interpreter**: `c:\Users\asmax\ML foot pred\.venv\Scripts\python.exe`
- Never suggest `pip install` without the venv path or with the system Python.
- Jupyter kernel to use: **"Python 3.12 (venv)"**

### Commit conventions
```
feat:     new feature
fix:      bug fix
refactor: code change with no behavior change
chore:    deps, config, tooling, CI
```
One subject line, imperative mood, ≤ 72 chars.

### Secrets
- **Never hardcode API keys, tokens, or credentials.**
- Always use `os.getenv("VAR_NAME")` — never string literals.
- All secrets live in `.env` (local) or Railway environment variables (prod).

### API error handling
- **Always handle API errors explicitly** — never let `raise_for_status()` be the only guard.
- Catch `httpx.HTTPStatusError`, `requests.HTTPError`, and timeout exceptions separately.
- Log the error with context (`match_id`, `team`, `endpoint`) before re-raising or returning `None`.

### ML — No data leakage
- **Always sort by date before any rolling or cumulative calculation.**
- Rolling features must use only past data: `df.shift(1)` before `.rolling()`.
- Never include post-match information (final score, actual result) as a feature.
- Lag all target-correlated signals by at least one match.

### ML — Temporal split (hard rule)
- **Never use `train_test_split` with `shuffle=True` on time-series data.**
- Always split chronologically: `train → val → test` in date order.
- Cross-validation must be `TimeSeriesSplit`, never `KFold` with shuffling.

### Docstrings
- **Every function must have a docstring** — one line minimum.
- Format: `"""What the function returns and its key side effects."""`
- For ML functions: add input shape/type expectations and what leakage guard is applied.

---

## Current Focus

**World Cup 2026 ML pipeline** — building the full supervised pipeline in `ml/notebooks/exploration.ipynb`:
1. EDA on `results.csv` (score distributions, tournament breakdown, neutral venue bias)
2. ELO computation from scratch (iterative, date-sorted)
3. Feature engineering with strict no-leakage discipline
4. XGBoost baseline + evaluation
5. Integration path: trained model → `services/` → replace/augment AI prono call

---

## Environment Variables

| Variable | Used in | Description |
|---|---|---|
| `DISCORD_TOKEN` | `bot.py` | Discord bot token |
| `FOOTBALL_DATA_KEY` | `services/api_football.py` | football-data.org API key |
| `ANTHROPIC_API_KEY` | `services/ai_call.py` | Claude API key |
| `MISTRAL_API_KEY` | `services/ai_call.py` | Mistral AI API key |
| `AI_PROVIDER` | `services/ai_call.py` | `"mistral"` (default) or `"claude"` |
| `GUILD_ID` | `bot.py` | Discord server ID for guild-scoped command sync |
| `STRIPE_SECRET_KEY` | payments | Stripe secret key |
| `STRIPE_PRICE_ID` | payments | Stripe price/product ID |
| `STRIPE_WEBHOOK_SECRET` | payments | Stripe webhook signing secret |
| `PREMIUM_ROLE_ID` | payments | Discord role ID granted on payment |

---

## What NOT To Do

### General
- **Do not commit `.env`** — it is in `.gitignore`. Never add it.
- **Do not use `print()` for structured logging in production** — use it only for Railway console debug during dev.
- **Do not import inside function bodies** unless there is a circular-import reason — the `import json` inside `database.py` functions should be moved to the top.
- **Do not silently swallow exceptions** — `except Exception: pass` is never acceptable.

### Discord / async
- **Do not mix sync and async DB calls** without `asyncio.to_thread()` — SQLite is blocking.
- **Do not call `interaction.response` after it has already been used** — always use `followup.send` for subsequent messages in the same interaction.
- **Do not store mutable state on the `client` object** — use the database layer.

### API
- **Do not retry on rate limit with a sleep loop** — raise `RateLimitError` and let the caller decide.
- **Do not hardcode competition codes as magic strings** outside of the `@app_commands.choices` definition.

### ML
- **Do not use accuracy as the sole metric** for an imbalanced multi-class problem — always report log loss and per-class F1.
- **Do not fit scalers or encoders on the full dataset** before splitting — fit on train, transform val/test.
- **Do not use future ELO values as features** — ELO for a match must be computed from all matches strictly before that date.
- **Do not use `pd.DataFrame.sample()` to create train/test sets** on this dataset.
- **Do not use `fillna(0)` on rolling features without documenting why** — missing rolling values (first N rows) should be dropped or flagged explicitly.
