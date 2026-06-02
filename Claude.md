# predictions-football-bot — CLAUDE.md

## Project Overview

Discord bot for World Cup 2026 match predictions powered by a full ML pipeline.
**Portfolio project** targeting a **ML Engineer internship**.

The bot predicts match outcomes (home/draw/away) using XGBoost trained on 150 years of international football data. Predictions update in real time as WC matches are played (ELO recalculated after each match).

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.12.x |
| Discord | discord.py | 2.3.2 |
| HTTP (sync) | requests | 2.31.0 |
| Data | pandas | 2.2.0 |
| ML | scikit-learn | 1.4.0 |
| ML (boosting) | xgboost | 2.0.3 |
| Database | SQLite (built-in) | — |
| Env vars | python-dotenv | 1.0.0 |
| Football data | football-data.org API v4 | — |
| Deployment | Railway | — |
| Python pin | `.python-version` → `3.12` | — |

---

## Project Structure

```
predictions-football-bot/
│
├── bot.py                        # Discord client, 3 commands, hourly resolver task
├── database.py                   # SQLite — tables: predictions, match_results
├── Procfile                      # Railway: worker: python bot.py
├── requirements.txt              # Pinned deps
├── .python-version               # Python 3.12 pin for Railway
├── .env                          # Local secrets — never committed
│
├── commands/
│   ├── prono.py                  # /prono — group selector → match select → ML prediction
│   ├── standings.py              # /standings — live group table + upcoming ML predictions
│   └── accuracy.py               # /accuracy — prediction accuracy stats
│
├── services/
│   ├── ml_model.py               # format_result() — formats predict_match() dict → Discord message
│   ├── wc_resolver.py            # Hourly: fetch finished WC matches → resolve DB + update ELO
│   └── elo_updater.py            # Post-match ELO update → appends to wc_elo_updates.csv
│
└── ml/
    ├── pipeline.py               # Orchestrator: features → train → predictions (--force to rebuild)
    ├── features.py               # Feature engineering: ELO, form (5 matches), H2H, neutral, tier
    ├── train.py                  # XGBoost training, temporal split, baseline ELO, metrics
    ├── predict.py                # Single-match inference (lru_cache on data + model)
    ├── run_wc2026.py             # Batch predictions for all 72 group stage matches
    ├── model.pkl                 # Serialized model (required by bot at runtime)
    ├── metrics.json              # Evaluation results (accuracy, log-loss)
    └── data/
        ├── results.csv           # Mart Jürisoo dataset — 49k intl matches 1872-2024
        ├── elo_history.csv       # ELO after every historical match (never modified in prod)
        ├── wc_elo_updates.csv    # Live ELO delta for WC matches played (2 rows/match)
        ├── wc2026_fixtures.csv   # Official WC 2026 schedule (104 matches)
        ├── wc2026_teams.csv      # FIFA name → dataset name mapping
        └── wc2026_predictions.csv # Pre-generated group stage predictions
```

---

## ML Pipeline — Key Concepts

### ELO calculation
All teams start at 1500. K-factor: 40 (WC), 30 (qualifier), 20 (friendly). Computed chronologically since 1872. The ELO for match N uses only the ELO from match N-1 — strictly no leakage.

### Features (16 total)
`elo_home`, `elo_away`, `elo_diff` — primary strength signal
`home/away_form_pts/gf/ga/n` — form over last 5 matches (vectorized with `shift(1) + rolling(5)`)
`h2h_home_pts`, `h2h_gd`, `h2h_n` — last 5 head-to-head meetings
`is_neutral` — 1 for all WC matches
`tournament_tier` — 4=WC, 3=continental, 2=qualifier, 1=friendly

### Temporal split (hard rule — never shuffle time-series data)
- Train: `< 2018`
- Val: `2018–2021` (used for XGBoost early stopping)
- Test: `2022–2024` (touched once for the final verdict)

### Live ELO update flow (prod)
```
Match finished (football-data.org)
  → wc_resolver.resolve_wc_predictions()
      → database.resolve_prediction()       (accuracy tracking)
      → database.save_match_result()        (standings)
      → elo_updater.update_elo_with_match() (appends wc_elo_updates.csv)
          → ml.predict._data.cache_clear()  (next /prono uses fresh ELO)
```

---

## Environment Variables

| Variable | Used in | Description |
|---|---|---|
| `DISCORD_TOKEN` | `bot.py` | Discord bot token |
| `GUILD_ID` | `bot.py` | Discord server ID (optional — instant dev sync) |
| `FOOTBALL_DATA_KEY` | `services/wc_resolver.py` | football-data.org API key for match resolution |

---

## Development Rules

### Code style
- No comments explaining WHAT — only WHY when non-obvious
- No dead code, no unused imports
- One file at a time when generating — never multiple files without confirmation

### ML — No data leakage (hard rules)
- Always sort by date before any rolling or cumulative calculation
- Use `shift(1)` before `.rolling()` — current match must never see its own result
- Never include post-match information as a feature
- Split chronologically — never shuffle time-series data
- `TimeSeriesSplit` for cross-validation, never `KFold` with shuffling

### Git commits
- No `Co-Authored-By` lines — user does not want Claude visible on GitHub
- Conventional commits: `feat:`, `fix:`, `refactor:`, `chore:`
- Imperative mood, ≤ 72 chars subject line

### Bot / async rules
- All blocking calls (pandas, SQLite, requests) must use `asyncio.to_thread()`
- Never call `interaction.response` twice — use `followup.send` for subsequent messages
- `lru_cache` on expensive loads (model, data, fixtures) — one load per process lifetime

### What NOT to do
- Do not modify `elo_history.csv` in production — it is the immutable historical source
- Do not commit `wc_elo_updates.csv` — it is generated in prod at runtime
- Do not commit `features.csv` — it is a training artifact, regenerated by `pipeline.py`
- Do not use `train_test_split(shuffle=True)` on this dataset
- Do not silently swallow exceptions — always log with context
