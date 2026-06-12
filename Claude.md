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
    ├── wc_features.py            # WC-specific features: FIFA rank, market value, WC titles
    ├── train.py                  # XGBoost training — train() général + train_wc() WC-only
    ├── poisson.py                # Dixon-Coles Poisson model — score prediction
    ├── predict.py                # Single-match inference (lru_cache on data + model)
    ├── run_wc2026.py             # Batch predictions for all 72 group stage matches
    ├── model.pkl                 # Modèle général (all matches, 20 features)
    ├── model_wc.pkl              # Modèle WC (WC 2002-2022, 24 features)
    ├── metrics.json              # Evaluation results (accuracy, log-loss)
    └── data/
        ├── results.csv           # Mart Jürisoo dataset — 49k intl matches 1872-2024
        ├── elo_history.csv       # ELO after every historical match (never modified in prod)
        ├── wc_elo_updates.csv    # Live ELO delta — stocké dans PERSISTENT_DIR en prod
        ├── wc2026_fixtures.csv   # Official WC 2026 schedule (104 matches)
        ├── wc2026_teams.csv      # FIFA name → dataset name mapping (48 équipes)
        ├── wc_teams_train.csv    # Features WC par équipe 2002-2022 (192 rows × 24 cols)
        ├── wc_teams_test.csv     # Features WC 2026 pour inférence (48 rows) — jamais en train
        ├── poisson_params.json   # Paramètres Dixon-Coles fittés (attack/defense/home_adv/rho)
        └── wc2026_predictions.csv # Pre-generated group stage predictions
```

---

## ML Pipeline — Key Concepts

### ELO calculation
All teams start at 1500. K-factor: 40 (WC), 30 (qualifier), 20 (friendly). Computed chronologically since 1872. The ELO for match N uses only the ELO from match N-1 — strictly no leakage.

### Deux modèles XGBoost
- **Général** (`model.pkl`) — 20 features, entraîné sur tous les matchs < 2018, utilisé pour les matchs hors-WC
- **WC** (`model_wc.pkl`) — 24 features, entraîné sur WC 2002-2014, utilisé automatiquement pour `tournament_tier=4`

### Features générales (20)
`elo_home`, `elo_away`, `elo_diff` — signal de force principal
`home/away_form_pts/gf/ga` — forme sur les 5 derniers matchs
`h2h_home_pts`, `h2h_gd`, `h2h_n` — 5 dernières confrontations directes
`is_neutral`, `tournament_tier` — contexte du match
`home/away_is_host` — avantage terrain WC
`home/away_wc_form_pts` — win rate en tournois majeurs (tier≥3)
`home/away_rest_days` — jours depuis le dernier match (cap 30)

### Features WC supplémentaires (4)
`rank_diff` — away_rank - home_rank (positif = home mieux classé FIFA)
`log_market_ratio` — log(home_value / away_value)
`wc_titles_diff`, `wc_participations_diff` — palmarès et expérience WC

### Score prediction — Dixon-Coles Poisson (`poisson.py`)
`lambda_home = attack_home × defense_away × home_adv`
Paramètres estimés par MLE sur 2477 matchs compétitifs depuis 2018.
Score prédit = maximum conditionnel à l'issue XGBoost (triangle home/draw/away).

### Temporal split — modèle général
- Train: `< 2018`
- Val: `2018–2021` (early stopping XGBoost)
- Test: `2022–2024`

### Temporal split — modèle WC
- Train: WC `2002–2014` (256 matchs)
- Val: WC `2018` (64 matchs)
- Test: WC `2022` (64 matchs)

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
- Do not commit `wc_elo_updates.csv` — generated in prod at runtime, stored in `PERSISTENT_DIR`
- Do not commit `features.csv` — training artifact, regenerated by `pipeline.py`
- Do not use `train_test_split(shuffle=True)` on this dataset
- Do not silently swallow exceptions — always log with context
- Do not use `wc_teams_test.csv` for training — inference only (WC 2026 predictions)
- Do not add `encoding='latin-1'` to results.csv reads — file is UTF-8, leave default

### Railway Volume
Ajouter une variable d'environnement `PERSISTENT_DIR=/data` dans Railway et monter un volume sur `/data`.
`football.db` et `wc_elo_updates.csv` y seront stockés et survivront aux redéploiements.
