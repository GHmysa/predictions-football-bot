# Football Prediction Bot — CdM 2026

Bot Discord de prédiction de matchs de la Coupe du Monde 2026, construit sur un pipeline ML complet entraîné sur 150 ans de matchs internationaux.

---

## Commandes Discord

| Commande | Description |
|---|---|
| `/prono groupe:X` | Sélecteur de match pour un groupe (A→L) → prédiction ML + score prédit |
| `/standings groupe:X` | Classement en temps réel + prédictions pour les matchs à venir |
| `/accuracy` | Précision des prédictions ML sur les matchs résolus |
| `/simulate groupe:X` | Simulation Monte Carlo (10 000 itérations) — probabilités de qualification |
| `/score` *(admin)* | Saisie manuelle du score d'un match → met à jour DB + ELO |

---

## Architecture ML

### Deux modèles selon le contexte

| Contexte | Issue (H/D/A) | Score |
|---|---|---|
| **Matchs CdM** (`tournament_tier=4`) | Poisson Dixon-Coles — 54.7% sur WC 2022 | Conditionné à l'issue Poisson |
| **Matchs ordinaires** | XGBoost général — 55.2% sur 2022-2024 | Conditionné à l'issue XGBoost |

Le Poisson outperforme XGBoost sur les matchs WC (+7.8 pp) car il capte la force individuelle attack/defense de chaque équipe, mieux adaptée à un tournoi où tous les participants sont de haut niveau.

### Features XGBoost (20)

| Feature | Description |
|---|---|
| `elo_home/away/diff` | ELO calculé depuis 1872, K variable (40=CdM / 30=qualif / 20=amical) |
| `home/away_form_pts/gf/ga` | Forme sur les 5 derniers matchs |
| `h2h_home_pts/gd/n` | Historique des 5 derniers duels directs |
| `is_neutral` | 1 pour tous les matchs CdM |
| `tournament_tier` | 4=CdM, 3=Continental, 2=Qualif, 1=Amical |
| `home/away_is_host` | Avantage terrain pour USA, Canada, Mexique |
| `home/away_wc_form_pts` | Win rate dans les tournois majeurs (tier ≥ 3) |
| `home/away_rest_days` | Jours depuis le dernier match (plafond 30) |

**Règle fondamentale anti-leakage** : toutes les features sont calculées avec uniquement les données **strictement antérieures** au match (`shift(1)` avant tout `rolling()`).

### Modèle Poisson — Dixon-Coles

`λ_home = attack_home × defense_away × home_adv`

- Paramètres estimés par MLE sur 2 477 matchs compétitifs depuis 2018 (filtrés aux équipes WC 2026)
- 180 équipes avec des ratings attack/defense individuels
- Correction Dixon-Coles (τ) sur les bas scores (0-0, 1-0, 0-1, 1-1)
- Score prédit = argmax de la score_matrix **dans le triangle correspondant à l'issue prédite**

### Mise à jour ELO en temps réel

Les scores sont saisis manuellement via `/score` après chaque match. La commande :
1. Enregistre le score en base (`match_results`) → `/standings` se met à jour
2. Résout la prédiction DB (`predictions`) → `/accuracy` se met à jour
3. Recalcule les ELO et les appende dans `wc_elo_updates.csv`
4. Invalide le cache `predict.py` → les prochains `/prono` utilisent l'ELO à jour

`elo_history.csv` (1872-2024) n'est **jamais modifié**. `wc_elo_updates.csv` est le delta léger du tournoi.

### Pipeline offline

```bash
python -m ml.train          # entraîne model.pkl + model_wc.pkl
python -m ml.run_wc2026     # génère wc2026_predictions.csv (72 matchs)
```

---

## Stack technique

| Couche | Technologie |
|---|---|
| Bot Discord | discord.py 2.3.2 |
| ML | XGBoost 2.0.3 · scikit-learn 1.4.0 |
| Optimisation | scipy (MLE Dixon-Coles) |
| Data | pandas 2.2.0 |
| Base de données | SQLite — `predictions` + `match_results` |
| Hébergement | Railway (volume `/data` pour la persistance) |
| Runtime | Python 3.12 |

---

## Structure du projet

```
predictions-football-bot/
│
├── bot.py                        # Discord client, 5 commandes, prefill prédictions au démarrage
├── database.py                   # SQLite — predictions, match_results, stats
│
├── commands/
│   ├── prono.py                  # /prono  — sélecteur groupe → prédiction ML + score
│   ├── standings.py              # /standings — classement temps réel
│   ├── accuracy.py               # /accuracy — précision des prédictions
│   ├── simulate.py               # /simulate — Monte Carlo qualification
│   └── admin.py                  # /score  — saisie manuelle score (admin)
│
├── services/
│   ├── ml_model.py               # format_result() — dict predict_match → message Discord
│   └── elo_updater.py            # Calcul ELO post-match → wc_elo_updates.csv
│
└── ml/
    ├── pipeline.py               # Orchestrateur : features → train → prédictions
    ├── features.py               # Feature engineering (ELO, forme, H2H, host, rest)
    ├── wc_features.py            # Features WC (FIFA rank, market value, palmarès)
    ├── train.py                  # XGBoost — train() général + train_wc() WC
    ├── poisson.py                # Dixon-Coles Poisson — score prediction
    ├── predict.py                # Inférence match unique (lru_cache données + modèles)
    ├── simulator.py              # Monte Carlo groupe + tournoi complet
    ├── run_wc2026.py             # Prédictions batch groupe stage
    ├── model.pkl                 # Modèle général sérialisé
    ├── model_wc.pkl              # Modèle WC sérialisé
    └── data/
        ├── results.csv           # Mart Jürisoo — matchs internationaux 1872-2024
        ├── elo_history.csv       # ELO calculé après chaque match historique (immuable)
        ├── wc2026_fixtures.csv   # Calendrier officiel CdM 2026 (104 matchs)
        ├── wc2026_teams.csv      # Mapping noms FIFA → noms dataset
        ├── wc_teams_train.csv    # Features WC par équipe 2002-2022 (entraînement)
        ├── wc_teams_test.csv     # Features WC 2026 (inférence uniquement)
        ├── poisson_params.json   # Paramètres Dixon-Coles fittés (180 équipes)
        └── wc2026_predictions.csv # Prédictions groupe stage pré-générées
```

---

## Installation locale

```bash
git clone <repo> && cd predictions-football-bot
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

Créer `.env` :
```env
DISCORD_TOKEN=...
GUILD_ID=...       # optionnel — sync instantané en dev
```

Lancer le bot :
```bash
python bot.py
```

---

## Déploiement Railway

1. Connecter le repo GitHub dans Railway
2. Créer un volume monté sur `/data`
3. Ajouter les variables d'environnement :

| Variable | Description |
|---|---|
| `DISCORD_TOKEN` | Token du bot Discord |
| `GUILD_ID` | ID du serveur (optionnel) |
| `PERSISTENT_DIR` | `/data` — chemin du volume Railway |

`football.db` et `wc_elo_updates.csv` sont stockés dans `PERSISTENT_DIR` et survivent aux redéploiements.

---

## Limites connues

| Limite | Détail |
|---|---|
| **Phases éliminatoires** | `/prono` couvre uniquement le groupe stage. Pour les KO, mettre à jour `wc2026_fixtures.csv` avec les équipes qualifiées |
| **Modèle WC = baseline** | 48.4% d'accuracy sur WC 2022 — égalité avec la règle naïve ELO. Normal avec 256 matchs d'entraînement |
| **Saisie manuelle** | Les scores sont entrés via `/score` après chaque match — pas de résolution automatique |

---

## Licence

MIT
