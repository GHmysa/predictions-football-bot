# ⚽ Football Prediction Bot — CdM 2026

Bot Discord de prédiction de matchs de football avec pipeline ML complet, construit comme projet portfolio ML Engineer.

Le bot prédit les résultats des matchs de la **Coupe du Monde 2026** à l'aide d'un modèle XGBoost entraîné sur 150 ans de matchs internationaux.

---

## Commandes Discord

| Commande | Description |
|---|---|
| `/prono groupe:X` | Sélecteur de match pour un groupe CdM 2026 (A→L) → prédiction ML avec probabilités |
| `/stats ligue:X` | Statistiques d'une équipe en club (5 derniers matchs, forme) |
| `/accuracy` | Précision globale des prédictions enregistrées en base |

---

## Pipeline ML

```
ml/data/results.csv          (Mart Jürisoo — matchs internationaux 1872→2024)
         │
         ▼
ml/features.py    →  ELO chronologique + forme 5 matchs + H2H  →  features.csv
         │
         ▼
ml/train.py       →  XGBoost  (train <2018 / val 2018-2021 / test 2022-2024)  →  model.pkl
         │
         ▼
ml/run_wc2026.py  →  72 matchs du groupe stage  →  wc2026_predictions.csv
```

**Lancer le pipeline complet :**
```bash
python -m ml.pipeline           # skip si model.pkl existe déjà
python -m ml.pipeline --force   # tout recalculer
```

### Features

| Feature | Description |
|---|---|
| `elo_home / elo_away / elo_diff` | ELO calculé depuis 1872 avec K variable (40 CdM / 30 qualif / 20 amical) |
| `home/away_form_pts/gf/ga/n` | Forme sur les 5 derniers matchs (taux de victoire, buts marqués/encaissés) |
| `h2h_home_pts / h2h_gd / h2h_n` | Historique des 5 derniers duels directs |
| `is_neutral` | 1 pour tous les matchs CdM (terrain neutre) |
| `tournament_tier` | 4=CdM, 3=Continental, 2=Qualification, 1=Amical |

### Évaluation

Split temporel strict (pas de shuffle sur séries temporelles) :
- **Baseline** : règle naïve ELO (prédit toujours le favori ELO)
- **XGBoost** : accuracy + log-loss sur le set de test 2022-2024
- Métriques complètes dans `ml/metrics.json`

---

## Stack technique

| Couche | Technologie |
|---|---|
| Bot Discord | discord.py 2.3.2 |
| ML | XGBoost 2.0.3 · scikit-learn 1.4.0 |
| Data | pandas 2.2.0 · Mart Jürisoo dataset |
| HTTP async | httpx 0.27.2 |
| Base de données | SQLite (prédictions + cache) |
| Hébergement | Railway |
| Runtime | Python 3.12 |

---

## Structure du projet

```
predictions-football-bot/
├── bot.py                       # Point d'entrée Discord
├── database.py                  # SQLite — cache pronos, prédictions, stats
│
├── commands/
│   ├── prono.py                 # /prono — sélecteur groupe CdM → prédiction ML
│   ├── stats.py                 # /stats — stats équipe club
│   └── accuracy.py              # /accuracy — précision historique
│
├── services/
│   ├── ml_model.py              # Wrapper ML → message Discord formaté
│   ├── api_football.py          # Client football-data.org (stats clubs)
│   ├── ai_call.py               # Claude / Mistral (non utilisé pour CdM)
│   └── resolver.py              # Job horaire — résolution des prédictions passées
│
└── ml/
    ├── pipeline.py              # Orchestrateur : features → train → prédictions
    ├── features.py              # Feature engineering (ELO, forme, H2H)
    ├── train.py                 # Entraînement XGBoost + évaluation
    ├── predict.py               # Inférence sur un match unique
    ├── run_wc2026.py            # Prédictions batch groupe stage CdM 2026
    ├── model.pkl                # Modèle sérialisé (requis par le bot)
    ├── metrics.json             # Métriques d'évaluation
    └── data/
        ├── results.csv          # Dataset Mart Jürisoo (1872→2024)
        ├── elo_history.csv      # ELO calculé après chaque match
        ├── wc2026_fixtures.csv  # Calendrier officiel CdM 2026
        ├── wc2026_teams.csv     # Mapping noms FIFA → dataset
        └── wc2026_predictions.csv  # Prédictions groupe stage
```

---

## Installation locale

### 1. Cloner et installer les dépendances

```bash
git clone <repo>
cd predictions-football-bot
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Variables d'environnement

Créer un fichier `.env` :

```env
DISCORD_TOKEN=your_discord_bot_token
GUILD_ID=your_discord_server_id

FOOTBALL_DATA_KEY=your_football_data_org_key

MISTRAL_API_KEY=your_mistral_api_key       # optionnel
ANTHROPIC_API_KEY=your_anthropic_api_key   # optionnel
AI_PROVIDER=mistral
```

### 3. Entraîner le modèle (première fois)

```bash
python -m ml.pipeline
```

### 4. Lancer le bot

```bash
python bot.py
```

---

## Limites connues

| Limite | Détail |
|---|---|
| **Modèle figé** | L'ELO et la forme ne se mettent pas à jour pendant la CdM. Pour intégrer les résultats en cours : ajouter les matchs à `results.csv` puis `python -m ml.pipeline --force` |
| **Résolution WC automatique** | `resolver.py` interroge football-data.org — les matchs CdM (IDs 200001+) n'y sont pas. Les prédictions WC ne s'auto-résolvent pas via `/accuracy` |
| **Phases éliminatoires** | `/prono` ne couvre que le groupe stage (équipes TBD pour le reste) |
| **Pas de prédiction de score** | Le modèle prédit H/D/A, pas un score exact |

---

## Licence

MIT
