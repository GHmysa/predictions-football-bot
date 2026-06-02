# Football Prediction Bot — CdM 2026

Bot Discord de prédiction de matchs de la Coupe du Monde 2026, construit sur un pipeline ML complet entraîné sur 150 ans de matchs internationaux.

---

## Commandes Discord

| Commande | Description |
|---|---|
| `/prono groupe:X` | Sélecteur de match pour un groupe (A→L) → prédiction ML avec barres de probabilité |
| `/standings groupe:X` | Classement en temps réel d'un groupe, avec résultats joués et prédictions à venir |
| `/accuracy` | Précision des prédictions ML sur les matchs résolus |

---

## Architecture ML

### Données
**Source** : dataset Mart Jürisoo — 49 287 matchs internationaux depuis 1872 (`results.csv`)

### Features
| Feature | Description |
|---|---|
| `elo_home/away/diff` | ELO calculé depuis 1872 avec K variable (40=CdM / 30=qualif / 20=amical). Seule vraie mesure de la force historique d'une équipe. |
| `home/away_form_pts/gf/ga` | Forme sur les 5 derniers matchs (taux de victoire, buts marqués/encaissés). Capture l'état récent indépendamment du rating ELO. |
| `h2h_home_pts/gd` | Historique des 5 derniers duels directs. Certaines équipes dominent systématiquement d'autres. |
| `is_neutral` | 1 pour tous les matchs CdM (terrain neutre). L'avantage du terrain est réel en football. |
| `tournament_tier` | 4=CdM, 3=Continental, 2=Qualification, 1=Amical. Les équipes ne jouent pas au même niveau selon l'enjeu. |

**Règle fondamentale anti-leakage** : toutes les features sont calculées avec uniquement les données **strictement antérieures** au match. `shift(1)` avant tout `rolling()`.

### Modèle
- **Algorithme** : XGBoost (`multi:softprob`, 3 classes : domicile / nul / extérieur)
- **Output** : 3 probabilités calibrées qui somment à 1
- **Split temporel strict** : train `<2018` / val `2018-2021` / test `2022-2024`
- **Baseline** : règle naïve ELO — le modèle doit la battre pour être utile
- **Métriques** : accuracy + log-loss sur le set de test (jamais utilisé pour tuner)

### Pipeline offline
```
python -m ml.pipeline           # skip si artifacts existent
python -m ml.pipeline --force   # tout recalculer
```
Enchaîne : `features.py` → `train.py` → `run_wc2026.py`

### Mise à jour en temps réel (pendant la CdM)
Toutes les heures, `auto_resolve` :
1. Interroge football-data.org → matchs terminés
2. Résout les prédictions en DB → `/accuracy` se met à jour
3. Enregistre les scores réels → `/standings` se met à jour
4. Recalcule les ELO et les appende dans `wc_elo_updates.csv`
5. Invalide le cache de `predict.py` → les prochains `/prono` utilisent l'ELO à jour

`elo_history.csv` (données 1872-2024) n'est **jamais modifié**. `wc_elo_updates.csv` est le delta léger du tournoi en cours.

---

## Stack technique

| Couche | Technologie |
|---|---|
| Bot Discord | discord.py 2.3.2 |
| ML | XGBoost 2.0.3 · scikit-learn 1.4.0 |
| Data | pandas 2.2.0 |
| HTTP sync | requests 2.31.0 (resolver) |
| Base de données | SQLite — `predictions` + `match_results` |
| Hébergement | Railway |
| Runtime | Python 3.12 |

---

## Structure du projet

```
predictions-football-bot/
│
├── bot.py                        # Discord client, 3 commandes, tâche horaire resolver
├── database.py                   # SQLite — predictions, match_results, stats
│
├── commands/
│   ├── prono.py                  # /prono  — sélecteur groupe → prédiction ML
│   ├── standings.py              # /standings — classement temps réel + prédictions à venir
│   └── accuracy.py               # /accuracy — précision des prédictions résolues
│
├── services/
│   ├── ml_model.py               # format_result() — dict predict_match → message Discord
│   ├── wc_resolver.py            # Résolution horaire via football-data.org + update ELO
│   └── elo_updater.py            # Calcul ELO post-match → wc_elo_updates.csv
│
└── ml/
    ├── pipeline.py               # Orchestrateur : features → train → prédictions
    ├── features.py               # Feature engineering (ELO, forme, H2H)
    ├── train.py                  # XGBoost, split temporel, baseline, métriques
    ├── predict.py                # Inférence match unique (lru_cache données + modèle)
    ├── run_wc2026.py             # Prédictions batch groupe stage
    ├── model.pkl                 # Modèle sérialisé (requis par le bot)
    ├── metrics.json              # Métriques d'évaluation (accuracy, log-loss)
    └── data/
        ├── results.csv           # Mart Jürisoo — matchs internationaux 1872-2024
        ├── elo_history.csv       # ELO calculé après chaque match historique
        ├── wc_elo_updates.csv    # Delta ELO des matchs CdM joués (généré en prod)
        ├── wc2026_fixtures.csv   # Calendrier officiel CdM 2026 (104 matchs)
        ├── wc2026_teams.csv      # Mapping noms FIFA → noms dataset
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
GUILD_ID=...
FOOTBALL_DATA_KEY=...   # football-data.org — utilisé par le resolver
```

Entraîner le modèle (une fois) :
```bash
python -m ml.pipeline
```

Lancer le bot :
```bash
python bot.py
```

---

## Limites connues

| Limite | Détail |
|---|---|
| **Pas de score exact** | Le modèle prédit H/D/A, pas un score précis |
| **Mapping noms football-data.org** | Si un nom d'équipe ne correspond pas, le resolver log un avertissement — corriger `_FDORG_TO_FIXTURE` dans `wc_resolver.py` |
| **Phases éliminatoires** | `/prono` ne couvre que le groupe stage. Pour les KO : mettre à jour `wc2026_fixtures.csv` avec les vraies équipes une fois qualifiées |
| **`wc_elo_updates.csv` non persisté** | Railway recrée le filesystem à chaque redémarrage. L'ELO repart de 0 après un redémarrage du pod (solution : le committer après chaque journée) |

---

## Licence

MIT
