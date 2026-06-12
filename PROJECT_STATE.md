# PROJECT_STATE.md — État complet du projet
> Mis à jour le 12/06/2026. Architecture deux modèles + Poisson déployée. Bot actif depuis le 11/06.

---

## Comment Railway et Discord fonctionnent ensemble

### Railway → bot Discord
Railway héberge le bot comme un **worker persistant** (Procfile : `worker: python bot.py`).

Quand tu pushs sur GitHub :
1. Railway détecte le push (intégration GitHub automatique)
2. Pull le code, installe `requirements.txt`, crée un nouveau container
3. Lance `python bot.py`
4. Le bot se connecte à Discord via `DISCORD_TOKEN` (variable d'env Railway)
5. `on_ready` : synchronise les slash commands, déclenche `auto_resolve`, pré-remplit les prédictions DB
6. Reste en vie 24/7 jusqu'au prochain déploiement

Les variables d'environnement sont définies dans le dashboard Railway — jamais dans le code.

| Variable | Fichier | Rôle |
|---|---|---|
| `DISCORD_TOKEN` | `bot.py` | Token du bot Discord |
| `GUILD_ID` | `bot.py` | ID du serveur Discord (optionnel — sync instantané en dev) |
| `FOOTBALL_DATA_KEY` | `services/wc_resolver.py` | Clé API football-data.org |
| `PERSISTENT_DIR` | `database.py`, `elo_updater.py`, `predict.py` | Chemin du volume Railway (`/data`) |

**Volume Railway** : monter un volume sur `/data` + ajouter `PERSISTENT_DIR=/data` en variable d'env.
`football.db` et `wc_elo_updates.csv` y sont stockés et survivent aux redéploiements.

**Temps de propagation des slash commands Discord** :
- Synced sur un guild (`GUILD_ID` défini) → **instantané**
- Synced globalement (sans `GUILD_ID`) → jusqu'à **1 heure**

### Les prédictions : pré-calculées ou à la volée ?

`wc2026_predictions.csv` **n'est PAS utilisé par le bot**. C'est un artifact d'analyse batch créé par `ml/run_wc2026.py`.

Quand un utilisateur fait `/prono groupe:C` :
```
1. prono.py lit wc2026_fixtures.csv → liste des 3 matchs du groupe (lru_cache)
2. Utilisateur sélectionne "Brazil vs Morocco"
3. predict_match("Brazil", "Morocco", "2026-06-13") est appelé
4. predict.py charge model_wc.pkl + results.csv + elo_history.csv (lru_cache — 1 fois par process)
5. Calcule les features à la volée : ELO, forme 5 matchs, H2H, FIFA rank, market value, WC titles
6. model.predict_proba() → [P(away), P(draw), P(home)]
7. Poisson → score_matrix → most likely score conditionnel à l'issue XGBoost
8. Retourne le dict → format_result() → message Discord avec barres Unicode
```

Première prédiction : ~1-2s (chargement fichiers + Poisson params). Suivantes : quasi-instantanées (cache).

### Mise à jour des scores (manuelle)

Il n'y a **pas de résolution automatique**. Les scores sont saisis manuellement via `/score` (admin). `wc_resolver.py` a été supprimé — il n'y a aucune tâche périodique dans le bot.

---

## 1. Commandes Discord (4 commandes actives)

### `/prono groupe:<A-L>`
Dropdown avec les 3 matchs du groupe → prédiction ML à la volée → barres de probabilité, ELO des deux équipes, score prédit, label de confiance (Favori clair / Légère faveur / Match serré).

**Effet de bord** : sauvegarde la prédiction en DB (`predictions`) pour `/accuracy`. `ON CONFLICT DO NOTHING` si déjà pré-remplie.

**Limitation** : groupe stage uniquement (72 matchs). Phases KO : équipes TBD.

---

### `/standings groupe:<A-L>`
Classement en temps réel. Source : table `match_results` alimentée par `auto_resolve` toutes les heures.

- Matchs joués → tableau classement + scores réels
- Matchs à venir → probabilités ML calculées à la volée

---

### `/accuracy`
Précision globale du modèle sur les prédictions résolues (résultat H/D/A correct ou non). Représentatif dès le premier match grâce au pré-remplissage automatique au démarrage (`_prefill_predictions()`).

---

### `/simulate groupe:<A-L>`
Simulation Monte Carlo (10 000 itérations) des probabilités de qualification dans un groupe. Affiche la probabilité top-2 pour chaque équipe, triée, avec barres Unicode.

---

## 2. Architecture — fichier par fichier

### Entrée

| Fichier | Rôle |
|---|---|
| `bot.py` | Point d'entrée. 4 commandes enregistrées, `_prefill_predictions()` au `on_ready`, tâche `auto_resolve` horaire. |
| `database.py` | Couche SQLite. Tables `predictions` et `match_results`. Init automatique. Chemin DB via `PERSISTENT_DIR`. |
| `requirements.txt` | Dépendances propres : discord.py, requests, python-dotenv, pandas, scikit-learn, xgboost, scipy. |
| `Procfile` | `worker: python bot.py` — Railway worker persistant. |
| `.python-version` | `3.12` — pin Python pour Railway. |

### Commandes

| Fichier | Rôle |
|---|---|
| `commands/prono.py` | `/prono` — fixtures via `lru_cache`, Select UI Discord, `predict_match()` dans un thread async. |
| `commands/standings.py` | `/standings` — scores réels depuis DB, classement calculé, `predict_match()` pour matchs à venir. |
| `commands/accuracy.py` | `/accuracy` — `database.get_stats()` → message Discord. |
| `commands/simulate.py` | `/simulate` — appel `ml.simulator.simulate_group()` dans un thread async, format barres Unicode. |

### Services

| Fichier | Rôle |
|---|---|
| `services/ml_model.py` | `format_result(dict) → str`. Barres Unicode, bold prédiction, score prédit, `_confidence_label()`. Fonction pure. |
| `services/wc_resolver.py` | Résolution horaire. football-data.org → mappe noms (`_FDORG_TO_FIXTURE`) → résout DB → ELO. |
| `services/elo_updater.py` | Calcule les nouveaux ELO post-match (K=40), appende à `wc_elo_updates.csv` dans `PERSISTENT_DIR`. Invalide le cache de `predict.py`. |

### ML

| Fichier | Rôle |
|---|---|
| `ml/features.py` | Feature engineering vectorisé : ELO pré-match (`shift(1)`), forme rolling 5 matchs, H2H, wc_form, rest_days, host. **20 features.** |
| `ml/wc_features.py` | Features WC supplémentaires : FIFA rank, market value, WC titles, WC participations. Jointure avec `wc_teams_train/test.csv`. **4 features.** |
| `ml/train.py` | Entraînement XGBoost `multi:softprob`. `train()` → `model.pkl` (20 features), `train_wc()` → `model_wc.pkl` (24 features). Split temporel strict. |
| `ml/poisson.py` | Dixon-Coles Poisson — `fit()` sur 2477 matchs compétitifs depuis 2018. `predict_score()` → score_matrix. Paramètres sérialisés dans `poisson_params.json`. |
| `ml/predict.py` | Inférence single-match. `lru_cache` sur données + modèles. WC model utilisé si `tournament_tier=4`. Score via Poisson conditionné à l'issue XGBoost. |
| `ml/simulator.py` | Monte Carlo pour groupe (probabilité top-2) et tournoi complet. KO : tirs au but simulés 50/50 pour les nuls. |
| `ml/pipeline.py` | Orchestrateur : `features → train → run_wc2026`. `--force` pour tout recalculer. |
| `ml/run_wc2026.py` | Batch 72 matchs groupe stage → `wc2026_predictions.csv`. Analyse locale, non utilisé par le bot. |

### Artefacts ML (répertoire `ml/`)

| Fichier | Rôle | Commitable |
|---|---|---|
| `model.pkl` | XGBoost général — 20 features, train < 2018. Requis en prod pour matchs hors-WC. | ✅ |
| `model_wc.pkl` | XGBoost WC — 24 features, train WC 2002-2014. Utilisé automatiquement pour `tournament_tier=4`. | ✅ |
| `model_config.json` | Config modèle général : features, métriques finales, split dates. | ✅ |
| `metrics.json` | Métriques détaillées des deux modèles (accuracy, log-loss, draw_recall, baseline). | ✅ |

### Base de données (SQLite — `football.db`)

Stockée dans `PERSISTENT_DIR` (`/data` en prod) → survit aux redéploiements.

**Table `predictions`** : prédictions utilisateur via `/prono` + pré-remplissage auto au démarrage
```
id, match_id (UNIQUE), competition, home_team, away_team,
predicted_home_goals, predicted_away_goals, predicted_result,
actual_home_goals, actual_away_goals, actual_result,
is_correct_result, is_correct_score, created_at, resolved_at
```

**Table `match_results`** : scores réels (alimentée par `wc_resolver`)
```
match_id (PK), home_team, away_team, home_score, away_score,
match_group, match_date
```

### Données ML (`ml/data/`)

| Fichier | Taille | Rôle | Commitable |
|---|---|---|---|
| `results.csv` | 3.7 MB | 49 215 matchs avec score (Jürisoo 1872-2024) + 72 WC 2026 (NaN, exclus) | ✅ |
| `elo_history.csv` | 3.9 MB | ELO propre après chaque match historique. **Source immuable — ne jamais modifier en prod.** | ✅ |
| `wc2026_fixtures.csv` | 7.4 KB | Calendrier CdM 2026 (104 matchs — 72 groupe + 32 KO TBD). | ✅ |
| `wc2026_teams.csv` | 1.2 KB | Mapping FIFA → dataset (8 divergences de noms). | ✅ |
| `wc_teams_train.csv` | ~15 KB | Features WC par équipe 2002-2022 (192 rows) — utilisé en entraînement. | ✅ |
| `wc_teams_test.csv` | ~3 KB | Features WC 2026 pour les 48 équipes — **uniquement inférence, jamais en train.** | ✅ |
| `poisson_params.json` | ~40 KB | Paramètres Dixon-Coles fittés : attack/defense 180 équipes, home_adv=1.318, rho=-0.092. | ✅ |
| `wc2026_predictions.csv` | 5.7 KB | Prédictions batch des 72 matchs. Référence et portfolio, non utilisé en prod. | ✅ |
| `statsbomb_xg_wc.csv` | — | 128 matchs WC 2018+2022 avec xG StatsBomb. Exploration uniquement. | ✅ |
| `features.csv` | ~8.7 MB | Feature matrix d'entraînement. Régénérable via `pipeline.py`. | ❌ .gitignore |
| `wc_elo_updates.csv` | runtime | Delta ELO live (2 lignes/match). Stocké dans `PERSISTENT_DIR` en prod. | ❌ runtime |

---

## 3. État du pipeline ML

### Architecture de prédiction

| Contexte | Issue (H/D/A) | Score |
|---|---|---|
| **Matchs WC** (`tournament_tier=4`) | Poisson Dixon-Coles | Conditionné à l'issue Poisson |
| **Matchs ordinaires** | XGBoost général (`model.pkl`) | Conditionné à l'issue XGBoost |

**Pourquoi Poisson pour la CdM :** backtest sur WC 2022 — Poisson 54.7% vs XGBoost 46.9% vs baseline ELO 56.2%. XGBoost manque de données WC (256 matchs train) et est mal calibré sur des équipes toutes de haut niveau. Poisson capte la force attack/defense individuelle sur 2 477 matchs récents.

### Modèle XGBoost général (`model.pkl`)

### Features générales (20)
```
ELO         : elo_home, elo_away, elo_diff
Forme dom.  : home_form_pts, home_form_gf, home_form_ga
Forme ext.  : away_form_pts, away_form_gf, away_form_ga
H2H         : h2h_home_pts, h2h_gd, h2h_n
Contexte    : is_neutral, tournament_tier, home_is_host, away_is_host
Tournoi     : home_wc_form_pts, away_wc_form_pts
Physique    : home_rest_days, away_rest_days
```

### Features WC supplémentaires (4)
```
rank_diff               : away_rank - home_rank (positif = home mieux classé FIFA)
log_market_ratio        : log(home_value / away_value)
wc_titles_diff          : home_titles - away_titles
wc_participations_diff  : home_participations - away_participations
```

### Métriques finales — modèle général

| Métrique | Baseline ELO (val) | XGBoost (val) | Baseline ELO (test) | XGBoost (test) |
|---|---|---|---|---|
| Accuracy | 57.04% | **55.82%** | 55.32% | **55.20%** |
| Log-loss | — | 0.8938 | — | 0.9217 |
| Draw recall | 28.47% | 37.23% | 23.30% | 33.16% |

> XGBoost est légèrement sous la baseline ELO en accuracy brute mais capte mieux les nuls (draw recall +10 pp) — le vrai apport du modèle.

### Backtest WC 2022 (64 matchs)

| Modèle | Accuracy |
|---|---|
| Baseline ELO naïve | 56.2% |
| **Poisson Dixon-Coles** | **54.7%** ← utilisé en prod |
| XGBoost WC (256 train) | 48.4% |
| XGBoost général | 46.9% |

Poisson est le meilleur de nos modèles sur WC. Il sous-prédit les nuls (2/15) mais donne des probabilités de draw non-nulles et réalistes (25-35%).

### Score prediction — Dixon-Coles Poisson

`ml/poisson.py` — MLE sur 2477 matchs compétitifs depuis 2018 (filtrés : au moins une équipe WC 2026).
- `lambda_home = attack_home × defense_away × home_adv`
- `home_adv = 1.318`, `rho = -0.092` (correction Dixon-Coles pour bas scores)
- Score prédit = argmax dans le triangle/diagonale de la score_matrix, conditionné à l'issue prédite (Poisson pour WC, XGBoost pour le reste)
- Équipes inconnues (ex. Curaçao) → fallback = médiane des équipes WC dans le modèle

### Split temporel (règle stricte — jamais de shuffle)
```
Général : Train ≤ 2017-12-31 | Val 2018-2021 | Test 2022-2024
WC      : Train WC 2002-2014 | Val WC 2018   | Test WC 2022
```

### ELO en temps réel (prod)
- `elo_history.csv` : source immuable (1872-2024)
- `wc_elo_updates.csv` : delta live — 2 lignes par match joué, K=40, stocké dans `PERSISTENT_DIR`
- `wc_resolver.py` vérifie `already_updated` avant chaque update → pas de doublons
- Cache `predict.py` (`_data.cache_clear()`) invalidé automatiquement après chaque update

---

## 4. Ce qui fonctionne en prod (vérifié le 12/06/2026)

### ✅ Opérationnel
- Bot Discord actif sur Railway
- Deux modèles XGBoost déployés (général + WC)
- Score prediction via Dixon-Coles Poisson (conditionnel à l'issue prédite)
- `/prono` : prédictions à la volée, tous les groupes A-L, avec score prédit
- `/standings` : opérationnel (scores réels depuis le 11/06)
- `/accuracy` : représentatif dès le 1er match — 72 prédictions pré-remplies en DB
- `/simulate` : Monte Carlo 10 000 simulations
- `/score` (admin) : saisie manuelle des scores → DB + ELO mis à jour
- `PERSISTENT_DIR` : `football.db` et `wc_elo_updates.csv` persistants sur volume Railway

### ⚠️ Risques résiduels
- **Mapping `_FDORG_TO_FIXTURE`** (`wc_resolver.py:26-37`) : si football-data.org utilise un nom non mappé, le match est skippé avec un log `Mapping manquant`. À surveiller après chaque journée.
- **`auto_resolve` silencieux** : si l'API football-data.org est down, pas d'alerte en place.

### ❌ Non implémenté
- Phases éliminatoires (équipes TBD dans `wc2026_fixtures.csv`)
- Commandes admin (résolution manuelle, mise à jour bracket KO)
- Monitoring / alerte si `auto_resolve` échoue

---

## 5. Flux après chaque match (saisie manuelle)

Après chaque match, utiliser la commande Discord `/score` (réservée admin) :

```
/score match_number:<N> home_score:<X> away_score:<Y>
  → database.save_match_result()   → /standings à jour
  → database.resolve_prediction()  → /accuracy compte le match
  → already_updated check          → pas de doublon ELO
  → update_elo_with_match()        → PERSISTENT_DIR/wc_elo_updates.csv +2 lignes
      → _data.cache_clear()        → prochain /prono = ELO frais
```

Le numéro de match correspond à la colonne `match_number` dans `wc2026_fixtures.csv`.

**Pour voir le contenu de `wc_elo_updates.csv` en prod :**
Railway → Service → Shell → `cat /data/wc_elo_updates.csv`

---

## 6. Prochaines étapes prioritaires

### Étape 1 — Monitoring `auto_resolve`
Détecter les échecs silencieux. Solution : compteur d'échecs consécutifs dans `bot.py` → envoi d'un message dans un channel Discord admin si > 3 cycles sans résolution alors que des matchs sont attendus.

### Étape 2 — Phases éliminatoires
Commande admin `/admin fixture <match_number> <home_team> <away_team>` pour renseigner les équipes qualifiées dans `wc2026_fixtures.csv`. Débloque `/prono`, `/standings` et `/simulate` pour les matchs KO.

### Étape 3 — Retrain post-groupe-stage
Après les 48 matchs de groupe (résultats réels disponibles), réentraîner les modèles :
```bash
python -m ml.train          # régénère model.pkl + model_wc.pkl
python -m ml.run_wc2026     # régénère wc2026_predictions.csv
```
Intègre les données CdM 2026 dans `results.csv` → meilleur modèle pour les KO.
Refitter aussi les paramètres Poisson : `python -c "from ml.poisson import fit; fit(force=True)"`.

---

## 7. Ce qui manque pour un produit complet

### Monétisation
- Système de tiers : gratuit (N `/prono` par jour), premium (illimité)
- Stripe checkout + webhook → attribution automatique du rôle Discord "Premium"
- Commande `/premium` avec lien de paiement

### UX Discord
- Notification proactive : webhook "dans 2h : France vs Sénégal — prédiction du modèle"
- `/bracket` : arbre visuel des phases éliminatoires avec probabilités

### Tests automatisés
- Tests unitaires sur `predict_match`, `resolve_wc_predictions`, `update_elo_with_match`
