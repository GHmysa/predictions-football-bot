# PROJECT_STATE.md — État complet du projet
> Mis à jour le 10/06/2026 — veille de la CdM 2026. Toutes les étapes prioritaires complétées. Bot prêt pour le premier match (11/06).

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
4. predict.py charge model.pkl + results.csv + elo_history.csv (lru_cache — 1 fois par process)
5. Calcule les features à la volée : ELO, forme 5 matchs, H2H
6. model.predict_proba() → [P(away), P(draw), P(home)]
7. Retourne le dict → format_result() → message Discord avec barres Unicode
```

Première prédiction : ~1-2s (chargement fichiers). Suivantes : quasi-instantanées (cache).

---

## 1. Commandes Discord (4 commandes actives)

### `/prono groupe:<A-L>`
Dropdown avec les 3 matchs du groupe → prédiction ML à la volée → barres de probabilité, ELO des deux équipes, label de confiance (Favori clair / Légère faveur / Match serré).

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
| `database.py` | Couche SQLite. Tables `predictions` et `match_results`. Init automatique au chargement du module. |
| `requirements.txt` | Dépendances propres : discord.py, requests, python-dotenv, pandas, scikit-learn, xgboost, matplotlib, seaborn. |
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
| `services/ml_model.py` | `format_result(dict) → str`. Barres Unicode, bold prédiction, `_confidence_label()`. Fonction pure, pas d'état. |
| `services/wc_resolver.py` | Résolution horaire. football-data.org → mappe 10 noms d'équipes (`_FDORG_TO_FIXTURE`) → résout DB → enregistre scores → met à jour ELO. Vérifie `already_updated` avant chaque ELO pour éviter les doublons. |
| `services/elo_updater.py` | Calcule les nouveaux ELO post-match (K=40) et appende à `wc_elo_updates.csv`. Invalide le cache de `predict.py` après chaque update. |

### ML

| Fichier | Rôle |
|---|---|
| `ml/features.py` | Feature engineering vectorisé : ELO pré-match (`shift(1)`), forme rolling 5 matchs (`shift(1) + rolling(5)`), H2H pre-indexé. 16 features. |
| `ml/train.py` | Entraînement XGBoost `multi:softprob` + `compute_sample_weight('balanced')`. Split temporel strict. Sauvegarde `model.pkl`. |
| `ml/predict.py` | Inférence single-match. `lru_cache` sur données + modèle. Fusionne `wc_elo_updates.csv` si présent. Cache invalidé par `elo_updater.py`. |
| `ml/simulator.py` | Monte Carlo pour groupe (probabilité top-2) et tournoi complet (victoire finale). KO : tirs au but simulés 50/50 pour les nuls. Cache interne des probas KO. |
| `ml/pipeline.py` | Orchestrateur : `features → train → run_wc2026`. `--force` pour tout recalculer depuis zéro. |
| `ml/run_wc2026.py` | Batch 72 matchs groupe stage → `wc2026_predictions.csv`. Analyse locale uniquement, non utilisé par le bot. |

### Artefacts ML (répertoire `ml/`)

| Fichier | Rôle | Commitable |
|---|---|---|
| `model.pkl` | Modèle XGBoost sérialisé. Requis par le bot en prod. | ✅ |
| `model_config.json` | Config du modèle : draw_threshold, métriques finales, liste des features. | ✅ |
| `metrics.json` | Résultats d'évaluation détaillés (accuracy, log-loss par split). | ✅ |
| `elo_evolution.png` | Graphique de l'évolution ELO des équipes. Artifact d'analyse. | ✅ |

### Base de données (SQLite — `football.db`)

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

⚠️ `football.db` est dans `.gitignore` (via `*.db`). Non commité. En cas de redéploiement Railway, la DB est reconstruite automatiquement : `_prefill_predictions()` réinsère les 72 prédictions, puis `auto_resolve` re-résout tous les matchs terminés depuis l'API.

### Données ML (`ml/data/`)

| Fichier | Taille | Rôle | Commitable |
|---|---|---|---|
| `results.csv` | 3.7 MB | 49 215 matchs avec score (dataset Jürisoo, 1872-2024) + 72 WC 2026 (NaN, exclus par `dropna`) | ✅ |
| `elo_history.csv` | 3.9 MB | ELO propre après chaque match historique (98 430 lignes). **Source immuable — ne jamais modifier en prod.** | ✅ |
| `wc2026_fixtures.csv` | 7.4 KB | Calendrier CdM 2026 (104 matchs — 72 groupe stage + 32 KO avec équipes TBD). | ✅ |
| `wc2026_teams.csv` | 1.2 KB | Mapping FIFA → dataset (8 divergences de noms). | ✅ |
| `wc2026_predictions.csv` | 5.7 KB | Prédictions batch pré-générées pour les 72 matchs. Référence et portfolio, non utilisé en prod. | ✅ |
| `elo_wc2026.csv` | 2.1 KB | Snapshot ELO pré-CdM 2026 des 32 équipes (au 31/03/2026) avec confederation et fifa_name. Référence uniquement. | ✅ |
| `elo_latest.csv` | 13.7 KB | ELO le plus récent de toutes les équipes connues. Référence uniquement. | ✅ |
| `goalscorers.csv` | 3.3 MB | Dataset Jürisoo — buteurs historiques. Non utilisé en prod. | ✅ |
| `shootouts.csv` | 29 KB | Dataset Jürisoo — tirs au but historiques. Non utilisé en prod. | ✅ |
| `features.csv` | ~8.7 MB | Feature matrix d'entraînement. Régénérable via `ml/pipeline.py`. | ❌ .gitignore |
| `wc_elo_updates.csv` | runtime | Delta ELO des matchs CdM joués (2 lignes/match). Généré en prod. Reconstruit auto après redémarrage. | ❌ runtime |

---

## 3. État du pipeline ML

### Modèle final (propre depuis le 04/06/2026)
- **Algorithme** : XGBoost `multi:softprob`
- **Class weights** : `compute_sample_weight('balanced')` — corrige le déséquilibre nuls/victoires
- **draw_threshold** : null (argmax brut — calibration et seuil analysés, non retenus)
- **Données d'entraînement** : 49 215 matchs avec score (NaN exclus par `dropna`)

### Features (16)
```
ELO         : elo_home, elo_away, elo_diff
Forme dom.  : home_form_pts, home_form_gf, home_form_ga, home_form_n
Forme ext.  : away_form_pts, away_form_gf, away_form_ga, away_form_n
H2H         : h2h_home_pts, h2h_gd, h2h_n
Contexte    : is_neutral, tournament_tier
```

### Métriques finales (test 2022-2024 — touché une seule fois)

| Métrique | Validation (2018-2021) | Test (2022-2024) |
|---|---|---|
| Accuracy | 55.48% | 55.35% |
| Log-loss | 0.8946 | 0.9218 |
| Recall nuls | 37.35% | 33.29% |

### Split temporel (règle stricte — jamais de shuffle sur séries temporelles)
```
Train : ≤ 2017-12-31
Val   : 2018-01-01 → 2021-12-31  (early stopping XGBoost)
Test  : 2022-01-01 → 2024-12-31  (verdict final)
```

### ELO en temps réel (prod)
- `elo_history.csv` : source immuable (1872-2024)
- `wc_elo_updates.csv` : delta live — 2 lignes par match joué, K=40
- `wc_resolver.py` vérifie `already_updated` (team + date) avant chaque update → pas de doublons
- Cache `predict.py` (`_data.cache_clear()`) invalidé automatiquement après chaque update
- En cas de redémarrage : `auto_resolve` se déclenche immédiatement et reconstruit le fichier depuis l'API

---

## 4. Ce qui fonctionne en prod (vérifié sur le code le 10/06/2026)

### ✅ Opérationnel
- Bot Discord actif sur Railway
- Modèle propre déployé (ELO corrigé, XGBoost brut, class weights)
- `/prono` : prédictions à la volée, tous les groupes A-L
- `/standings` : opérationnel (affichera les scores dès le 11/06)
- `/accuracy` : représentatif dès le 1er match — 72 prédictions pré-remplies en DB au démarrage
- `/simulate` : Monte Carlo 10 000 simulations, probabilités de qualification par groupe
- `auto_resolve` : tâche horaire — résolution prédictions + enregistrement scores + mise à jour ELO
- Persistance ELO : reconstruction automatique après redémarrage (pas de commit manuel nécessaire)
- `requirements.txt` : propre (httpx, flask, stripe supprimés)

### ⚠️ Risques résiduels
- **Mapping `_FDORG_TO_FIXTURE`** (`wc_resolver.py:26-37`) : si football-data.org utilise un nom d'équipe non mappé, le match est skippé avec un log `Mapping manquant`. À surveiller après chaque journée de groupe.
- **`football.db` perdu sur redéploiement** : reconstruction automatique, mais `created_at` des prédictions repart de zéro. Sans impact fonctionnel.
- **`auto_resolve` silencieux en cas d'échec** : si l'API football-data.org est down plusieurs heures, les scores ne sont pas mis à jour. Pas d'alerte en place actuellement.

### ❌ Non implémenté
- Phases éliminatoires (équipes TBD dans `wc2026_fixtures.csv`)
- Commandes admin (résolution manuelle, mise à jour bracket KO)
- Monitoring / alerte si `auto_resolve` échoue
- Monétisation (Stripe, rôles premium Discord)

---

## 5. Flux automatisé après chaque match (depuis le 11/06)

**Rien à faire manuellement.** Voici le flux complet :

```
Match terminé (football-data.org : status = FINISHED)
  ↓  dans l'heure suivante
auto_resolve (bot.py — @tasks.loop hours=1)
  → asyncio.to_thread(resolve_wc_predictions)
      → GET /competitions/WC/matches (football-data.org)
      → tri chronologique des matchs terminés
      → pour chaque match :
          → _resolve_name() : mappe nom fdorg → nom fixture
          → database.save_match_result()       → /standings à jour
          → database.resolve_prediction()      → /accuracy compte le match
          → already_updated check              → pas de doublon ELO
          → update_elo_with_match()            → wc_elo_updates.csv +2 lignes
              → _data.cache_clear()            → prochain /prono = ELO frais
```

**Logs Railway à surveiller après chaque journée :**
```
[AUTO-RESOLVE] Lancement du cycle de résolution…
[WC RESOLVER] 5 en attente | 3 matchs terminés dans l'API
[WC RESOLVER] ✅ Résolu : France 2-1 Belgique (H)
[ELO UPDATE] France (France) : 1989.4 → 1997.2 | Belgique (Belgium) : 1876.3 → 1868.5
[WC RESOLVER] 3 prédiction(s) résolue(s) ce cycle.
```

**Si tu vois `Mapping manquant` dans les logs :**
Ajouter l'entrée dans `_FDORG_TO_FIXTURE` dans `services/wc_resolver.py` (lignes 26-37) et redéployer.

---

## 6. Prochaines étapes prioritaires (post-11/06)

### Étape 1 — Monitoring `auto_resolve`
Détecter les échecs silencieux. Solution : compteur d'échecs consécutifs dans `bot.py` → envoi d'un message dans un channel Discord admin si > 3 cycles sans résolution alors que des matchs sont attendus.

### Étape 2 — Phases éliminatoires
Commande admin `/admin fixture <match_number> <home_team> <away_team>` pour renseigner les équipes qualifiées dans `wc2026_fixtures.csv`. Débloque `/prono`, `/standings` et `/simulate` pour les matchs KO.

### Étape 3 — Retrain post-groupe-stage
Après les 48 matchs de groupe (résultats réels disponibles), réentraîner le modèle :
```bash
python -m ml.pipeline --force
```
Intègre les données CdM 2026 dans `results.csv` + recalcule ELO → meilleur modèle pour les KO.

---

## 7. Ce qui manque pour un produit complet

### Monétisation
- Système de tiers : gratuit (N `/prono` par jour), premium (illimité)
- Stripe checkout + webhook → attribution automatique du rôle Discord "Premium"
- Commande `/premium` avec lien de paiement

### UX Discord
- Notification proactive : webhook "dans 2h : France vs Sénégal — prédiction du modèle"
- `/bracket` : arbre visuel des phases éliminatoires avec probabilités

### Infrastructure
- Volume Railway persistant pour `football.db` (évite la reconstruction au redéploiement)
- Tests automatisés des fonctions critiques (`predict_match`, `resolve_wc_predictions`)
