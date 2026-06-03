# PROJECT_STATE.md — État complet du projet
> Mis à jour le 04/06/2026 — pipeline régénéré, modèle propre, push effectué.

---

## Comment Railway et Discord fonctionnent ensemble

### Railway → bot Discord
Railway héberge le bot comme un **worker persistant** (Procfile : `worker: python bot.py`).

Quand tu pushs sur GitHub :
1. Railway détecte le push (intégration GitHub automatique)
2. Pull le code, installe `requirements.txt`, crée un nouveau container
3. Lance `python bot.py`
4. Le bot se connecte à Discord via `DISCORD_TOKEN` (variable d'env Railway)
5. `on_ready` : synchronise les slash commands avec l'API Discord
6. Reste en vie 24/7 jusqu'au prochain déploiement

Les variables d'environnement (`DISCORD_TOKEN`, `GUILD_ID`, `FOOTBALL_DATA_KEY`) sont définies dans le dashboard Railway — jamais dans le code.

**Temps de propagation des commandes Discord** :
- Synced sur un guild (`GUILD_ID` défini) → **instantané**
- Synced globalement (sans `GUILD_ID`) → jusqu'à **1 heure**

### Les prédictions : pré-calculées ou à la volée ?

`wc2026_predictions.csv` **n'est PAS utilisé par le bot Discord**. C'est un artifact d'analyse créé par `ml/run_wc2026.py` pour visualiser les prédictions en batch depuis le terminal.

Quand un utilisateur fait `/prono groupe:C` :
```
1. prono.py lit wc2026_fixtures.csv → liste des 3 matchs du groupe (lru_cache)
2. Utilisateur sélectionne "Brazil vs Morocco"
3. predict_match("Brazil", "Morocco", "2026-06-13") est appelé
4. predict.py charge model.pkl + results.csv + elo_history.csv (lru_cache — 1 fois par process)
5. Calcule les features à la volée : ELO, forme 5 matchs, H2H
6. model.predict_proba() → [P(away), P(draw), P(home)]
7. Retourne le dict → format_result() → message Discord avec barres
```

Les features sont donc **recalculées en temps réel** à chaque `/prono`. La première prédiction prend ~1-2s (chargement des fichiers). Les suivantes sont quasi-instantanées (tout est en cache).

`wc2026_predictions.csv` sert à :
- Consulter les prédictions en dehors du bot (référence rapide)
- Portfolio / démonstration
- Détecter des incohérences avant la mise en prod

---

## 1. Commandes Discord

### `/prono groupe:<A-L>`
Affiche un sélecteur dropdown avec les 3 matchs du groupe. L'utilisateur choisit un match → prédiction ML calculée à la volée → message avec barres de probabilité, ELO, indicateur de confiance (Favori clair / Légère faveur / Match serré).

**Effets de bord** : sauvegarde la prédiction en DB (`predictions`) pour `/accuracy`.

**Limitation** : groupe stage uniquement (72 matchs). Phases KO : équipes TBD.

---

### `/standings groupe:<A-L>`
Classement en temps réel. Source : table `match_results` alimentée par le resolver toutes les heures.

- Matchs joués → tableau classement + scores réels
- Matchs à venir → probabilités ML calculées à la volée

---

### `/accuracy`
Précision globale des prédictions résolues (H/D/A correct ou non). Ne montrera des données que quand des matchs auront été résolus ET que l'utilisateur aura fait `/prono` sur ces matchs.

---

## 2. Architecture — fichier par fichier

### Entrée

| Fichier | Rôle |
|---|---|
| `bot.py` | Point d'entrée. 3 commandes, tâche `auto_resolve` horaire, sync commands Discord. |
| `database.py` | Couche SQLite. 2 tables (`predictions`, `match_results`), 6 fonctions. |
| `requirements.txt` | ⚠️ Contient `httpx`, `flask`, `stripe` — plus utilisés. À nettoyer. |

### Commandes

| Fichier | Rôle |
|---|---|
| `commands/prono.py` | `/prono` — fixtures (lru_cache), Select UI, `predict_match()` en thread async. |
| `commands/standings.py` | `/standings` — scores réels depuis DB, classement calculé, `predict_match()` pour matchs à venir. |
| `commands/accuracy.py` | `/accuracy` — `get_stats()` → message Discord. |

### Services

| Fichier | Rôle |
|---|---|
| `services/ml_model.py` | `format_result(dict) → str`. Barres Unicode, bold prédiction, `_confidence_label()`. Fonction pure. |
| `services/wc_resolver.py` | Résolution horaire. football-data.org → mappe noms (10 entrées) → résout DB → enregistre scores → déclenche ELO update. |
| `services/elo_updater.py` | Post-match ELO. Lit `elo_history.csv` + `wc_elo_updates.csv`, calcule nouveaux ELO (K=40), appende, invalide cache `predict.py`. |

### ML

| Fichier | Rôle |
|---|---|
| `ml/features.py` | Feature engineering. `dropna` des matchs sans score, ELO pré-match (`shift(1)`), forme rolling (`shift(1) + rolling(5)`), H2H pre-indexé. 16 features. |
| `ml/train.py` | XGBoost + `compute_sample_weight('balanced')`. Calibration et seuil nul analysés mais non retenus. Sauvegarde XGBoost brut. |
| `ml/predict.py` | Inférence. `lru_cache` sur données + modèle. Fusionne `wc_elo_updates.csv` si présent. |
| `ml/pipeline.py` | Orchestrateur `features → train → run_wc2026`. `--force` pour tout recalculer. |
| `ml/run_wc2026.py` | Batch 72 matchs → `wc2026_predictions.csv`. Analyse uniquement, non utilisé par le bot. |

### Base de données (SQLite — `football.db`)

**Table `predictions`** : prédictions utilisateur via `/prono`
```
match_id, competition, home/away_team, predicted_result,
actual_result, is_correct_result, created_at, resolved_at
```

**Table `match_results`** : scores réels (source : wc_resolver)
```
match_id, home/away_team, home/away_score, match_group, match_date
```

### Données ML (`ml/data/`)

| Fichier | Taille | Rôle | Commitable |
|---|---|---|---|
| `results.csv` | 3.7 MB | 49 215 matchs avec score + 72 WC 2026 (NaN, ignorés par `dropna`) | ✅ |
| `elo_history.csv` | 3.9 MB | ELO propre après chaque match historique (98 430 lignes) | ✅ |
| `wc2026_fixtures.csv` | 7.4 KB | Calendrier CdM 2026 (104 matchs). KO : TBD. | ✅ |
| `wc2026_teams.csv` | 1.2 KB | Mapping FIFA → dataset (8 divergences) | ✅ |
| `wc2026_predictions.csv` | 5.7 KB | Prédictions batch pré-générées. Référence uniquement, non utilisé en prod. | ✅ |
| `model_config.json` | 1 KB | draw_threshold, métriques finales, features | ✅ |
| `features.csv` | 8.7 MB | Feature matrix entraînement. Régénérable. | ❌ .gitignore |
| `wc_elo_updates.csv` | runtime | Delta ELO matchs CdM joués. Généré en prod. | ❌ runtime |

---

## 3. État du pipeline ML

### Modèle final (propre depuis le 04/06/2026)
- **Algorithme** : XGBoost `multi:softprob` + `compute_sample_weight('balanced')`
- **draw_threshold** : null (argmax brut)
- **Calibration** : analysée, non retenue
- **Données** : 49 215 matchs avec score (NaN exclus par `dropna`)

### Features (16)
```
ELO         : elo_home, elo_away, elo_diff
Forme dom.  : home_form_pts, home_form_gf, home_form_ga, home_form_n
Forme ext.  : away_form_pts, away_form_gf, away_form_ga, away_form_n
H2H         : h2h_home_pts, h2h_gd, h2h_n
Contexte    : is_neutral, tournament_tier
```

### Métriques finales (ELO propre, test 2022-2024)

| Métrique | Validation | Test |
|---|---|---|
| Accuracy | 55.48% | 55.35% |
| Log-loss | 0.8946 | 0.9218 |
| Recall nuls | 37.35% | 33.29% |

### Split temporel
```
Train : ≤ 2017-12-31
Val   : 2018-01-01 → 2021-12-31  (early stopping)
Test  : 2022-01-01 → 2024-12-31  (verdict final — touché une seule fois)
```

### ELO en temps réel (prod)
- `elo_history.csv` : source immuable (1872-2024)
- `wc_elo_updates.csv` : delta live, 2 lignes par match joué (K=40)
- Cache `predict.py` invalidé automatiquement après chaque update
- ⚠️ Non persisté entre redémarrages Railway (voir priorité 1 ci-dessous)

---

## 4. Ce qui fonctionne en prod vs en cours

### ✅ Fonctionnel
- Bot Discord actif sur Railway
- Modèle propre déployé (ELO corrigé, XGBoost brut, class weights)
- `/prono` : prédictions à la volée, toutes les 12 équipes A-L
- `/standings` : prêt (vide jusqu'au 1er match le 11/06)
- `/accuracy` : prêt (vide jusqu'aux premières résolutions)
- `auto_resolve` : tâche horaire active
- ELO mis à jour automatiquement après chaque match joué

### ⚠️ Risques en prod
- **`wc_elo_updates.csv` non persisté** : perdu si Railway redémarre. Les prédictions reprennent depuis l'ELO pré-CdM.
- **Mapping `_FDORG_TO_FIXTURE`** : si football-data.org utilise un nom d'équipe non mappé, le resolver log un warning et skip le match.
- **`/accuracy` non représentatif** : ne compte que les matchs où un utilisateur a fait `/prono`.

### ❌ Non implémenté
- Monétisation (Stripe, rôles premium)
- Commandes admin (résolution manuelle, mise à jour bracket KO)
- Résolution automatique pour matchs sans prédiction DB
- Phases éliminatoires (équipes TBD)

---

## 5. Ce qui manque pour un produit complet

### Monétisation
- Système de tiers : gratuit (N /prono par jour), premium (illimité)
- Stripe checkout + Flask webhook → rôle Discord "Premium" automatique
- Commande `/premium` avec lien de paiement

### Déploiement
- Volume Railway persistant pour `wc_elo_updates.csv`
- Nettoyage `requirements.txt` : retirer `httpx`, `flask`, `stripe`
- Monitoring : alerte si `auto_resolve` échoue 3 cycles consécutifs

### Features ML
- Phases éliminatoires : commande admin `/admin fixture <num> <home> <away>`
- `/accuracy` complet : pré-remplir `predictions` au démarrage pour tous les matchs du groupe stage avec la prédiction du modèle (pas seulement quand l'utilisateur fait `/prono`)
- Retrain post-groupe-stage : réentraîner sur les résultats CdM 2026 avant les KO

### UX Discord
- Notification proactive (webhook) : "dans 2h : France vs Senegal — voici la prédiction"
- `/bracket` : arbre des phases éliminatoires

---

## 6. Les 3 prochaines étapes prioritaires

### Étape 1 — Persistance ELO entre redémarrages Railway (AVANT LE 11/06)
Après chaque journée de groupe, committer `wc_elo_updates.csv` :
```bash
git add ml/data/wc_elo_updates.csv
git commit -m "chore: wc_elo_updates journee X"
git push
```
Railway rechargera le fichier depuis le repo au prochain démarrage.

### Étape 2 — `/accuracy` représentatif dès le premier match
Pré-remplir la table `predictions` au démarrage du bot pour tous les matchs du groupe stage avec la prédiction du modèle. Actuellement, `/accuracy` est vide si personne n'a fait `/prono`. Avec le pré-remplissage, la précision du modèle sera tracée automatiquement.

### Étape 3 — Nettoyage `requirements.txt`
Retirer `httpx` (0.27.2), `flask` (3.1.1), `stripe` (12.2.0) — plus aucun fichier ne les importe. Alléger le container Railway.
