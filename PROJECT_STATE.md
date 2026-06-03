# PROJECT_STATE.md — État complet du projet
> Généré le 03/06/2026. À mettre à jour après chaque sprint majeur.

---

## 1. Commandes Discord

### `/prono groupe:<A-L>`
**Ce que ça fait** : affiche un sélecteur dropdown avec les 3 matchs du groupe choisi. L'utilisateur sélectionne un match → le modèle ML calcule les probabilités → le bot répond avec un message structuré contenant les barres de probabilité, les ELO des deux équipes, l'indicateur de confiance (Favori clair / Légère faveur / Match serré), et la prédiction finale.

**Effets de bord** :
- Sauvegarde la prédiction dans la table `predictions` (SQLite) pour tracking de précision via `/accuracy`
- Les prédictions sont encodées comme score symbolique (1-0 = domicile, 0-0 = nul, 0-1 = extérieur)

**Limitation** : ne couvre que les 72 matchs du groupe stage. Les phases éliminatoires ont des équipes "TBD" dans `wc2026_fixtures.csv`.

---

### `/standings groupe:<A-L>`
**Ce que ça fait** : construit et affiche le classement en temps réel d'un groupe.

**Si des matchs ont été joués** : tableau de classement (Pts / J / V / N / D / GF / GA / +/-) avec les ✓ sur les 2 équipes actuellement qualifiées, liste des scores réels, et prédictions ML pour les matchs restants du groupe.

**Si aucun match joué** : liste des 6 matchs du groupe avec les probabilités ML pour chacun.

**Source des données** : table `match_results` (SQLite), alimentée par `wc_resolver.py` toutes les heures.

---

### `/accuracy`
**Ce que ça fait** : affiche la précision des prédictions résolues. Pourcentage de résultats corrects (H/D/A), total de matchs analysés, breakdown par compétition.

**Limitation** : ne montrera des données que lorsque des matchs auront été résolus. La résolution est automatique via `auto_resolve` (toutes les heures), mais dépend de football-data.org pour les scores réels ET du fait que l'utilisateur ait utilisé `/prono` sur ce match.

---

## 2. Architecture — fichier par fichier

### Entrée

| Fichier | Rôle |
|---|---|
| `bot.py` | Point d'entrée. Enregistre 3 commandes, lance `auto_resolve` toutes les heures, sync le command tree Discord au démarrage. |
| `database.py` | Couche SQLite. 2 tables + 6 fonctions. Aucune logique métier. |
| `requirements.txt` | Dépendances pinnées. **⚠️ Contient httpx, flask, stripe — plus utilisés.** |

### Commandes

| Fichier | Rôle |
|---|---|
| `commands/prono.py` | `/prono` — lit `wc2026_fixtures.csv` (lru_cache), affiche le Select UI, appelle `predict_match()` dans un thread (blocking → async), sauvegarde en DB. |
| `commands/standings.py` | `/standings` — lit DB pour les scores réels, calcule classement (points/GD/GF), appelle `predict_match()` pour les matchs à venir. |
| `commands/accuracy.py` | `/accuracy` — appelle `database.get_stats()`, formate en message Discord. |

### Services

| Fichier | Rôle |
|---|---|
| `services/ml_model.py` | `format_result(dict) → str`. Fonction pure de formatage. Barres Unicode, bold sur la prédiction, indicateur de confiance. |
| `services/wc_resolver.py` | Résolution horaire. Appelle football-data.org, mappe les noms (10 entrées dans `_FDORG_TO_FIXTURE`), résout les prédictions DB, enregistre les scores, déclenche la mise à jour ELO. |
| `services/elo_updater.py` | Mise à jour ELO post-match. Lit `elo_history.csv` + `wc_elo_updates.csv`, calcule les nouveaux ELO (K=40), appende dans `wc_elo_updates.csv`, invalide le cache de `predict.py`. |

### ML

| Fichier | Rôle |
|---|---|
| `ml/features.py` | Feature engineering. Filtre les matchs sans score (`dropna`), calcule ELO pré-match (`shift(1)`), forme rolling 5 matchs (`shift(1) + rolling(5)`), H2H pre-indexé. Exporte `FEATURE_COLS` (16 features). |
| `ml/train.py` | Entraînement. Baseline ELO naïf, XGBoost avec `compute_sample_weight('balanced')`, calibration isotonique (pour analyse), optimisation de seuil (pour analyse). **Sauvegarde le XGBoost brut.** |
| `ml/predict.py` | Inférence. `lru_cache` sur results.csv + elo_history.csv + model.pkl. Fusionne `wc_elo_updates.csv` si existant. API publique : `predict_match(home, away, date)`. |
| `ml/pipeline.py` | Orchestrateur. `features → train → run_wc2026`. Skip les étapes existantes. `--force` pour tout recalculer. |
| `ml/run_wc2026.py` | Batch predictions. Lit `wc2026_fixtures.csv`, prédit les 72 matchs, affiche par groupe, sauvegarde `wc2026_predictions.csv`. |

### Base de données (SQLite — `football.db`)

**Table `predictions`** : prédictions des utilisateurs via `/prono`
```
match_id (UNIQUE), competition, home_team, away_team,
predicted_home_goals, predicted_away_goals, predicted_result,
actual_home_goals, actual_away_goals, actual_result,
is_correct_result, is_correct_score, created_at, resolved_at
```

**Table `match_results`** : scores réels des matchs joués (source : wc_resolver)
```
match_id (PRIMARY KEY), home_team, away_team,
home_score, away_score, match_group, match_date
```

**Pas de `server.py`** : le webhook Stripe/Flask n'existe pas dans la codebase actuelle.

### Données ML (`ml/data/`)

| Fichier | Taille | Rôle | Commitable |
|---|---|---|---|
| `results.csv` | 3.7 MB | Mart Jürisoo — 49 287 matchs internationaux 1872-2024 + 72 WC 2026 (NaN scores) | ✅ oui |
| `elo_history.csv` | 3.9 MB | ELO après chaque match historique. **⚠️ CORROMPU** — inclut les WC 2026 avec NaN traités comme défaites. À régénérer. | ✅ oui |
| `wc2026_fixtures.csv` | 7.4 KB | Calendrier officiel CdM 2026 (104 matchs). Groupe stage : équipes réelles. KO : TBD. | ✅ oui |
| `wc2026_teams.csv` | 1.2 KB | Mapping FIFA name → dataset name (10 entrées divergentes). | ✅ oui |
| `wc2026_predictions.csv` | 5.7 KB | Prédictions batch pré-générées. **⚠️ BASÉ SUR ELO CORROMPU.** À régénérer. | ✅ oui |
| `features.csv` | 8.7 MB | Feature matrix pour entraînement. **⚠️ BASÉ SUR ELO CORROMPU.** À régénérer. | ❌ .gitignore |
| `elo_history.csv` (ELO legacy) | — | `elo_latest.csv`, `elo_wc2026.csv` — produits du notebook, non utilisés en prod. | ✅ oui |
| `wc_elo_updates.csv` | N/A | Généré en prod après chaque match joué. Absent localement. | ❌ runtime only |

---

## 3. État du pipeline ML

### Architecture décidée
- **Algorithme** : XGBoost (`multi:softprob`, 3 classes)
- **Class weights** : `compute_sample_weight('balanced')` — rééquilibre les nuls (~25%)
- **Calibration** : analysée mais **non retenue** pour le modèle final
- **Seuil nul** : analysé mais **non retenu** (draw_threshold = null dans model_config)
- **Décision finale** : XGBoost brut + class weights = meilleur compromis accuracy/recall nuls

### Features (16)
```
ELO         : elo_home, elo_away, elo_diff
Forme dom.  : home_form_pts, home_form_gf, home_form_ga, home_form_n
Forme ext.  : away_form_pts, away_form_gf, away_form_ga, away_form_n
H2H         : h2h_home_pts, h2h_gd, h2h_n
Contexte    : is_neutral, tournament_tier
```

### Split temporel
```
Train  : ≤ 2017-12-31
Val    : 2018-01-01 → 2021-12-31  (early stopping XGBoost)
Test   : 2022-01-01 → 2024-12-31  (verdict final, touché une seule fois)
```

### Métriques
**⚠️ `model.pkl`, `metrics.json`, `model_config.json` sont basés sur l'ELO corrompu.**
Le pipeline n'a pas encore été relancé après le fix du 03/06 (`dropna` sur les matchs NaN).

**Action requise avant la mise en prod** :
1. Régénérer `elo_history.csv` via le notebook (ajouter la cellule `dropna` avant la boucle ELO)
2. `python -m ml.pipeline --force`

### ELO en temps réel (prod)
- `elo_history.csv` : source immuable (1872-2024)
- `wc_elo_updates.csv` : delta live (2 lignes par match joué, K=40)
- Fusion automatique dans `predict.py._data()` à chaque inférence après invalidation du cache
- **⚠️ Non persisté entre redémarrages Railway** (filesystem éphémère)

---

## 4. Ce qui fonctionne en prod vs en cours

### ✅ Fonctionnel en prod (Railway)
- Bot Discord connecté et actif
- `/prono` : prédictions ML pour les 72 matchs du groupe stage
- `/standings` : classement en temps réel (vide tant que des matchs ne sont pas joués)
- `/accuracy` : affichage de la précision (vide tant que des matchs ne sont pas résolus)
- `auto_resolve` : tâche horaire qui tente de résoudre les prédictions via football-data.org
- ELO mis à jour automatiquement après chaque match (si `wc_elo_updates.csv` existe)

### ⚠️ En attente / action requise
- **ELO et modèle corrompus** : régénération nécessaire avant le 11/06 (premier match)
- **`wc_elo_updates.csv` non persisté** : si Railway redémarre en cours de tournoi, l'ELO CdM est perdu. Les prédictions reprendront depuis l'ELO pré-CdM.
- **Résolution des prédictions WC** : fonctionne seulement si l'utilisateur a utilisé `/prono` pour ce match (sinon pas de prédiction à résoudre) ET si le mapping `_FDORG_TO_FIXTURE` est complet
- **`/standings` Groupe stage seulement** : les phases éliminatoires ne seront pas prédites automatiquement

### ❌ Non implémenté
- `server.py` (Flask/Stripe) : absent de la codebase
- Monétisation / premium
- Commandes admin (résolution manuelle, mise à jour bracket KO)
- Résolution automatique pour les matchs sans prédiction DB
- Persistance de `wc_elo_updates.csv` entre redémarrages Railway

---

## 5. Ce qui manque pour un produit complet

### Monétisation
- Système de tiers (gratuit : N prédictions/jour, premium : illimité)
- Intégration Stripe (paiement) + Flask webhook (confirmation)
- Rôle Discord "Premium" accordé automatiquement post-paiement
- Commande `/premium` avec lien de checkout

### Déploiement
- **Volume Railway persistant** pour `wc_elo_updates.csv` (sinon ELO perdu au redémarrage)
- Nettoyage `requirements.txt` : retirer `httpx`, `flask`, `stripe` (plus utilisés)
- Monitoring : alertes si `auto_resolve` échoue ou si football-data.org est down
- Rollback propre si `model.pkl` est corrompu au démarrage

### Features ML
- **Phases éliminatoires** : commande admin `/admin fixture <num> <home> <away>` pour remplir le bracket + prédictions automatiques
- **Résolution complète** : enregistrer TOUS les matchs WC en DB (pas seulement ceux prédits via `/prono`) pour que `/accuracy` reflète la réalité complète
- **Calibration courbe fiabilité** : vérifier que "60% France" → France gagne ~60% du temps sur les matchs résolus
- **Features supplémentaires** : rang FIFA, absences joueurs clés, distance parcourue (fatigue)
- **Retrain en cours de tournoi** : après la phase de groupes, réentraîner sur les données CdM 2026 pour affiner les prédictions KO

### UX Discord
- Notification automatique ("prochain match dans 2h : voici la prédiction") via webhook
- Historique des prédictions par utilisateur
- Commande `/bracket` pour afficher l'arbre des phases éliminatoires

---

## 6. Les 3 prochaines étapes prioritaires

### Étape 1 — Régénérer le modèle avant le 11/06 (URGENT)
Le modèle actuel est entraîné sur des ELO corrompus. Avant le premier match (11 juin) :
1. Ajouter la cellule `dropna` dans le notebook avant la boucle ELO
2. Re-run le notebook complet → génère `elo_history.csv` propre
3. `python -m ml.pipeline --force` → génère features, modèle, et prédictions
4. Committer `model.pkl`, `model_config.json`, `metrics.json`, `wc2026_predictions.csv`

### Étape 2 — Persistance de l'ELO sur Railway (AVANT LE 11/06)
Railway redémarre le pod à chaque déploiement. `wc_elo_updates.csv` est sur le filesystem éphémère → perdu à chaque redémarrage. Solution : après chaque journée, committer `wc_elo_updates.csv` dans git et pusher. Le bot rechargera le fichier depuis le repo au prochain démarrage.
Commande à lancer localement après chaque journée de groupe :
```bash
git add ml/data/wc_elo_updates.csv && git commit -m "chore: wc_elo_updates journee X" && git push
```

### Étape 3 — Résolution complète et `/accuracy` significatif
Actuellement, `/accuracy` ne compte que les matchs pour lesquels un utilisateur a fait `/prono`. Pour avoir des stats représentatives, le resolver devrait enregistrer TOUS les matchs WC dans `match_results` ET dans `predictions` (avec la prédiction du modèle, pas seulement de l'utilisateur). Cela signifie pré-remplir `predictions` au démarrage du bot pour tous les matchs du groupe stage, avec la prédiction ML correspondante.
