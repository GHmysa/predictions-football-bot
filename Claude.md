# discord-football-bot — Contexte projet

## Description
Bot Discord de prédiction de matchs football avec pipeline ML intégré.
Projet portfolio orienté stage ML Engineer.

## Stack
- Python 3.12, venv activé via `.\venv\Scripts\activate`
- discord.py, Flask, Stripe, SQLite
- pandas, scikit-learn, XGBoost, matplotlib, seaborn
- API : football-data.org (données temps réel)
- Hébergement : Railway (bot) / Azure prévu (ML)

## Structure
- `bot.py` — point d'entrée Discord
- `commands/` — commandes slash Discord
- `services/` — API football, IA, resolver
- `database.py` — SQLite cache + predictions
- `ml/` — pipeline ML complet (en cours)
- `server.py` — Flask webhook Stripe

## Pipeline ML — Coupe du Monde 2026
- Données : dataset Mart Jürisoo (results.csv) + ELO calculés maison
- Target : résultat 1/N/2 (classification 3 classes)
- Modèle : XGBoost avec split temporel
- Features : ELO diff, forme récente, H2H, terrain neutre, tournoi
- Évaluation : accuracy + log loss + matrice de confusion

## Conventions
- Commits : `feat:`, `fix:`, `refactor:`, `chore:`
- Toujours expliquer le code comme si j'apprends
- Ne jamais générer plusieurs fichiers sans demander
- Signaler les mauvaises pratiques

## Variables d'environnement (.env)
DISCORD_TOKEN, FOOTBALL_DATA_KEY, MISTRAL_API_KEY,
AI_PROVIDER, STRIPE_SECRET_KEY, STRIPE_PRICE_ID,
STRIPE_WEBHOOK_SECRET, PREMIUM_ROLE_ID