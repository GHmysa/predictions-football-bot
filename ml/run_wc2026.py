"""
ml/run_wc2026.py — Prédictions batch pour tous les matchs du groupe stage CdM 2026.

Lit wc2026_fixtures.csv, prédit chaque match avec le modèle entraîné,
et affiche les résultats groupés par poule.

Exécution : python -m ml.run_wc2026   (depuis predictions-football-bot/)
"""
from __future__ import annotations

import pandas as pd
from ml.predict import predict_match
from ml.features import DATA_DIR

FIXTURES_PATH = DATA_DIR / "wc2026_fixtures.csv"


def run_group_stage() -> pd.DataFrame:
    """Prédit tous les matchs du groupe stage et retourne un DataFrame de résultats."""
    fixtures = pd.read_csv(FIXTURES_PATH)
    group_stage = fixtures[fixtures["stage"] == "Group Stage"].copy()

    rows = []
    for _, match in group_stage.iterrows():
        result = predict_match(
            home_team=match["home_team"],
            away_team=match["away_team"],
            date=match["date"],
            is_neutral=True,
            tournament_tier=4,
        )
        p = result["probabilities"]
        rows.append({
            "group":      match["group"],
            "date":       match["date"],
            "home_team":  match["home_team"],
            "away_team":  match["away_team"],
            "prediction": result["prediction_fr"],
            "conf":       f'{result["confidence"]:.0%}',
            "p_home":     f'{p["home"]:.0%}',
            "p_draw":     f'{p["draw"]:.0%}',
            "p_away":     f'{p["away"]:.0%}',
            "elo_home":   result["elo_home"],
            "elo_away":   result["elo_away"],
        })

    return pd.DataFrame(rows)


def display(predictions: pd.DataFrame) -> None:
    """Affiche les prédictions par groupe dans le terminal."""
    for group in sorted(predictions["group"].unique()):
        grp = predictions[predictions["group"] == group]
        print(f"\n{'='*72}")
        print(f"  GROUPE {group}")
        print(f"{'='*72}")
        print(f"  {'Date':<12} {'Domicile':<26} {'Extérieur':<26} {'Prédiction':<22} {'Conf':>5}  Dom./Nul/Ext.")
        print(f"  {'-'*70}")
        for _, r in grp.iterrows():
            print(
                f"  {r['date']:<12} "
                f"{r['home_team']:<26} "
                f"{r['away_team']:<26} "
                f"{r['prediction']:<22} "
                f"{r['conf']:>5}  "
                f"{r['p_home']} / {r['p_draw']} / {r['p_away']}"
            )


if __name__ == "__main__":
    print("Chargement du modèle et des données...")
    predictions = run_group_stage()
    display(predictions)

    out = DATA_DIR / "wc2026_predictions.csv"
    predictions.to_csv(out, index=False)
    print(f"\n{'='*72}")
    print(f"Predictions sauvegardees -> {out}")
    print(f"Total : {len(predictions)} matchs prédits")
