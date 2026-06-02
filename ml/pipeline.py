"""
ml/pipeline.py — Point d'entrée unique pour tout le pipeline ML.

Étapes dans l'ordre :
  1. Feature engineering  → ml/data/features.csv
  2. Entraînement         → ml/model.pkl + ml/metrics.json
  3. Prédictions CdM 2026 → ml/data/wc2026_predictions.csv

Les étapes 1 et 2 sont skippées si les artifacts existent déjà.
Utilise --force pour tout reconstruire depuis zéro.

Exécution :
    python -m ml.pipeline           # skip si déjà fait
    python -m ml.pipeline --force   # tout recalculer
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from ml.features import DATA_DIR, build_features
from ml.train import MODEL_PATH, train
from ml.run_wc2026 import display, run_group_stage

FEATURES_PATH = DATA_DIR / "features.csv"
METRICS_PATH  = Path(__file__).parent / "metrics.json"


def _step(label: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  {label}")
    print(f"{'━'*60}")


def run(force: bool = False) -> None:
    """Exécute le pipeline complet. Si force=False, skippe les étapes déjà faites."""
    t_total = time.time()

    # --- Étape 1 : Feature engineering ---
    if force or not FEATURES_PATH.exists():
        _step("Étape 1/3 — Feature engineering")
        t0 = time.time()
        features = build_features()
        features.to_csv(FEATURES_PATH, index=False)
        print(f"→ {len(features):,} matchs  |  {time.time() - t0:.1f}s")
    else:
        _step("Étape 1/3 — Feature engineering  [skippée — features.csv existe]")

    # --- Étape 2 : Entraînement ---
    if force or not MODEL_PATH.exists():
        _step("Étape 2/3 — Entraînement XGBoost")
        t0 = time.time()
        train(features_path=FEATURES_PATH)
        print(f"→ Entraînement terminé en {time.time() - t0:.1f}s")
    else:
        _step("Étape 2/3 — Entraînement  [skippé — model.pkl existe]")

    # --- Étape 3 : Prédictions CdM 2026 (toujours rafraîchies) ---
    _step("Étape 3/3 — Prédictions groupe stage CdM 2026")
    t0 = time.time()
    predictions = run_group_stage()
    display(predictions)

    out = DATA_DIR / "wc2026_predictions.csv"
    predictions.to_csv(out, index=False)
    print(f"\n→ {len(predictions)} matchs prédits  |  {time.time() - t0:.1f}s")
    print(f"→ Sauvegardé : {out}")

    print(f"\n{'━'*60}")
    print(f"  Pipeline terminé en {time.time() - t_total:.1f}s")
    print(f"{'━'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML CdM 2026")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalcule features et modèle même s'ils existent déjà",
    )
    args = parser.parse_args()
    run(force=args.force)
