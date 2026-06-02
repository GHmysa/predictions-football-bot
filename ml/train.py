"""
ml/train.py — Entraînement et évaluation du modèle de prédiction.

Split temporel strict (pas de shuffle) :
  train  : < 2018
  val    : 2018 – 2021 (inclus)
  test   : 2022 – 2024 (inclus)

Exécution : python -m ml.train   (depuis predictions-football-bot/)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
)
from xgboost import XGBClassifier

from ml.features import build_features, DATA_DIR, FEATURE_COLS

MODEL_PATH   = Path(__file__).parent / "model.pkl"
METRICS_PATH = Path(__file__).parent / "metrics.json"
TARGET_COL = "result"  # 0=away, 1=draw, 2=home

# Dates de coupure pour le split temporel
TRAIN_END = "2017-12-31"
VAL_END   = "2021-12-31"
TEST_END  = "2024-12-31"


# ---------------------------------------------------------------------------
# Baseline : prédit toujours l'équipe avec le meilleur ELO
# (draw si elo_diff < seuil)
# ---------------------------------------------------------------------------

def baseline_predict(X: pd.DataFrame, draw_threshold: float = 50.0) -> np.ndarray:
    """
    Règle naïve ELO :
      - elo_diff > +threshold  → victoire domicile (2)
      - elo_diff < -threshold  → victoire extérieur (0)
      - sinon                  → nul (1)
    """
    preds = np.where(
        X["elo_diff"] > draw_threshold, 2,
        np.where(X["elo_diff"] < -draw_threshold, 0, 1),
    )
    return preds


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray,
             y_proba: np.ndarray | None = None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    metrics = {"accuracy": round(float(acc), 4)}

    if y_proba is not None:
        ll = log_loss(y_true, y_proba, labels=[0, 1, 2])
        metrics["log_loss"] = round(float(ll), 4)

    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {acc:.4f}  ({int(acc * len(y_true))}/{len(y_true)})")
    if y_proba is not None:
        print(f"  Log-loss  : {ll:.4f}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted)")
    print(f"  Labels : 0=away  1=draw  2=home")
    cm_df = pd.DataFrame(cm, index=["actual_away", "actual_draw", "actual_home"],
                          columns=["pred_away", "pred_draw", "pred_home"])
    print(cm_df.to_string())
    return metrics


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def train(features_path: str | Path | None = None) -> XGBClassifier:
    features_path = Path(features_path or DATA_DIR / "features.csv")

    if not features_path.exists():
        print("features.csv not found — running build_features()...")
        df = build_features()
        df.to_csv(features_path, index=False)
    else:
        df = pd.read_csv(features_path, parse_dates=["date"])

    print(f"Dataset : {len(df):,} matchs  |  {df['result'].value_counts().to_dict()}")

    # --- Split temporel ---
    train_df = df[df["date"] <= TRAIN_END]
    val_df   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test_df  = df[(df["date"] > VAL_END)   & (df["date"] <= TEST_END)]

    print(f"\nSplit temporel :")
    train_end_yr = int(TRAIN_END[:4])
    val_end_yr   = int(VAL_END[:4])
    test_end_yr  = int(TEST_END[:4])
    print(f"  Train  (≤ {TRAIN_END})              : {len(train_df):>6,} matchs")
    print(f"  Val    ({train_end_yr + 1} – {val_end_yr})                    : {len(val_df):>6,} matchs")
    print(f"  Test   ({val_end_yr + 1}  – {test_end_yr})                    : {len(test_df):>6,} matchs")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL].values
    X_val,   y_val   = val_df[FEATURE_COLS],   val_df[TARGET_COL].values
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df[TARGET_COL].values

    # --- Baseline ---
    print("\n=== BASELINE (ELO naïf) ===")
    baseline_val  = evaluate("Baseline — validation",  y_val,  baseline_predict(X_val))
    baseline_test = evaluate("Baseline — test",        y_test, baseline_predict(X_test))

    # --- XGBoost ---
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"\nXGBoost best iteration : {model.best_iteration}")

    # --- Évaluation ---
    print("\n=== XGBOOST ===")
    val_proba  = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test)

    xgb_val  = evaluate("XGBoost — validation", y_val,  model.predict(X_val),  val_proba)
    xgb_test = evaluate("XGBoost — test",       y_test, model.predict(X_test), test_proba)

    # --- Gain vs baseline ---
    print(f"\n  Gain accuracy vs baseline (test) : "
          f"+{(xgb_test['accuracy'] - baseline_test['accuracy']):.4f}")

    # --- Feature importance ---
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\nFeature importance (top 10) :")
    print(importance.head(10).map("{:.4f}".format).to_string())

    # --- Sauvegarde ---
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModèle sauvegardé → {MODEL_PATH}")

    all_metrics = {
        "baseline_val":  baseline_val,
        "baseline_test": baseline_test,
        "xgb_val":       xgb_val,
        "xgb_test":      xgb_test,
        "best_iteration": int(model.best_iteration),
        "features":      FEATURE_COLS,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Métriques sauvegardées → {METRICS_PATH}")

    return model


if __name__ == "__main__":
    train()
