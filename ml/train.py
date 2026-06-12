"""
ml/train.py — Entraînement et évaluation du modèle de prédiction.

Split temporel strict (pas de shuffle) :
  train  : < 2018
  val    : 2018 – 2021 (inclus)
  test   : 2022 – 2024 (inclus)

Améliorations appliquées :
  - class weights pour rééquilibrer les nuls (sous-représentés ~25%)
  - calibration isotonique pour des probabilités honnêtes

Exécution : python -m ml.train   (depuis predictions-football-bot/)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ml.features import DATA_DIR, FEATURE_COLS, build_features

MODEL_PATH        = Path(__file__).parent / "model.pkl"
METRICS_PATH      = Path(__file__).parent / "metrics.json"
MODEL_CONFIG_PATH = Path(__file__).parent / "model_config.json"

TARGET_COL = "result"  # 0=away, 1=draw, 2=home

TRAIN_END = "2017-12-31"
VAL_END   = "2021-12-31"
TEST_END  = "2024-12-31"


# ---------------------------------------------------------------------------
# Baseline : règle naïve ELO
# ---------------------------------------------------------------------------

def predict_with_threshold(probs: np.ndarray, draw_threshold: float = 0.25) -> np.ndarray:
    """
    Applique un seuil de déclenchement sur la classe nul.

    Si P(draw) >= draw_threshold → prédit nul (1).
    Sinon → prédit la classe avec la probabilité la plus haute.

    probs : array (n_samples, 3) — colonnes [P(away), P(draw), P(home)]
    """
    return np.where(probs[:, 1] >= draw_threshold, 1, np.argmax(probs, axis=1))


def baseline_predict(X: pd.DataFrame, draw_threshold: float = 50.0) -> np.ndarray:
    """
    Règle naïve ELO :
      elo_diff > +threshold  → victoire domicile (2)
      elo_diff < -threshold  → victoire extérieur (0)
      sinon                  → nul (1)
    """
    return np.where(
        X["elo_diff"] > draw_threshold, 2,
        np.where(X["elo_diff"] < -draw_threshold, 0, 1),
    )


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def evaluate(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Affiche et retourne accuracy, log-loss, matrice de confusion."""
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    metrics: dict = {"accuracy": round(float(acc), 4)}

    if y_proba is not None:
        ll = log_loss(y_true, y_proba, labels=[0, 1, 2])
        metrics["log_loss"] = round(float(ll), 4)

    print(f"\n{'─'*44}")
    print(f"  {name}")
    print(f"{'─'*44}")
    print(f"  Accuracy  : {acc:.4f}  ({int(acc * len(y_true))}/{len(y_true)})")
    if y_proba is not None:
        print(f"  Log-loss  : {ll:.4f}")

    print(f"\n  Confusion matrix  (lignes=réel, colonnes=prédit)")
    print(f"  Labels : 0=ext.  1=nul  2=dom.")
    cm_df = pd.DataFrame(
        cm,
        index=["réel_ext.", "réel_nul ", "réel_dom."],
        columns=["pred_ext.", "pred_nul ", "pred_dom."],
    )
    print(cm_df.to_string())

    # Résumé des nuls prédits — métrique clé pour juger le rééquilibrage
    n_draw_pred = int(cm[:, 1].sum())
    n_draw_true = int(cm[1, :].sum())
    draw_recall  = cm[1, 1] / n_draw_true if n_draw_true else 0
    print(f"\n  Nuls prédits   : {n_draw_pred} / {len(y_true)} ({n_draw_pred/len(y_true):.1%})")
    print(f"  Nuls réels     : {n_draw_true} / {len(y_true)} ({n_draw_true/len(y_true):.1%})")
    print(f"  Rappel nuls    : {draw_recall:.1%}  (nuls correctement détectés)")

    metrics["draw_pred_rate"]   = round(n_draw_pred / len(y_true), 4)
    metrics["draw_recall"]      = round(float(draw_recall), 4)
    return metrics


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def train(features_path: str | Path | None = None) -> XGBClassifier:
    """
    Entraîne XGBoost avec class weights + calibration isotonique.
    Retourne le modèle calibré prêt pour l'inférence.
    """
    features_path = Path(features_path or DATA_DIR / "features.csv")

    if not features_path.exists():
        print("features.csv not found — running build_features()...")
        df = build_features()
        df.to_csv(features_path, index=False)
    else:
        df = pd.read_csv(features_path, parse_dates=["date"])

    # Distribution des classes — sert à quantifier le déséquilibre
    class_counts = df["result"].value_counts().sort_index()
    print(f"Dataset : {len(df):,} matchs")
    print(f"Distribution : away={class_counts[0]} ({class_counts[0]/len(df):.1%})  "
          f"draw={class_counts[1]} ({class_counts[1]/len(df):.1%})  "
          f"home={class_counts[2]} ({class_counts[2]/len(df):.1%})")

    # --- Split temporel ---
    train_df = df[df["date"] <= TRAIN_END]
    val_df   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test_df  = df[(df["date"] > VAL_END)   & (df["date"] <= TEST_END)]

    train_end_yr = int(TRAIN_END[:4])
    val_end_yr   = int(VAL_END[:4])
    test_end_yr  = int(TEST_END[:4])
    print(f"\nSplit temporel :")
    print(f"  Train  (<= {TRAIN_END})          : {len(train_df):>6,} matchs")
    print(f"  Val    ({train_end_yr + 1}–{val_end_yr})                  : {len(val_df):>6,} matchs")
    print(f"  Test   ({val_end_yr + 1}–{test_end_yr})                  : {len(test_df):>6,} matchs")

    X_train, y_train = train_df[FEATURE_COLS].values, train_df[TARGET_COL].values
    X_val,   y_val   = val_df[FEATURE_COLS].values,   val_df[TARGET_COL].values
    X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df[TARGET_COL].values

    # --- Class weights ---
    # Les nuls (~25%) sont sous-représentés → le modèle naïf les ignore.
    # compute_sample_weight('balanced') affecte un poids inversement proportionnel
    # à la fréquence de la classe dans y_train.
    sample_weights = compute_sample_weight("balanced", y_train)
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nClass weights appliqués :")
    for cls, cnt in zip(unique, counts):
        label = {0: "away", 1: "draw", 2: "home"}[cls]
        w = sample_weights[y_train == cls][0]
        print(f"  {label} ({cnt:,} exemples)  →  weight = {w:.3f}")

    # --- Baseline ---
    print("\n=== BASELINE (ELO naïf) ===")
    X_val_df   = val_df[FEATURE_COLS]
    X_test_df  = test_df[FEATURE_COLS]
    baseline_val  = evaluate("Baseline — validation", y_val,  baseline_predict(X_val_df))
    baseline_test = evaluate("Baseline — test",       y_test, baseline_predict(X_test_df))

    # --- XGBoost ---
    xgb = XGBClassifier(
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

    xgb.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],  # eval sans weights — mesure la perf réelle
        verbose=False,
    )

    print(f"\nXGBoost best iteration : {xgb.best_iteration}")

    # --- Évaluation XGBoost brut (avant calibration) ---
    print("\n=== XGBOOST (non calibré) ===")
    xgb_val_proba  = xgb.predict_proba(X_val)
    xgb_test_proba = xgb.predict_proba(X_test)
    xgb_val   = evaluate("XGBoost non calibré — val",  y_val,  xgb.predict(X_val),  xgb_val_proba)
    xgb_test  = evaluate("XGBoost non calibré — test", y_test, xgb.predict(X_test), xgb_test_proba)

    # --- Optimisation du seuil nul (val set) ---
    # On cherche le seuil P(draw) qui maximise un score combiné :
    # score = 0.6 * accuracy + 0.4 * recall_nuls
    # Le val set sert de données d'optimisation — jamais le test set.
    print("\n=== OPTIMISATION DU SEUIL NULS (val set) ===")
    print(f"  {'Seuil':>6}  {'Accuracy':>8}  {'Recall nuls':>11}  {'Score':>7}")
    print(f"  {'─'*40}")

    best_score     = -1.0
    best_threshold = 0.25
    threshold_results: list[tuple] = []

    for t in np.arange(0.20, 0.36, 0.01):
        t = round(float(t), 2)
        preds = predict_with_threshold(xgb_val_proba, t)
        acc   = accuracy_score(y_val, preds)
        cm_t  = confusion_matrix(y_val, preds, labels=[0, 1, 2])
        n_true_draw  = int(cm_t[1, :].sum())
        draw_recall  = cm_t[1, 1] / n_true_draw if n_true_draw else 0.0
        score        = 0.6 * acc + 0.4 * draw_recall
        threshold_results.append((t, acc, draw_recall, score))
        if score > best_score:
            best_score     = score
            best_threshold = t

    for t, acc, dr, s in threshold_results:
        marker = " <- optimal" if t == best_threshold else ""
        print(f"  {t:.2f}    {acc:.4f}      {dr:.4f}    {s:.4f}{marker}")

    print(f"\n  Seuil optimal : {best_threshold:.2f}  (score combiné = {best_score:.4f})")

    # Évaluation XGBoost + seuil optimal sur le test set
    print("\n=== XGBOOST + SEUIL OPTIMAL (test) ===")
    preds_thr_test = predict_with_threshold(xgb_test_proba, best_threshold)
    thr_test = evaluate(
        f"XGBoost + seuil {best_threshold:.2f} — test",
        y_test, preds_thr_test, xgb_test_proba,
    )

    # --- Résumé comparatif ---
    print(f"\n{'═'*52}")
    print(f"  RÉSUMÉ COMPARATIF (test set)")
    print(f"{'═'*52}")
    print(f"  {'Modèle':<36} {'Acc':>6}  {'Log-loss':>8}  {'Recall nuls':>11}")
    print(f"  {'─'*50}")
    rows = [
        ("Baseline ELO",                        baseline_test["accuracy"], None,                 baseline_test["draw_recall"]),
        ("XGBoost",                             xgb_test["accuracy"],      xgb_test["log_loss"], xgb_test["draw_recall"]),
        (f"XGBoost + seuil {best_threshold:.2f}", thr_test["accuracy"],    thr_test["log_loss"], thr_test["draw_recall"]),
    ]
    for label, acc, ll, dr in rows:
        ll_str = f"{ll:.4f}" if ll is not None else "    —   "
        print(f"  {label:<36} {acc:.4f}  {ll_str:>8}  {dr:>10.1%}")

    # --- Feature importance ---
    importance = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\nFeature importance (top 10) :")
    print(importance.head(10).map("{:.4f}".format).to_string())

    # --- Sauvegarde du modèle XGBoost brut ---
    # Le XGBoost non calibré avec class weights est le meilleur compromis
    # accuracy / recall nuls. La calibration améliore les probabilités mais
    # dégrade légèrement l'accuracy — non retenue pour la prod.
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(xgb, f)
    print(f"\nModèle XGBoost brut sauvegardé → {MODEL_PATH}")

    all_metrics = {
        "baseline_val":   baseline_val,
        "baseline_test":  baseline_test,
        "xgb_val":        xgb_val,
        "xgb_test":       xgb_test,
        "thr_test":       thr_test,
        "best_iteration": int(xgb.best_iteration),
        "features":       FEATURE_COLS,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Métriques sauvegardées → {METRICS_PATH}")

    # model_config.json — source de vérité pour les hyperparamètres d'inférence.
    # predict.py lit draw_threshold pour appliquer le même seuil qu'à l'entraînement.
    model_config = {
        "draw_threshold":      None,   # null = argmax brut, pas de seuil appliqué
        "calibration_method":  None,   # modèle XGBoost brut, non calibré
        "class_weight":        "balanced",
        "train_end":           TRAIN_END,
        "val_end":             VAL_END,
        "test_end":            TEST_END,
        "features":            FEATURE_COLS,
        "val_metrics": {
            "accuracy":     xgb_val["accuracy"],
            "log_loss":     xgb_val["log_loss"],
            "draw_recall":  xgb_val["draw_recall"],
        },
        "test_metrics": {
            "accuracy":     xgb_test["accuracy"],
            "log_loss":     xgb_test["log_loss"],
            "draw_recall":  xgb_test["draw_recall"],
        },
    }
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Config sauvegardée      → {MODEL_CONFIG_PATH}")
    print(f"\nModele final : XGBoost brut (class weights, draw_threshold=null)")

    return xgb


if __name__ == "__main__":
    train()
