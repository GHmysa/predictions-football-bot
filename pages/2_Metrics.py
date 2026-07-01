import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

import database

st.set_page_config(page_title="Métriques ML", page_icon="📈", layout="wide")
st.title("📈 Métriques ML")

METRICS_PATH = Path(__file__).parent.parent / "ml" / "metrics.json"

col_xgb, col_poisson = st.columns(2)

# ── XGBoost ──────────────────────────────────────────────────────────────────
with col_xgb:
    st.subheader("XGBoost — modèle général")
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            m = json.load(f)

        rows = []
        for split in ("xgb_val", "xgb_test", "baseline_val", "baseline_test"):
            if split in m:
                rows.append({
                    "Split": split,
                    "Accuracy": f"{m[split]['accuracy']:.1%}",
                    "Log-loss": m[split].get("log_loss", "—"),
                    "Draw rate": f"{m[split].get('draw_pred_rate', 0):.1%}",
                    "Draw recall": f"{m[split].get('draw_recall', 0):.1%}",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if "best_iteration" in m:
            st.caption(f"Best iteration XGBoost : {m['best_iteration']}")

        with st.expander("Features (20)"):
            st.write(m.get("features", []))
    else:
        st.warning("ml/metrics.json introuvable — relancer ml/pipeline.py")

# ── Poisson ──────────────────────────────────────────────────────────────────
with col_poisson:
    st.subheader("Poisson Dixon-Coles")
    from ml.poisson import fit_or_load
    params = fit_or_load()

    st.metric("Matchs d'entraînement", params["n_matches"])

    pcol1, pcol2 = st.columns(2)
    pcol1.metric("home_adv", f"{params['home_adv']:.4f}",
                 help="Multiplicateur d'avantage terrain (λ_home × home_adv)")
    pcol2.metric("rho", f"{params['rho']:.4f}",
                 help="Correction Dixon-Coles pour scores faibles (0-0, 1-0, 0-1, 1-1)")

    if "ref_date" in params:
        st.caption(f"Dernière mise à jour : {params['ref_date']}")

    with st.expander("Top 10 attaques"):
        attacks = {k: v for k, v in params.items() if k.startswith("attack_")}
        top_att = sorted(attacks.items(), key=lambda x: -x[1])[:10]
        att_df = pd.DataFrame(top_att, columns=["Paramètre", "Valeur"])
        att_df["Équipe"] = att_df["Paramètre"].str.replace("attack_", "")
        st.dataframe(att_df[["Équipe", "Valeur"]].round(3), hide_index=True)

    with st.expander("Top 10 défenses (plus petit = meilleur)"):
        defenses = {k: v for k, v in params.items() if k.startswith("defense_")}
        top_def = sorted(defenses.items(), key=lambda x: x[1])[:10]
        def_df = pd.DataFrame(top_def, columns=["Paramètre", "Valeur"])
        def_df["Équipe"] = def_df["Paramètre"].str.replace("defense_", "")
        st.dataframe(def_df[["Équipe", "Valeur"]].round(3), hide_index=True)

# ── Accuracy live ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Accuracy live (prédictions résolues)")

stats = database.get_stats()
c1, c2, c3 = st.columns(3)
c1.metric("Total résolu", stats["total"])
c2.metric("Corrects H/D/A", stats["correct_results"])
c3.metric("Accuracy globale", f"{stats['result_rate']}%")

if stats["by_competition"]:
    comp_df = pd.DataFrame([
        {"Compétition": k, "Total": v["total"],
         "Corrects": v["correct_results"], "Accuracy": f"{v['rate']}%"}
        for k, v in stats["by_competition"].items()
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
