import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

# ── Accuracy par phase ────────────────────────────────────────────────────────
FIXTURES_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"
STAGE_ORDER = ["Group Stage", "Round of 32", "Round of 16", "Quarter Finals", "Semi Finals", "Finals"]

@st.cache_data(ttl=30)
def _accuracy_by_phase() -> pd.DataFrame:
    fixtures = pd.read_csv(FIXTURES_PATH)[["match_number", "stage"]]
    with database.get_connection() as conn:
        rows = conn.execute("""
            SELECT match_id, is_correct_result
            FROM predictions
            WHERE actual_result IS NOT NULL
        """).fetchall()
    if not rows:
        return pd.DataFrame()
    pred_df = pd.DataFrame(rows, columns=["match_id", "is_correct_result"])
    pred_df["match_number"] = pred_df["match_id"] - 200_000
    merged = pred_df.merge(fixtures, on="match_number", how="left")
    merged["stage"] = merged["stage"].fillna("Unknown")
    grouped = (
        merged.groupby("stage")
        .agg(total=("is_correct_result", "count"), correct=("is_correct_result", "sum"))
        .reset_index()
    )
    grouped["accuracy"] = (grouped["correct"] / grouped["total"] * 100).round(1)
    grouped["Phase"] = grouped["stage"]
    # Preserve stage order
    cat = pd.Categorical(grouped["Phase"], categories=STAGE_ORDER, ordered=True)
    grouped["Phase"] = cat
    return grouped.sort_values("Phase")[["Phase", "total", "correct", "accuracy"]].rename(
        columns={"total": "Matchs", "correct": "Corrects", "accuracy": "Accuracy (%)"}
    )

phase_df = _accuracy_by_phase()
if not phase_df.empty:
    st.subheader("Accuracy par phase")
    st.dataframe(
        phase_df.style.format({"Accuracy (%)": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )

# ── Courbe d'accuracy cumulée ─────────────────────────────────────────────────
st.divider()
st.subheader("Accuracy cumulée au fil du tournoi")


@st.cache_data(ttl=30)
def _accuracy_curve() -> pd.DataFrame:
    fixtures = pd.read_csv(FIXTURES_PATH)[["match_number", "stage", "date"]]
    with database.get_connection() as conn:
        rows = conn.execute("""
            SELECT match_id, is_correct_result
            FROM predictions
            WHERE actual_result IS NOT NULL
        """).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["match_id", "is_correct_result"])
    df["match_number"] = df["match_id"] - 200_000
    merged = df.merge(fixtures, on="match_number", how="left").sort_values("date")
    merged = merged.reset_index(drop=True)
    merged["match_idx"] = range(1, len(merged) + 1)
    merged["cumulative_acc"] = merged["is_correct_result"].expanding().mean() * 100
    return merged


curve_df = _accuracy_curve()

if curve_df.empty:
    st.info("Aucune prédiction résolue pour le moment.")
else:
    fig = go.Figure()

    # ── Ligne accuracy cumulée
    fig.add_trace(go.Scatter(
        x=curve_df["match_idx"],
        y=curve_df["cumulative_acc"].round(1),
        mode="lines",
        name="Accuracy cumulée",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))

    # ── Dots par match (vert = correct, rouge = incorrect)
    for correct, color, label in ((1, "#2ecc71", "Correct"), (0, "#e74c3c", "Incorrect")):
        mask = curve_df["is_correct_result"] == correct
        fig.add_trace(go.Scatter(
            x=curve_df.loc[mask, "match_idx"],
            y=curve_df.loc[mask, "cumulative_acc"].round(1),
            mode="markers",
            name=label,
            marker=dict(color=color, size=7),
            hovertemplate=(
                curve_df.loc[mask, "date"].astype(str)
                + "<br>"
                + f"{'✅' if correct else '❌'}"
                + "<extra></extra>"
            ),
        ))

    # ── Baseline 33.3%
    fig.add_hline(
        y=33.3,
        line_dash="dot",
        line_color="gray",
        annotation_text="Baseline aléatoire (33%)",
        annotation_position="bottom right",
    )

    # ── Marqueurs de transition de phase
    stage_starts = (
        curve_df.groupby("stage")["match_idx"].min()
        .reindex([s for s in STAGE_ORDER if s != "Group Stage"])
        .dropna()
    )
    for stage, idx in stage_starts.items():
        fig.add_vline(
            x=idx - 0.5,
            line_dash="dash",
            line_color="rgba(255,255,255,0.3)",
            annotation_text=stage,
            annotation_position="top left",
            annotation_font_size=10,
        )

    fig.update_layout(
        xaxis_title="Nème prédiction résolue",
        yaxis_title="Accuracy cumulée (%)",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=20, b=40),
        height=400,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Chaque point = un match joué · {len(curve_df)} prédictions résolues au total")
