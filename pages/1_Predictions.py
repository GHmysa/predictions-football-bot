import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

import database

st.set_page_config(page_title="Prédictions", page_icon="📊", layout="wide")
st.title("📊 Prédictions")

RESULT_LABEL = {"H": "Domicile", "D": "Nul", "A": "Extérieur"}


FIXTURES_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"


@st.cache_data(ttl=30)
def _load() -> pd.DataFrame:
    with database.get_connection() as conn:
        rows = conn.execute("""
            SELECT match_id, home_team, away_team,
                   predicted_home_goals, predicted_away_goals, predicted_result,
                   actual_home_goals, actual_away_goals, actual_result,
                   is_correct_result, created_at
            FROM predictions
            ORDER BY created_at DESC
        """).fetchall()
    df = pd.DataFrame(rows, columns=[
        "match_id", "Domicile", "Extérieur", "Pred dom", "Pred ext", "Pred résultat",
        "Réel dom", "Réel ext", "Réel résultat", "Correct", "Date prédiction",
    ])
    fixtures = pd.read_csv(FIXTURES_PATH)[["match_number", "stage", "date"]]
    df["match_number"] = df["match_id"] - 200_000
    df = df.merge(fixtures, on="match_number", how="left")
    df.rename(columns={"stage": "Phase", "date": "Date match"}, inplace=True)
    return df


df = _load()
df["Pred résultat"] = df["Pred résultat"].map(RESULT_LABEL).fillna(df["Pred résultat"])
df["Réel résultat"] = df["Réel résultat"].map(RESULT_LABEL).fillna(df["Réel résultat"])

tab_pending, tab_resolved = st.tabs(["⏳ En attente", "✅ Résolues"])

with tab_pending:
    pending = df[df["Réel résultat"].isna()][
        ["Date match", "Phase", "Domicile", "Extérieur", "Pred dom", "Pred ext", "Pred résultat"]
    ]
    st.caption(f"{len(pending)} prédictions en attente de résultat")
    st.dataframe(pending, use_container_width=True, hide_index=True)

with tab_resolved:
    resolved = df[df["Réel résultat"].notna()].copy()
    resolved[""] = resolved["Correct"].map({1: "✅", 0: "❌"})
    display = resolved[["Date match", "Phase", "Domicile", "Extérieur", "Pred résultat", "Réel résultat", ""]]
    st.caption(f"{len(resolved)} prédictions résolues")
    st.dataframe(display, use_container_width=True, hide_index=True)

    if len(resolved) > 0:
        acc = resolved["Correct"].sum() / len(resolved)
        st.metric("Accuracy", f"{acc:.1%}", help="Taux de bons résultats H/D/A")

# ── Export CSV ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Export")

export_cols = [
    "Date match", "Phase", "Domicile", "Extérieur",
    "Pred résultat", "Pred dom", "Pred ext",
    "Réel résultat", "Réel dom", "Réel ext", "Correct",
]
export_df = df[export_cols].copy()

st.download_button(
    label="⬇️ Télécharger toutes les prédictions (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="wc2026_predictions.csv",
    mime="text/csv",
)
