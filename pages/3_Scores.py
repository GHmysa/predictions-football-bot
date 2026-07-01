import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

import database

FIXTURES_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"

st.set_page_config(page_title="Scores", page_icon="⚽", layout="wide")
st.title("⚽ Scores")

tab_missing, tab_recent, tab_entry = st.tabs(["⚠️ Scores manquants", "📋 Scores récents", "➕ Saisir un score"])

# ── Scores manquants ─────────────────────────────────────────────────────────
with tab_missing:
    @st.cache_data(ttl=15)
    def _missing_scores() -> pd.DataFrame:
        fixtures = pd.read_csv(FIXTURES_PATH)
        today = pd.Timestamp("today").normalize()
        played = fixtures[
            (fixtures["home_team"] != "To be announced") &
            (fixtures["away_team"] != "To be announced") &
            (pd.to_datetime(fixtures["date"]) < today)
        ]["match_number"].tolist()

        with database.get_connection() as conn:
            entered = {
                r[0] - 200_000
                for r in conn.execute("SELECT match_id FROM match_results").fetchall()
            }

        missing_nums = [mn for mn in played if mn not in entered]
        if not missing_nums:
            return pd.DataFrame()

        missing = fixtures[fixtures["match_number"].isin(missing_nums)].copy()
        return missing[["match_number", "stage", "date", "home_team", "away_team"]].sort_values("date")

    missing_df = _missing_scores()
    if missing_df.empty:
        st.success("Tous les matchs passés ont un score enregistré.")
    else:
        st.warning(f"{len(missing_df)} match(s) joué(s) sans score en DB :")
        missing_df.columns = ["#", "Phase", "Date", "Domicile", "Extérieur"]
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

# ── Scores récents ────────────────────────────────────────────────────────────
with tab_recent:
    @st.cache_data(ttl=15)
    def _load_results() -> pd.DataFrame:
        with database.get_connection() as conn:
            rows = conn.execute("""
                SELECT home_team, away_team, home_score, away_score, match_date, match_group
                FROM match_results
                ORDER BY match_date DESC
                LIMIT 40
            """).fetchall()
        df = pd.DataFrame(rows, columns=["Domicile", "Extérieur", "Dom", "Ext", "Date", "Groupe"])
        df["Score"] = df["Dom"].astype(str) + " – " + df["Ext"].astype(str)
        return df[["Date", "Groupe", "Domicile", "Score", "Extérieur"]]

    st.dataframe(_load_results(), use_container_width=True, hide_index=True)

# ── Saisie de score ───────────────────────────────────────────────────────────
with tab_entry:
    st.markdown(
        "Équivalent de `/score` sur Discord : met à jour la DB, l'ELO et refitte le Poisson.  \n"
        "⚠️ Le bot Discord utilisera les nouveaux paramètres au prochain redémarrage ou via `/score`."
    )

    fixtures = pd.read_csv(FIXTURES_PATH)

    col_num, col_info = st.columns([1, 3])
    with col_num:
        match_number = st.number_input("Numéro du match", min_value=1, max_value=104, step=1, value=1)

    row = fixtures[fixtures["match_number"] == match_number]
    if not row.empty:
        r = row.iloc[0]
        with col_info:
            st.info(f"**#{int(match_number)} — {r['home_team']} vs {r['away_team']}**  \n{r['date']} · {r['stage']}")

        col_h, col_a = st.columns(2)
        with col_h:
            home_score = st.number_input(f"Buts {r['home_team']}", min_value=0, max_value=20, step=1, key="home_score")
        with col_a:
            away_score = st.number_input(f"Buts {r['away_team']}", min_value=0, max_value=20, step=1, key="away_score")

        if st.button("✅ Enregistrer le score", type="primary"):
            with st.spinner("Mise à jour ELO + refit Poisson (~10s)..."):
                from commands.admin import _apply_score
                msg = _apply_score(int(match_number), int(home_score), int(away_score))
            st.success(msg.replace("**", "").replace("_", ""))
            st.cache_data.clear()
    else:
        with col_info:
            st.warning(f"Match #{int(match_number)} introuvable dans les fixtures.")
