"""
dashboard.py — Point d'entrée Streamlit du dashboard admin WC 2026.

Lancement local : streamlit run dashboard.py
Railway        : web: streamlit run dashboard.py --server.port $PORT --server.headless true
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="WC 2026 Admin",
    page_icon="🏆",
    layout="wide",
)

import database
from ml.poisson import fit_or_load as _poisson_params

st.title("🏆 WC 2026 — Dashboard Admin")


@st.cache_data(ttl=30)
def _overview() -> dict:
    with database.get_connection() as conn:
        n_results  = conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
        n_preds    = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        n_resolved = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE actual_result IS NOT NULL"
        ).fetchone()[0]
        n_correct  = conn.execute(
            "SELECT COALESCE(SUM(is_correct_result), 0) FROM predictions WHERE actual_result IS NOT NULL"
        ).fetchone()[0]
    return dict(n_results=n_results, n_preds=n_preds, n_resolved=n_resolved, n_correct=n_correct)


stats  = _overview()
params = _poisson_params()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Matchs joués", stats["n_results"])
col2.metric("Prédictions résolues", f"{stats['n_resolved']}/{stats['n_preds']}")
accuracy = round(stats["n_correct"] / stats["n_resolved"] * 100, 1) if stats["n_resolved"] else 0
col3.metric("Accuracy H/D/A", f"{accuracy}%")
col4.metric("Poisson — matchs fit", params["n_matches"])

st.divider()
st.markdown(
    "Naviguez via le menu de gauche :  \n"
    "**📊 Prédictions** · **📈 Métriques** · **⚽ Scores** · **🏆 ELO**"
)
