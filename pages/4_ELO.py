import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

ELO_HISTORY_PATH = Path(__file__).parent.parent / "ml" / "data" / "elo_history.csv"
_PERSISTENT_DIR  = Path(os.environ.get("PERSISTENT_DIR", Path(__file__).parent.parent / "ml" / "data"))
WC_ELO_PATH      = _PERSISTENT_DIR / "wc_elo_updates.csv"
WC_TEAMS_PATH    = Path(__file__).parent.parent / "ml" / "data" / "wc2026_teams.csv"

st.set_page_config(page_title="ELO", page_icon="🏆", layout="wide")
st.title("🏆 ELO — Équipes CdM 2026")


@st.cache_data(ttl=60)
def _load_elo() -> pd.DataFrame:
    elo = pd.read_csv(ELO_HISTORY_PATH, parse_dates=["date"])
    if WC_ELO_PATH.exists() and WC_ELO_PATH.stat().st_size > 0:
        wc = pd.read_csv(WC_ELO_PATH, parse_dates=["date"])
        elo = pd.concat([elo, wc], ignore_index=True)
    return elo.sort_values("date")


@st.cache_data(ttl=300)
def _wc_current_elos() -> pd.DataFrame:
    wc_teams = pd.read_csv(WC_TEAMS_PATH)
    elo_all  = _load_elo()
    rows = []
    for _, t in wc_teams.iterrows():
        subset = elo_all[elo_all["team"] == t["dataset_name"]]
        if not subset.empty:
            rows.append({
                "fifa_name":    t["fifa_name"],
                "dataset_name": t["dataset_name"],
                "elo":          round(float(subset.iloc[-1]["elo"]), 1),
                "last_match":   subset.iloc[-1]["date"].date(),
            })
    return (
        pd.DataFrame(rows)
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )


df = _wc_current_elos()
df.index += 1

col_table, col_chart = st.columns([1, 2])

with col_table:
    st.subheader("Classement ELO actuel")
    display = df[["fifa_name", "elo", "last_match"]].copy()
    display.columns = ["Équipe", "ELO", "Dernier match"]
    st.dataframe(display, use_container_width=True)

with col_chart:
    st.subheader("Graphe ELO")
    fig = px.bar(
        df.head(20),
        x="elo",
        y="fifa_name",
        orientation="h",
        labels={"elo": "ELO", "fifa_name": ""},
        color="elo",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Historique ELO d'une équipe ───────────────────────────────────────────────
st.divider()
st.subheader("Historique ELO — zoom équipe")

team_options = df["fifa_name"].tolist()
selected     = st.selectbox("Équipe", team_options)

if selected:
    dataset_name = df[df["fifa_name"] == selected]["dataset_name"].values[0]
    history = _load_elo()
    team_history = history[history["team"] == dataset_name].tail(50)

    if not team_history.empty:
        fig2 = px.line(
            team_history,
            x="date", y="elo",
            labels={"date": "Date", "elo": "ELO"},
            title=f"ELO {selected} — 50 derniers matchs",
        )
        fig2.update_traces(line_color="#1f77b4")
        st.plotly_chart(fig2, use_container_width=True)
