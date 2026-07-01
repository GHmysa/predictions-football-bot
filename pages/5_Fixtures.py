"""
pages/5_Fixtures.py — Mise à jour des équipes "To be announced" pour les phases KO.

Les modifications sont écrites dans wc2026_fixtures.csv.
Le bot Discord les prend en compte au prochain redémarrage (son cache _fixtures()
est module-level et ne se vide pas entre les requêtes Discord).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

FIXTURES_PATH = Path(__file__).parent.parent / "ml" / "data" / "wc2026_fixtures.csv"

st.set_page_config(page_title="Fixtures", page_icon="🗓️", layout="wide")
st.title("🗓️ Mise à jour des Fixtures")


@st.cache_data(ttl=10)
def _load() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_PATH)


def _save(df: pd.DataFrame) -> None:
    df.to_csv(FIXTURES_PATH, index=False)
    st.cache_data.clear()


df = _load()
tba = df[
    (df["home_team"] == "To be announced") | (df["away_team"] == "To be announced")
].sort_values("date")

# ── Matches à compléter ───────────────────────────────────────────────────────
if tba.empty:
    st.success("Toutes les fixtures ont des équipes définies.")
else:
    st.info(f"{len(tba)} match(s) avec équipe(s) à définir.")

    # Liste des équipes encore qualifiées (apparaissent dans des matchs déjà connus)
    known_teams = sorted(set(
        df[~df["home_team"].isin(["To be announced"])]["home_team"].tolist() +
        df[~df["away_team"].isin(["To be announced"])]["away_team"].tolist()
    ))

    for _, row in tba.iterrows():
        mn       = int(row["match_number"])
        stage    = row["stage"]
        date     = row["date"]
        home_cur = row["home_team"]
        away_cur = row["away_team"]

        with st.expander(f"**#{mn} — {stage} · {date}**  ({home_cur} vs {away_cur})"):
            col_h, col_a = st.columns(2)

            if home_cur == "To be announced":
                home_new = col_h.selectbox(
                    "Équipe domicile",
                    options=["To be announced"] + known_teams,
                    key=f"home_{mn}",
                )
            else:
                home_new = home_cur
                col_h.markdown(f"**Domicile** : {home_cur}")

            if away_cur == "To be announced":
                away_new = col_a.selectbox(
                    "Équipe extérieure",
                    options=["To be announced"] + known_teams,
                    key=f"away_{mn}",
                )
            else:
                away_new = away_cur
                col_a.markdown(f"**Extérieur** : {away_cur}")

            if home_new != "To be announced" and away_new != "To be announced":
                if st.button(f"✅ Confirmer #{mn}", key=f"btn_{mn}"):
                    df_fresh = pd.read_csv(FIXTURES_PATH)
                    df_fresh.loc[df_fresh["match_number"] == mn, "home_team"] = home_new
                    df_fresh.loc[df_fresh["match_number"] == mn, "away_team"] = away_new
                    _save(df_fresh)
                    st.success(f"#{mn} mis à jour : **{home_new} vs {away_new}**")
                    st.rerun()

# ── Vue complète des fixtures KO ──────────────────────────────────────────────
st.divider()
st.subheader("Toutes les fixtures KO")

ko = df[df["stage"] != "Group Stage"][
    ["match_number", "stage", "date", "home_team", "away_team"]
].copy()
ko.columns = ["#", "Phase", "Date", "Domicile", "Extérieur"]

st.dataframe(
    ko.style.apply(
        lambda row: ["background-color: #3d2020" if "To be announced" in str(row.values) else "" for _ in row],
        axis=1,
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "⚠️ Les modifications écrivent directement dans `ml/data/wc2026_fixtures.csv` sur ce serveur. "
    "Pour les rendre permanentes (survive au redéploiement), commite et push le fichier CSV."
)
