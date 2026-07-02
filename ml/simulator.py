"""ml/simulator.py — Monte Carlo simulation for WC 2026."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ml.predict import predict_match

# Cached between tournament simulations — avoids re-running ML inference for
# the same KO matchup across n iterations.
_ko_probs_cache: dict[tuple[str, str], tuple[float, float, float]] = {}


def _match_probs(home: str, away: str, date: str) -> tuple[float, float, float]:
    r = predict_match(home, away, date, is_neutral=True, tournament_tier=4)
    p = r["probabilities"]
    return p["home"], p["draw"], p["away"]


def _ko_probs(home: str, away: str) -> tuple[float, float, float]:
    key = (home, away)
    if key not in _ko_probs_cache:
        _ko_probs_cache[key] = _match_probs(home, away, "2026-07-10")
    return _ko_probs_cache[key]


def _draw(ph: float, pd_: float, pa: float) -> int:
    """Returns 0=home win, 1=draw, 2=away win."""
    p = np.array([ph, pd_, pa])
    p /= p.sum()  # guard against float rounding (predict_match rounds to 3dp)
    return int(np.random.choice(3, p=p))


def _draw_ko(home: str, away: str) -> str:
    """KO round: draw goes to 50-50 penalty shootout."""
    ph, pd_, pa = _ko_probs(home, away)
    outcome = _draw(ph, pd_, pa)
    if outcome == 0:
        return home
    if outcome == 2:
        return away
    return home if np.random.random() < 0.5 else away


def _simulate_group_table(
    pairs: list[tuple[str, str]],
    probs: list[tuple[float, float, float]],
) -> dict[str, dict]:
    table: dict[str, dict] = defaultdict(lambda: {"pts": 0, "gd": 0})
    for (home, away), (ph, pd_, pa) in zip(pairs, probs):
        outcome = _draw(ph, pd_, pa)
        if outcome == 0:
            table[home]["pts"] += 3
            table[home]["gd"] += 1
            table[away]["gd"] -= 1
        elif outcome == 1:
            table[home]["pts"] += 1
            table[away]["pts"] += 1
        else:
            table[away]["pts"] += 3
            table[away]["gd"] += 1
            table[home]["gd"] -= 1
    return table


def _rank_group(table: dict) -> list[str]:
    return sorted(table, key=lambda t: (-table[t]["pts"], -table[t]["gd"], t))


def _resolve_slot(
    slot: str,
    group_ranked: dict[str, list[str]],
    third_pool: list[tuple],
) -> str:
    """
    Resolve a bracket slot to a team name.
    Slots: '1A' (1st of group A), '2B', '3ABCDF' (best 3rd from those groups).
    third_pool: [(team, pts, gd, group)] sorted best-first, mutated in place.
    """
    if len(slot) == 2 and slot[0].isdigit():
        return group_ranked[slot[1]][int(slot[0]) - 1]
    if slot.startswith("3") and len(slot) > 2:
        eligible = set(slot[1:])
        for i, entry in enumerate(third_pool):
            if entry[3] in eligible:
                third_pool.pop(i)
                return entry[0]
        if third_pool:
            return third_pool.pop(0)[0]
    return slot


# ─── Public API ──────────────────────────────────────────────────────────────

def simulate_ko_from_r16(
    fixtures: pd.DataFrame,
    n: int = 10_000,
) -> tuple[dict[str, float] | None, list[str]]:
    """
    Monte Carlo simulation of WC 2026 from R16 onwards.

    Uses consecutive-pair bracket: R16 winners pair up for QF, QF winners for SF, etc.
    This matches the FIFA 2026 bracket structure for rounds R16 and beyond.

    Returns
    -------
    (win_probs, tba_list)
        win_probs : {team: win_probability} sorted descending, or None if any R16 team is TBA
        tba_list  : list of TBA slot descriptions (empty when simulation succeeded)
    """
    r16 = fixtures[fixtures["stage"] == "Round of 16"].sort_values("match_number")

    r16_pairs: list[tuple[str, str]] = []
    tba_list: list[str] = []
    for _, row in r16.iterrows():
        h, a = str(row["home_team"]), str(row["away_team"])
        if h == "To be announced":
            tba_list.append(f"#{int(row['match_number'])} domicile")
        if a == "To be announced":
            tba_list.append(f"#{int(row['match_number'])} extérieur")
        r16_pairs.append((h, a))

    if tba_list:
        return None, tba_list

    win_count: dict[str, int] = defaultdict(int)

    for _ in range(n):
        current = [_draw_ko(h, a) for h, a in r16_pairs]
        while len(current) > 1:
            current = [
                _draw_ko(current[i], current[i + 1])
                for i in range(0, len(current), 2)
            ]
        win_count[current[0]] += 1

    all_teams = list({t for h, a in r16_pairs for t in (h, a)})
    return (
        {
            team: round(win_count[team] / n, 3)
            for team in sorted(all_teams, key=lambda t: -win_count[t])
            if win_count[team] > 0
        },
        [],
    )


def simulate_group(group_matches: list[dict], n: int = 10_000) -> dict[str, float]:
    """
    Monte Carlo simulation of a WC 2026 group.

    Parameters
    ----------
    group_matches : match dicts from wc2026_fixtures.csv (group stage, one group)
    n             : number of simulations

    Returns
    -------
    dict of team → probability of qualifying (top 2 finish), sorted descending
    """
    pairs = [(m["home_team"], m["away_team"]) for m in group_matches]
    probs = [_match_probs(m["home_team"], m["away_team"], m["date"]) for m in group_matches]

    qual_count: dict[str, int] = defaultdict(int)
    for _ in range(n):
        table = _simulate_group_table(pairs, probs)
        for team in _rank_group(table)[:2]:
            qual_count[team] += 1

    all_teams = list({t for pair in pairs for t in pair})
    return {
        team: round(qual_count[team] / n, 3)
        for team in sorted(all_teams, key=lambda t: -qual_count[t])
    }


def simulate_tournament(fixtures: pd.DataFrame, n: int = 10_000) -> dict[str, float]:
    """
    Monte Carlo simulation of the full WC 2026 tournament.

    Group stage uses ML probabilities per match. KO rounds use ML probabilities
    with 50-50 penalty shootout for draws. The 8 best 3rd-place teams (by pts/gd)
    fill the 3rd-place bracket slots.

    Returns dict of team → tournament win probability (non-zero finishers only).
    """
    gs  = fixtures[fixtures["stage"] == "Group Stage"]
    r32 = (
        fixtures[fixtures["stage"] == "Round of 32"]
        .sort_values("match_number")
        .to_dict("records")
    )

    groups = sorted(gs["group"].unique())
    group_data: dict[str, dict] = {}
    for g in groups:
        gdf = gs[gs["group"] == g].sort_values("date")
        group_data[g] = {
            "pairs": [(row["home_team"], row["away_team"]) for _, row in gdf.iterrows()],
            "probs": [
                _match_probs(row["home_team"], row["away_team"], row["date"])
                for _, row in gdf.iterrows()
            ],
        }

    win_count: dict[str, int] = defaultdict(int)

    for _ in range(n):
        group_ranked: dict[str, list[str]] = {}
        thirds: list[tuple] = []

        for g, data in group_data.items():
            table = _simulate_group_table(data["pairs"], data["probs"])
            ranked = _rank_group(table)
            group_ranked[g] = ranked
            third = ranked[2]
            thirds.append((third, table[third]["pts"], table[third]["gd"], g))

        thirds.sort(key=lambda x: (-x[1], -x[2], x[0]))
        third_pool = thirds[:8]

        r32_pairs = [
            (
                _resolve_slot(m["home_team"], group_ranked, third_pool),
                _resolve_slot(m["away_team"], group_ranked, third_pool),
            )
            for m in r32
        ]

        # KO bracket: R32 → R16 → QF → SF, then Final
        current = r32_pairs
        while len(current) > 1:
            next_round = []
            for i in range(0, len(current) - 1, 2):
                w1 = _draw_ko(*current[i])
                w2 = _draw_ko(*current[i + 1])
                next_round.append((w1, w2))
            current = next_round

        if current:
            win_count[_draw_ko(*current[0])] += 1

    all_teams = [t for data in group_data.values() for pair in data["pairs"] for t in pair]
    return {
        team: round(win_count[team] / n, 3)
        for team in sorted(set(all_teams), key=lambda t: -win_count[t])
        if win_count[team] > 0
    }
