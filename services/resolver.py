import os
import requests
import database

BASE_URL = "https://api.football-data.org/v4"


def _headers() -> dict:
    return {"X-Auth-Token": os.getenv("FOOTBALL_DATA_KEY")}


def _fetch_match(match_id: int) -> dict | None:
    try:
        response = requests.get(
            f"{BASE_URL}/matches/{match_id}",
            headers=_headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[RESOLVER] Erreur fetch match {match_id} : {e}")
        return None


def resolve_pending_predictions() -> None:
    pending = database.get_pending_predictions()
    if not pending:
        print("[RESOLVER] Aucun pronostic en attente.")
        return

    print(f"[RESOLVER] {len(pending)} pronostic(s) en attente à vérifier…")

    for p in pending:
        match_id = p["match_id"]
        home     = p["home_team"]
        away     = p["away_team"]
        pred_h   = p["pred_home"]
        pred_a   = p["pred_away"]

        data = _fetch_match(match_id)
        if data is None:
            continue

        status = data.get("status")
        if status != "FINISHED":
            print(f"[RESOLVER] {home} vs {away} — statut '{status}', skip")
            continue

        score = data.get("score", {}).get("fullTime", {})
        actual_h = score.get("home")
        actual_a = score.get("away")

        if actual_h is None or actual_a is None:
            print(f"[RESOLVER] {home} vs {away} — score manquant, skip")
            continue

        database.resolve_prediction(match_id, actual_h, actual_a)

        correct = (
            (pred_h > pred_a and actual_h > actual_a) or
            (pred_h == pred_a and actual_h == actual_a) or
            (pred_h < pred_a and actual_h < actual_a)
        )
        icon = "✅" if correct else "❌"
        print(
            f"[RESOLVER] Résolu : {home} {actual_h}-{actual_a} {away} "
            f"| Prédit : {pred_h}-{pred_a} | {icon}"
        )
