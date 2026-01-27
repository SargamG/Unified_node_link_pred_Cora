import os
import json
from datetime import datetime

LEADERBOARD_PATH = "leaderboard.json"

def load_leaderboard():
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH) as f:
            return json.load(f)
    return {"last_updated": None, "submissions": []}

def save_leaderboard(data):
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(data, f, indent=2)

def main():
    participant = os.environ.get("GITHUB_ACTOR", "unknown")

    with open("scores.json") as f:
        scores = json.load(f)

    node_f1 = scores["node_f1"]
    link_auc = scores["link_auc"]
    final_score = scores["final_score"]

    data = load_leaderboard()

    old = next(
        (s for s in data["submissions"] if s["participant"] == participant),
        None
    )
    
    if old and final_score <= old["final_score"]:
        print("Score not better than previous. Leaderboard unchanged.")
        return
    
    # Remove old entry
    data["submissions"] = [
        s for s in data["submissions"]
        if s["participant"] != participant
    ]

    # Add new entry
    data["submissions"].append({
        "participant": participant,
        "node_f1": node_f1,
        "link_auc": link_auc,
        "final_score": final_score,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Sort
    data["submissions"] = sorted(
        data["submissions"],
        key=lambda x: x["final_score"],
        reverse=True
    )

    data["last_updated"] = datetime.utcnow().isoformat()
    save_leaderboard(data)

    print("Leaderboard updated.")

if __name__ == "__main__":
    main()
