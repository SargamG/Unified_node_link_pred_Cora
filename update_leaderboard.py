import os
import pandas as pd
import json

LEADERBOARD_PATH = "leaderboard.md"

def parse_scores(_=None):
    if not os.path.exists("scores.json"):
        raise RuntimeError("scores.json not found. Scoring step likely failed.")

    with open("scores.json", "r") as f:
        scores = json.load(f)

    return scores["node_f1"], scores["link_auc"], scores["final_score"]


def load_leaderboard():
    """
    Load leaderboard.md into a DataFrame.
    """
    if not os.path.exists(LEADERBOARD_PATH):
        return pd.DataFrame(
            columns=["Participant", "Node F1", "Link AUC", "Final Score"]
        )

    rows = []
    with open(LEADERBOARD_PATH, "r") as f:
        for line in f:
            if line.startswith("|") and "Participant" not in line and "---" not in line:
                parts = [p.strip() for p in line.strip().split("|")[1:-1]]
                if parts[0] != "—":
                    rows.append(parts)

    if not rows:
        return pd.DataFrame(
            columns=["Participant", "Node F1", "Link AUC", "Final Score"]
        )

    df = pd.DataFrame(
        rows,
        columns=["Rank", "Participant", "Node F1", "Link AUC", "Final Score"]
    ).drop(columns=["Rank"])

    df[["Node F1", "Link AUC", "Final Score"]] = df[
        ["Node F1", "Link AUC", "Final Score"]
    ].astype(float)

    return df


def save_leaderboard(df):
    """
    Write updated leaderboard.md.
    """
    df = df.sort_values("Final Score", ascending=False).reset_index(drop=True)

    lines = []
    lines.append("# 🏆 Leaderboard\n")
    lines.append(
        "This leaderboard tracks the best submission score for each participant.\n\n"
    )
    lines.append(
        "- **Final Score** = 0.5 × Node Macro-F1 + 0.5 × Link ROC-AUC\n"
    )
    lines.append(
        "- Only the **best score per participant** is retained.\n"
    )
    lines.append(
        "- Submissions are evaluated automatically via GitHub Actions.\n\n"
    )
    lines.append(
        "---\n\n"
    )
    lines.append(
        "| Rank | Participant | Node F1 | Link AUC | Final Score |\n"
    )
    lines.append(
        "|------|-------------|---------|----------|-------------|\n"
    )

    for i, row in df.iterrows():
        lines.append(
            f"| {i+1} | {row['Participant']} | "
            f"{row['Node F1']:.4f} | {row['Link AUC']:.4f} | {row['Final Score']:.4f} |\n"
        )

    with open(LEADERBOARD_PATH, "w") as f:
        f.writelines(lines)


def main():
    participant = os.environ.get("GITHUB_ACTOR", "unknown")

    node_f1, link_auc, final_score = parse_scores()
    leaderboard = load_leaderboard()

    # Update only if better score
    if participant in leaderboard["Participant"].values:
        prev_score = leaderboard.loc[
            leaderboard["Participant"] == participant, "Final Score"
        ].values[0]

        if final_score <= prev_score:
            print("Submission score not better than previous. Leaderboard unchanged.")
            return

        leaderboard = leaderboard[leaderboard["Participant"] != participant]

    new_row = {
        "Participant": participant,
        "Node F1": node_f1,
        "Link AUC": link_auc,
        "Final Score": final_score,
    }

    leaderboard = pd.concat(
        [leaderboard, pd.DataFrame([new_row])],
        ignore_index=True
    )

    save_leaderboard(leaderboard)
    print("Leaderboard updated successfully.")


if __name__ == "__main__":
    main()
