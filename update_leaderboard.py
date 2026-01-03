import os
import re
import pandas as pd

LEADERBOARD_PATH = "leaderboard.md"
SCORE_LOG_PATH = "score.txt"

def parse_scores(score_log_path):
    """
    Extract Node F1, Link AUC, and Final Score
    from scoring_script.py output.
    """
    with open(score_log_path, "r") as f:
        text = f.read()

    # More flexible regex that handles variable whitespace
    node_match = re.search(r"Node Macro-F1\s*:\s*([0-9.]+)", text)
    link_match = re.search(r"Link ROC-AUC\s*:\s*([0-9.]+)", text)
    final_match = re.search(r"Final Score\s*:\s*([0-9.]+)", text)
    
    if not node_match or not link_match or not final_match:
        raise ValueError(f"Could not parse scores from {score_log_path}. Content:\n{text}")
    
    node_f1 = float(node_match.group(1))
    link_auc = float(link_match.group(1))
    final_score = float(final_match.group(1))

    return node_f1, link_auc, final_score


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

    node_f1, link_auc, final_score = parse_scores(SCORE_LOG_PATH)
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
