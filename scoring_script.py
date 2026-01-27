import pandas as pd
import sys
from sklearn.metrics import f1_score, roc_auc_score
import json

"""
Usage:
    python scoring_script.py submissions/submission.csv

Assumptions:
- test.csv           : public, contains id, task_type, src, dst
- test_labels.csv    : hidden, contains id, task_type, label
- submission.csv     : participant file with columns: id, prediction
"""

def main():
    if len(sys.argv) != 2:
        print("Usage: python scoring_script.py submissions/submission.csv")
        sys.exit(1)

    submission_path = sys.argv[1]

    # -------------------------------
    # Load files
    # -------------------------------
    submission = pd.read_csv(submission_path)
    truth = pd.read_csv("data/test_labels.csv")

    # Basic sanity checks
    if "id" not in submission.columns or "prediction" not in submission.columns:
        raise ValueError("Submission must contain columns: id, prediction")

    if len(submission) != len(truth):
        raise ValueError("Submission size does not match test set size")

    # Merge predictions with ground truth
    df = truth.merge(submission, on="id", how="left")

    if df["prediction"].isnull().any():
        raise ValueError("Some test IDs are missing predictions")

    # -------------------------------
    # Split by task type
    # -------------------------------
    node_df = df[df["task_type"] == "node"]
    link_df = df[df["task_type"] == "link"]

    # -------------------------------
    # Node classification metric
    # -------------------------------
    # prediction = class label (int)
    node_f1 = f1_score(
        node_df["label"],
        node_df["prediction"].astype(int),
        average="macro"
    )

    # -------------------------------
    # Link prediction metric
    # -------------------------------
    # prediction = probability in [0, 1]
    link_auc = roc_auc_score(
        link_df["label"],
        link_df["prediction"].astype(float)
    )

    # -------------------------------
    # Final combined score
    # -------------------------------
    final_score = 0.5 * node_f1 + 0.5 * link_auc

    # -------------------------------
    # Print results
    # -------------------------------
    print(f"Node Macro-F1 : {node_f1:.4f}")
    print(f"Link ROC-AUC  : {link_auc:.4f}")
    print(f"Final Score  : {final_score:.4f}")

    scores = {
        "node_f1": node_f1,
        "link_auc": link_auc,
        "final_score": final_score,
    }
    
    with open("scores.json", "w") as f:
        json.dump(scores, f)
    
if __name__ == "__main__":
    main()
