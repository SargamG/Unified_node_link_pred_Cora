from __future__ import annotations

import argparse

import pandas as pd

from metrics import link_roc_auc, node_macro_f1


def _load_preds(pred_path: str) -> pd.DataFrame:
    preds = pd.read_csv(pred_path)
    if "id" not in preds.columns:
        raise ValueError("predictions.csv must contain column: id")
    if "y_pred" in preds.columns:
        preds = preds.rename(columns={"y_pred": "prediction"})
    if "prediction" not in preds.columns:
        raise ValueError("predictions.csv must contain column: y_pred (preferred) or prediction")
    return preds[["id", "prediction"]].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Path to predictions.csv")
    ap.add_argument(
        "--labels",
        required=True,
        help="Path to hidden test labels CSV (expected columns: id, task_type, label)",
    )
    args = ap.parse_args()

    preds = _load_preds(args.pred).sort_values("id")
    truth = pd.read_csv(args.labels).sort_values("id")

    if not {"id", "task_type", "label"}.issubset(set(truth.columns)):
        raise ValueError("labels file must contain columns: id, task_type, label")

    merged = truth.merge(preds, on="id", how="left")
    if merged["prediction"].isna().any():
        missing = merged[merged["prediction"].isna()]["id"].astype(str).head(10).tolist()
        raise ValueError(f"Missing predictions for some IDs (first10={missing})")

    node_df = merged[merged["task_type"] == "node"].copy()
    link_df = merged[merged["task_type"] == "link"].copy()

    if len(node_df) == 0 or len(link_df) == 0:
        raise ValueError("labels file must contain both task_type == 'node' and task_type == 'link'")

    node_f1 = node_macro_f1(
        node_df["label"].astype(int),
        pd.to_numeric(node_df["prediction"], errors="raise").astype(int),
    )
    link_auc = link_roc_auc(
        link_df["label"].astype(int),
        pd.to_numeric(link_df["prediction"], errors="raise").astype(float),
    )
    final_score = 0.5 * node_f1 + 0.5 * link_auc

    # Print human-readable breakdown + machine-readable score line
    print(f"NODE_F1={node_f1:.8f}")
    print(f"LINK_AUC={link_auc:.8f}")
    print(f"SCORE={final_score:.8f}")


if __name__ == "__main__":
    main()

