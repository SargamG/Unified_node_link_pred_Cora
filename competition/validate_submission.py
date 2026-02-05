from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_preds(pred_path: Path) -> pd.DataFrame:
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
        "--test",
        required=True,
        help="Path to public test file with IDs (expected: data/public/test.csv)",
    )
    ap.add_argument(
        "--write-metadata",
        default="",
        help="Optional path to write inferred metadata.json stub (empty fields if missing).",
    )
    args = ap.parse_args()

    pred_path = Path(args.pred)
    test_path = Path(args.test)

    preds = _load_preds(pred_path)
    test = pd.read_csv(test_path)

    if "id" not in test.columns:
        raise ValueError("test file must contain column: id")

    if preds["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in predictions.csv")

    if preds["prediction"].isna().any():
        raise ValueError("NaN predictions found in predictions.csv")

    pred_ids = set(preds["id"].astype(str))
    test_ids = set(test["id"].astype(str))
    if pred_ids != test_ids:
        missing = sorted(list(test_ids - pred_ids))[:10]
        extra = sorted(list(pred_ids - test_ids))[:10]
        raise ValueError(
            "Prediction IDs do not match test IDs. "
            f"missing(first10)={missing} extra(first10)={extra}"
        )

    # Optional: basic value checks when task_type is available in test.csv
    if "task_type" in test.columns:
        merged = test[["id", "task_type"]].merge(preds, on="id", how="left")
        node_rows = merged[merged["task_type"] == "node"]
        link_rows = merged[merged["task_type"] == "link"]

        if len(node_rows):
            # node predictions should be integer-ish labels
            coerced = pd.to_numeric(node_rows["prediction"], errors="coerce")
            if coerced.isna().any():
                raise ValueError("Node task rows contain non-numeric predictions")
            if (coerced % 1 != 0).any():
                raise ValueError("Node task predictions must be integer class labels")

        if len(link_rows):
            coerced = pd.to_numeric(link_rows["prediction"], errors="coerce")
            if coerced.isna().any():
                raise ValueError("Link task rows contain non-numeric predictions")
            if ((coerced < 0) | (coerced > 1)).any():
                raise ValueError("Link task predictions must be probabilities in [0,1]")

    if args.write_metadata:
        out = Path(args.write_metadata)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "team": "",
                    "model": "",
                    "llm_name": "",
                    "notes": "",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print("VALID SUBMISSION")


if __name__ == "__main__":
    main()

