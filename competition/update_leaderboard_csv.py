from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_existing(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if (r.get("team") or "").strip()]


def _write_all(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "team",
        "model",
        "score",
        "node_f1",
        "link_auc",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to leaderboard.csv")
    ap.add_argument("--team", required=True, help="Team name")
    ap.add_argument("--model", default="", help="Model type (e.g., human, llm-only, human+llm)")
    ap.add_argument("--notes", default="", help="Notes")
    ap.add_argument("--score", required=True, type=float, help="Final score")
    ap.add_argument("--node-f1", required=True, type=float, help="Node Macro-F1")
    ap.add_argument("--link-auc", required=True, type=float, help="Link ROC-AUC")
    ap.add_argument("--metadata", default="", help="Optional metadata.json path (overrides model/notes)")
    args = ap.parse_args()

    team = args.team.strip()
    if not team:
        raise ValueError("team must be non-empty")

    model = (args.model or "").strip()
    notes = (args.notes or "").strip()

    if args.metadata:
        meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        model = (meta.get("model") or model).strip()
        notes = (meta.get("notes") or notes).strip()
        team = (meta.get("team") or team).strip()

    csv_path = Path(args.csv)
    rows = _read_existing(csv_path)

    # Enforce ONE submission per team
    for r in rows:
        if (r.get("team") or "").strip() == team:
            raise ValueError(f"Team '{team}' has already submitted. Only one submission is allowed.")


    rows.append(
        {
            "timestamp_utc": _utc_now_iso(),
            "team": team,
            "model": model.lower(),
            "score": f"{args.score:.8f}",
            "node_f1": f"{args.node_f1:.8f}",
            "link_auc": f"{args.link_auc:.8f}",
            "notes": notes,
        }
    )

    _write_all(csv_path, rows)


if __name__ == "__main__":
    main()

