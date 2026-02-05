import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "leaderboard" / "leaderboard.csv"
MD_PATH = ROOT / "leaderboard" / "leaderboard.md"


def read_rows():
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if (r.get("team") or "").strip()]
    return rows


def main():
    rows = read_rows()

    def score_key(r):
        try:
            return float(r.get("score", "-inf"))
        except Exception:
            return float("-inf")

    def ts_key(r):
        try:
            return datetime.fromisoformat((r.get("timestamp_utc", "") or "").replace("Z", "+00:00"))
        except Exception:
            return datetime.fromtimestamp(0)

    rows.sort(key=lambda r: (score_key(r), ts_key(r)), reverse=True)

    lines = []
    lines.append("# 🏆 Leaderboard\n")
    lines.append(
        "This leaderboard is **auto-updated** when a submission PR is merged. "
        "For interactive search and filters, enable GitHub Pages and open **/docs/leaderboard.html**.\n\n"
    )
    lines.append(
        "- **Final Score** = 0.5 × Node Macro-F1 + 0.5 × Link ROC-AUC\n"
        "- Only the **best score per team** is retained.\n\n"
    )

    lines.append("| Rank | Team | Model | Node F1 | Link AUC | Final Score | Date (UTC) | Notes |\n")
    lines.append("|---:|---|---|---:|---:|---:|---|---|\n")
    for i, r in enumerate(rows, start=1):
        team = (r.get("team") or "").strip()
        model = (r.get("model") or "").strip()
        node_f1 = (r.get("node_f1") or "").strip()
        link_auc = (r.get("link_auc") or "").strip()
        score = (r.get("score") or "").strip()
        ts = (r.get("timestamp_utc") or "").strip()
        notes = (r.get("notes") or "").strip()
        model_disp = f"`{model}`" if model else ""
        lines.append(
            f"| {i} | {team} | {model_disp} | {node_f1} | {link_auc} | {score} | {ts} | {notes} |\n"
        )

    MD_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

