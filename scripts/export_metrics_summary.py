from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _best_epoch(rows: list[dict]) -> dict | None:
    best = None
    for row in rows:
        test = row.get("test", {})
        rmse = test.get("rmse")
        if rmse is None:
            continue
        if best is None or rmse < best["test"]["rmse"]:
            best = row
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Export test metrics summary from JSONL logs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL paths.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "best_epoch_by_rmse",
                "rmse",
                "acc",
                "track_mae",
                "extreme_f1",
                "spectral_distance",
                "checkpoint_metric",
                "checkpoint_value",
            ]
        )
        for p in args.inputs:
            path = Path(p)
            rows = _load_jsonl(path)
            best = _best_epoch(rows)
            if best is None:
                writer.writerow([str(path), "", "", "", "", "", "", "", ""])
                continue
            test = best.get("test", {})
            writer.writerow(
                [
                    str(path),
                    best.get("epoch", ""),
                    test.get("rmse", ""),
                    test.get("acc", ""),
                    test.get("track_mae", ""),
                    test.get("extreme_f1", ""),
                    test.get("spectral_distance", ""),
                    best.get("checkpoint_metric", ""),
                    best.get("checkpoint_value", ""),
                ]
            )

    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
