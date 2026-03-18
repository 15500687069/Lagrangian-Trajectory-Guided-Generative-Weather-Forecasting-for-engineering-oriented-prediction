from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _better(current: float, best: float, mode: str) -> bool:
    return current > best if mode == "max" else current < best


def _load_records(jsonl_path: Path, section: str, metric: str) -> list[dict]:
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            block = row.get(section, {})
            if not isinstance(block, dict):
                continue
            if metric not in block:
                continue
            if "epoch" not in row:
                continue
            records.append(
                {
                    "epoch": int(row["epoch"]),
                    "metric_value": float(block[metric]),
                    "section": section,
                    "metric": metric,
                    "raw": row,
                }
            )
    return records


def _copy_or_link(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def main() -> None:
    p = argparse.ArgumentParser(description="Select the best checkpoint by metric from JSONL records.")
    p.add_argument("--run-dir", required=True, help="Training output dir containing epoch_XXX.pt and JSONL logs.")
    p.add_argument("--jsonl", default="test_metrics.jsonl", help="JSONL filename inside run-dir.")
    p.add_argument("--section", default="test", help="Metric section key in JSONL, e.g. test/val/train.")
    p.add_argument("--metric", default="track_mae", help="Metric key, e.g. track_mae/rmse/acc.")
    p.add_argument("--mode", choices=["min", "max"], default="min")
    p.add_argument("--output-name", default="best_track_mae.pt", help="Output checkpoint filename.")
    p.add_argument("--summary-name", default="best_track_mae_summary.json", help="Summary JSON filename.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    jsonl_path = run_dir / args.jsonl
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    records = _load_records(jsonl_path=jsonl_path, section=args.section, metric=args.metric)
    if not records:
        raise RuntimeError(
            f"No records found for section={args.section}, metric={args.metric} in {jsonl_path}"
        )

    best = records[0]
    for rec in records[1:]:
        if _better(rec["metric_value"], best["metric_value"], args.mode):
            best = rec

    epoch = int(best["epoch"])
    ckpt_name = f"epoch_{epoch:03d}.pt"
    src_ckpt = run_dir / ckpt_name
    if not src_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint for best epoch not found: {src_ckpt}")

    dst_ckpt = run_dir / args.output_name
    summary = {
        "run_dir": str(run_dir),
        "jsonl": str(jsonl_path),
        "section": args.section,
        "metric": args.metric,
        "mode": args.mode,
        "best_epoch": epoch,
        "best_metric_value": float(best["metric_value"]),
        "source_checkpoint": str(src_ckpt),
        "selected_checkpoint": str(dst_ckpt),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    method = _copy_or_link(src_ckpt, dst_ckpt)
    summary["materialize_method"] = method
    summary_path = run_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Best checkpoint selected.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
