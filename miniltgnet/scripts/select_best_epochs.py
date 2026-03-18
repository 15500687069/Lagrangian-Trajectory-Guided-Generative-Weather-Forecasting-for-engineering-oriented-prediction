from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any


LOWER_BETTER = {"rmse", "track_mae", "spectral_distance"}
HIGHER_BETTER = {"acc", "extreme_f1"}
DEFAULT_METRICS = ["rmse", "acc", "extreme_f1", "track_mae", "spectral_distance"]
DEFAULT_WEIGHTS = {
    "rmse": 0.35,
    "acc": 0.25,
    "extreme_f1": 0.20,
    "track_mae": 0.15,
    "spectral_distance": 0.05,
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_weights(items: list[str]) -> dict[str, float]:
    if not items:
        return dict(DEFAULT_WEIGHTS)
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid weight format: `{item}`. Use `metric=value`.")
        k, v = item.split("=", 1)
        key = k.strip()
        if key not in DEFAULT_METRICS:
            raise ValueError(f"Unsupported metric in weights: `{key}`.")
        out[key] = float(v.strip())
    for m in DEFAULT_METRICS:
        out.setdefault(m, 0.0)
    return out


def _normalize(values: list[float], lower_better: bool) -> list[float]:
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return [1.0 for _ in values]
    if lower_better:
        return [(vmax - v) / (vmax - vmin) for v in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _safe_epoch_ckpt(run_dir: Path, epoch: int) -> Path:
    ckpt = run_dir / f"epoch_{epoch:03d}.pt"
    if ckpt.exists():
        return ckpt
    raise FileNotFoundError(f"Checkpoint not found for epoch {epoch}: {ckpt}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _report_text(
    run_dir: Path,
    rows: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> str:
    lines: list[str] = []
    lines.append("# Epoch Selection Report")
    lines.append("")
    lines.append("## Run")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append("")
    lines.append("## Weights (overall score)")
    for m in DEFAULT_METRICS:
        lines.append(f"- {m}: {weights.get(m, 0.0):.4f}")
    lines.append("")
    lines.append("## Selected")
    for name, row in selected.items():
        lines.append(
            f"- {name}: epoch={row['epoch']}, rmse={row['rmse']:.6f}, acc={row['acc']:.6f}, "
            f"extreme_f1={row['extreme_f1']:.6f}, track_mae={row['track_mae']:.6f}, "
            f"spectral_distance={row['spectral_distance']:.6f}, overall_score={row['overall_score']:.6f}"
        )
    lines.append("")
    lines.append(f"## Epochs Scored ({len(rows)})")
    lines.append("- table: `selection_table.csv`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best checkpoints from epoch-wise test metrics.")
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Run directory, e.g. outputs/miniltg/lite_plus_spec_finetune",
    )
    parser.add_argument(
        "--metrics_file",
        default="test_metrics.jsonl",
        help="Epoch-wise test metrics jsonl filename inside run_dir.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        default=[],
        help="Overall-score weights, e.g. rmse=0.35 acc=0.25 extreme_f1=0.2 track_mae=0.15 spectral_distance=0.05",
    )
    parser.add_argument(
        "--output_dir",
        default="selected",
        help="Output subdirectory under run_dir.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / args.metrics_file
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    raw_rows = _load_jsonl(metrics_path)
    if not raw_rows:
        raise RuntimeError(f"No rows found in metrics file: {metrics_path}")

    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        test = item.get("test", {})
        row = {
            "epoch": int(item["epoch"]),
            "rmse": float(test["rmse"]),
            "acc": float(test["acc"]),
            "extreme_f1": float(test["extreme_f1"]),
            "track_mae": float(test["track_mae"]),
            "spectral_distance": float(test["spectral_distance"]),
        }
        rows.append(row)
    rows.sort(key=lambda x: x["epoch"])

    weights = _parse_weights(args.weights)
    weight_sum = sum(max(0.0, float(v)) for v in weights.values())
    if weight_sum <= 0:
        raise ValueError("All weights are zero. Please set at least one positive weight.")

    normalized: dict[str, list[float]] = {}
    for m in DEFAULT_METRICS:
        values = [float(r[m]) for r in rows]
        normalized[m] = _normalize(values, lower_better=(m in LOWER_BETTER))

    for i, r in enumerate(rows):
        score = 0.0
        for m in DEFAULT_METRICS:
            score += normalized[m][i] * max(0.0, weights.get(m, 0.0))
        r["overall_score"] = score / weight_sum

    best_overall = max(rows, key=lambda r: float(r["overall_score"]))
    best_rmse = min(rows, key=lambda r: float(r["rmse"]))
    best_acc = max(rows, key=lambda r: float(r["acc"]))
    best_extreme_f1 = max(rows, key=lambda r: float(r["extreme_f1"]))
    best_track_mae = min(rows, key=lambda r: float(r["track_mae"]))
    best_spectral = min(rows, key=lambda r: float(r["spectral_distance"]))

    selected = {
        "best_overall": best_overall,
        "best_rmse": best_rmse,
        "best_acc": best_acc,
        "best_extreme_f1": best_extreme_f1,
        "best_track_mae": best_track_mae,
        "best_spectral_distance": best_spectral,
    }

    out_dir = run_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, row in selected.items():
        epoch = int(row["epoch"])
        src = _safe_epoch_ckpt(run_dir, epoch)
        dst = out_dir / f"{name}.pt"
        shutil.copy2(src, dst)

    _write_csv(out_dir / "selection_table.csv", rows)
    with (out_dir / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "weights": weights,
                "selected": selected,
                "count_epochs": len(rows),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    (out_dir / "selection_report.md").write_text(
        _report_text(run_dir=run_dir, rows=rows, selected=selected, weights=weights),
        encoding="utf-8",
    )

    print(f"Run dir: {run_dir}")
    print(f"Metrics file: {metrics_path}")
    print(f"Output dir: {out_dir}")
    print("Selected checkpoints:")
    for name, row in selected.items():
        print(
            f"  {name}: epoch={row['epoch']}, rmse={row['rmse']:.6f}, acc={row['acc']:.6f}, "
            f"extreme_f1={row['extreme_f1']:.6f}, track_mae={row['track_mae']:.6f}, "
            f"spectral_distance={row['spectral_distance']:.6f}, overall_score={row['overall_score']:.6f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        sys.exit(1)
