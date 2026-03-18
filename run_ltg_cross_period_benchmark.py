from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltg_net.config import load_config


METRICS = ["rmse", "acc", "track_mae", "extreme_f1", "spectral_distance"]


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _sanitize_label(text: str) -> str:
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    x = x.strip("._-")
    return x or "period"


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=False)


def _read_summary_row(summary_csv: Path, method: str) -> dict[str, Any]:
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method") == method:
                return row
    raise RuntimeError(f"Method '{method}' not found in {summary_csv}")


def _parse_periods(period_args: list[str]) -> list[dict[str, str]]:
    periods: list[dict[str, str]] = []
    for spec in period_args:
        parts = [x.strip() for x in str(spec).split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid --period '{spec}'. Use format: label,YYYY-MM-DD,YYYY-MM-DD")
        label, start, end = parts
        periods.append({"label": label, "test_start": start, "test_end": end})
    return periods


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-period generalization benchmark for LTG-Net.")
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--period",
        action="append",
        required=True,
        help="Repeatable. Format: label,YYYY-MM-DD,YYYY-MM-DD",
    )
    parser.add_argument("--output_root", default="outputs/ltg/cross_period")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--strict_script", default="scripts/run_ltg_strict_benchmark.py")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["model", "persistence", "linear", "climatology"],
        choices=["model", "persistence", "linear", "climatology"],
    )
    parser.add_argument("--leads", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--ci_alpha", type=float, default=0.05)
    parser.add_argument("--reference", default="persistence")
    parser.add_argument("--strict_p_threshold", type=float, default=0.95)
    args = parser.parse_args()

    periods = _parse_periods(args.period)
    base_config = Path(args.base_config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    strict_script = Path(args.strict_script).resolve()
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not strict_script.exists():
        raise FileNotFoundError(f"Strict script not found: {strict_script}")

    merged_base = load_config(base_config)
    base_name = str(merged_base.get("experiment", {}).get("name", "ltg_cross_period"))
    base_split = merged_base.get("data", {}).get("split", {})

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in periods:
        raw_label = str(item["label"])
        label = _sanitize_label(raw_label)
        run_dir = output_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / f"config_{label}.yaml"

        cfg_override = {
            "base_config": str(base_config).replace("\\", "/"),
            "experiment": {
                "name": f"{base_name}_{label}",
                "output_dir": str(run_dir).replace("\\", "/"),
            },
            "data": {
                "split": {
                    "test_start": str(item["test_start"]),
                    "test_end": str(item["test_end"]),
                }
            },
        }
        _write_yaml(cfg_path, cfg_override)

        bench_dir = run_dir / "strict_benchmark"
        cmd = [
            args.python_bin,
            str(strict_script),
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(checkpoint),
            "--split",
            args.split,
            "--methods",
            *[str(m) for m in args.methods],
            "--bootstrap",
            str(int(args.bootstrap)),
            "--ci_alpha",
            str(float(args.ci_alpha)),
            "--reference",
            str(args.reference),
            "--strict_p_threshold",
            str(float(args.strict_p_threshold)),
            "--output_dir",
            str(bench_dir),
        ]
        if args.leads:
            cmd.extend(["--leads", *[str(int(x)) for x in args.leads]])
        _run_cmd(cmd, cwd=ROOT)

        summary_csv = bench_dir / "strict_summary.csv"
        result_json = bench_dir / "strict_results.json"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing strict summary: {summary_csv}")
        if not result_json.exists():
            raise FileNotFoundError(f"Missing strict results json: {result_json}")

        model_row = _read_summary_row(summary_csv, method="model")
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        gate = payload.get("gate", {})
        row = {
            "label": raw_label,
            "test_start": str(item["test_start"]),
            "test_end": str(item["test_end"]),
            "gate_pass": bool(gate.get("pass", False)),
            "config": str(cfg_path),
            "benchmark_dir": str(bench_dir),
        }
        for metric in METRICS:
            row[metric] = float(model_row[metric])
        rows.append(row)

    # Use first period as reference to show relative drift.
    if rows:
        ref = rows[0]
        for row in rows:
            row["delta_rmse_vs_first"] = float(row["rmse"] - ref["rmse"])
            row["delta_acc_vs_first"] = float(row["acc"] - ref["acc"])
            row["delta_track_mae_vs_first"] = float(row["track_mae"] - ref["track_mae"])
            row["delta_extreme_f1_vs_first"] = float(row["extreme_f1"] - ref["extreme_f1"])
            row["delta_spectral_distance_vs_first"] = float(
                row["spectral_distance"] - ref["spectral_distance"]
            )

    summary_csv = output_root / "cross_period_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "label",
            "test_start",
            "test_end",
            "gate_pass",
            *METRICS,
            "delta_rmse_vs_first",
            "delta_acc_vs_first",
            "delta_track_mae_vs_first",
            "delta_extreme_f1_vs_first",
            "delta_spectral_distance_vs_first",
            "config",
            "benchmark_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    aggregate = {
        "base_config": str(base_config),
        "checkpoint": str(checkpoint),
        "base_test_split": {
            "test_start": str(base_split.get("test_start", "")),
            "test_end": str(base_split.get("test_end", "")),
        },
        "periods": rows,
        "pass_rate": float(sum(1 for r in rows if r["gate_pass"]) / max(1, len(rows))),
    }
    summary_json = output_root / "cross_period_summary.json"
    summary_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved cross-period CSV: {summary_csv}")
    print(f"Saved cross-period JSON: {summary_json}")
    print("Cross-period model summary:")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
