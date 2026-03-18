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
    parser.add_argument("--checkpoint", default="", help="Used by inference_mode=single.")
    parser.add_argument("--field_checkpoint", default="", help="Used by inference_mode=field/fused.")
    parser.add_argument("--track_checkpoint", default="", help="Used by inference_mode=track/fused.")
    parser.add_argument(
        "--inference_modes",
        nargs="+",
        default=["field", "track"],
        choices=["single", "field", "track", "fused"],
    )
    parser.add_argument("--fused_encoder_source", default="field", choices=["field", "track"])
    parser.add_argument("--save_fused_checkpoint", action="store_true")
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
    parser.add_argument("--gate_profile", default="legacy", choices=["legacy", "by_mode", "all"])
    parser.add_argument("--gate_metrics", default="")
    parser.add_argument("--calibrate_on_split", default="none", choices=["none", "train", "val", "test"])
    parser.add_argument("--calibration_json", default="")
    parser.add_argument("--calib_alpha_field_grid", default="1.0,0.95,0.9")
    parser.add_argument("--calib_alpha_traj_grid", default="1.0,0.9,0.8")
    parser.add_argument("--calib_beta_high_grid", default="0.0,0.12")
    parser.add_argument("--calib_k_ratio_grid", default="0.55,0.65")
    parser.add_argument("--calib_prefix_leads_grid", default="0,2")
    parser.add_argument("--calib_alpha_field_prefix_grid", default="1.0,0.6")
    parser.add_argument("--calib_alpha_traj_prefix_grid", default="1.0,0.7")
    parser.add_argument("--calib_max_batches", type=int, default=0)
    parser.add_argument("--calib_max_candidates", type=int, default=120)
    parser.add_argument("--calib_rmse_tol", type=float, default=0.005)
    parser.add_argument("--calib_acc_tol", type=float, default=0.005)
    parser.add_argument("--calib_f1_tol", type=float, default=0.010)
    parser.add_argument("--calib_track_tol", type=float, default=0.300)
    parser.add_argument("--calib_spectral_tol", type=float, default=0.030)
    args = parser.parse_args()

    periods = _parse_periods(args.period)
    base_config = Path(args.base_config).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    field_checkpoint = Path(args.field_checkpoint).resolve() if args.field_checkpoint else None
    track_checkpoint = Path(args.track_checkpoint).resolve() if args.track_checkpoint else None
    strict_script = Path(args.strict_script).resolve()
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    for mode in args.inference_modes:
        mode = str(mode)
        if mode == "single":
            if checkpoint is None or (not checkpoint.exists()):
                raise FileNotFoundError("single mode requires valid --checkpoint")
        elif mode == "field":
            if field_checkpoint is None or (not field_checkpoint.exists()):
                raise FileNotFoundError("field mode requires valid --field_checkpoint")
        elif mode == "track":
            if track_checkpoint is None or (not track_checkpoint.exists()):
                raise FileNotFoundError("track mode requires valid --track_checkpoint")
        elif mode == "fused":
            if (
                field_checkpoint is None
                or track_checkpoint is None
                or (not field_checkpoint.exists())
                or (not track_checkpoint.exists())
            ):
                raise FileNotFoundError("fused mode requires valid --field_checkpoint and --track_checkpoint")
    if not strict_script.exists():
        raise FileNotFoundError(f"Strict script not found: {strict_script}")

    merged_base = load_config(base_config)
    base_name = str(merged_base.get("experiment", {}).get("name", "ltg_cross_period"))
    base_split = merged_base.get("data", {}).get("split", {})

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for mode in args.inference_modes:
        mode = str(mode)
        for item in periods:
            raw_label = str(item["label"])
            label = _sanitize_label(raw_label)
            run_dir = output_root / mode / label
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = run_dir / f"config_{label}.yaml"

            cfg_override = {
                "base_config": str(base_config).replace("\\", "/"),
                "experiment": {
                    "name": f"{base_name}_{mode}_{label}",
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
                "--inference_mode",
                mode,
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
                "--gate_profile",
                str(args.gate_profile),
                "--calibrate_on_split",
                str(args.calibrate_on_split),
                "--calib_alpha_field_grid",
                str(args.calib_alpha_field_grid),
                "--calib_alpha_traj_grid",
                str(args.calib_alpha_traj_grid),
                "--calib_beta_high_grid",
                str(args.calib_beta_high_grid),
                "--calib_k_ratio_grid",
                str(args.calib_k_ratio_grid),
                "--calib_prefix_leads_grid",
                str(args.calib_prefix_leads_grid),
                "--calib_alpha_field_prefix_grid",
                str(args.calib_alpha_field_prefix_grid),
                "--calib_alpha_traj_prefix_grid",
                str(args.calib_alpha_traj_prefix_grid),
                "--calib_max_batches",
                str(int(args.calib_max_batches)),
                "--calib_max_candidates",
                str(int(args.calib_max_candidates)),
                "--calib_rmse_tol",
                str(float(args.calib_rmse_tol)),
                "--calib_acc_tol",
                str(float(args.calib_acc_tol)),
                "--calib_f1_tol",
                str(float(args.calib_f1_tol)),
                "--calib_track_tol",
                str(float(args.calib_track_tol)),
                "--calib_spectral_tol",
                str(float(args.calib_spectral_tol)),
                "--output_dir",
                str(bench_dir),
            ]
            if str(args.gate_metrics):
                cmd.extend(["--gate_metrics", str(args.gate_metrics)])
            if str(args.calibration_json):
                cmd.extend(["--calibration_json", str(args.calibration_json)])
            if mode == "single":
                assert checkpoint is not None
                cmd.extend(["--checkpoint", str(checkpoint)])
            elif mode == "field":
                assert field_checkpoint is not None
                cmd.extend(["--field_checkpoint", str(field_checkpoint)])
            elif mode == "track":
                assert track_checkpoint is not None
                cmd.extend(["--track_checkpoint", str(track_checkpoint)])
            elif mode == "fused":
                assert field_checkpoint is not None and track_checkpoint is not None
                cmd.extend(
                    [
                        "--field_checkpoint",
                        str(field_checkpoint),
                        "--track_checkpoint",
                        str(track_checkpoint),
                        "--fused_encoder_source",
                        str(args.fused_encoder_source),
                    ]
                )
                if args.save_fused_checkpoint:
                    cmd.extend(["--save_fused_checkpoint", str(run_dir / "fused_field_track.pt")])
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
                "mode": mode,
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

    # Per-mode: use first period as reference to show drift.
    for mode in args.inference_modes:
        mode_rows = [r for r in rows if r["mode"] == str(mode)]
        if not mode_rows:
            continue
        ref = mode_rows[0]
        for row in mode_rows:
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
            "mode",
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
        "checkpoint": str(checkpoint) if checkpoint else "",
        "field_checkpoint": str(field_checkpoint) if field_checkpoint else "",
        "track_checkpoint": str(track_checkpoint) if track_checkpoint else "",
        "inference_modes": [str(m) for m in args.inference_modes],
        "base_test_split": {
            "test_start": str(base_split.get("test_start", "")),
            "test_end": str(base_split.get("test_end", "")),
        },
        "periods": rows,
        "pass_rate": float(sum(1 for r in rows if r["gate_pass"]) / max(1, len(rows))),
        "pass_rate_by_mode": {
            str(mode): float(
                sum(1 for r in rows if r["mode"] == str(mode) and r["gate_pass"])
                / max(1, sum(1 for r in rows if r["mode"] == str(mode)))
            )
            for mode in args.inference_modes
        },
    }
    summary_json = output_root / "cross_period_summary.json"
    summary_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved cross-period CSV: {summary_csv}")
    print(f"Saved cross-period JSON: {summary_json}")
    print("Cross-period model summary:")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
