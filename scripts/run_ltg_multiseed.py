from __future__ import annotations

import argparse
import csv
import json
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


def _to_float(value: Any) -> float:
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LTG-Net multi-seed reproducibility and aggregate strict benchmark.")
    parser.add_argument("--base_config", required=True, help="Base config path for each seed run.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="At least 3 seeds recommended.")
    parser.add_argument("--output_root", default="outputs/ltg/multiseed")
    parser.add_argument("--single_checkpoint_name", default="best.pt")
    parser.add_argument("--field_checkpoint_name", default="best_field.pt")
    parser.add_argument("--track_checkpoint_name", default="best_track.pt")
    parser.add_argument(
        "--inference_modes",
        nargs="+",
        default=["field", "track"],
        choices=["single", "field", "track", "fused"],
    )
    parser.add_argument("--fused_encoder_source", default="field", choices=["field", "track"])
    parser.add_argument("--save_fused_checkpoint", action="store_true")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--strict_script", default="scripts/run_ltg_strict_benchmark.py")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
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
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    args = parser.parse_args()

    if len(args.seeds) < 3:
        print("[warn] seeds < 3. For robust reproducibility, use at least 3 seeds.")

    base_config = Path(args.base_config).resolve()
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    strict_script = Path(args.strict_script).resolve()
    if not strict_script.exists():
        raise FileNotFoundError(f"Strict benchmark script not found: {strict_script}")

    merged_base = load_config(base_config)
    base_name = str(merged_base.get("experiment", {}).get("name", "ltg_seed"))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_dir = output_root / f"seed_{int(seed)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_cfg_path = run_dir / f"config_seed_{int(seed)}.yaml"
        seed_cfg = {
            "base_config": str(base_config).replace("\\", "/"),
            "experiment": {
                "seed": int(seed),
                "name": f"{base_name}_seed_{int(seed)}",
                "output_dir": str(run_dir).replace("\\", "/"),
            },
        }
        _write_yaml(seed_cfg_path, seed_cfg)

        if not args.skip_train:
            _run_cmd(
                [
                    args.python_bin,
                    "-m",
                    "ltg_net.cli",
                    "train",
                    "--config",
                    str(seed_cfg_path),
                ],
                cwd=ROOT,
            )

        single_ckpt = run_dir / args.single_checkpoint_name
        field_ckpt = run_dir / args.field_checkpoint_name
        track_ckpt = run_dir / args.track_checkpoint_name

        for mode in args.inference_modes:
            bench_dir = run_dir / f"strict_benchmark_{mode}"
            mode = str(mode)
            if mode == "single":
                if not single_ckpt.exists():
                    raise FileNotFoundError(
                        f"single mode checkpoint missing for seed={seed}: {single_ckpt}"
                    )
            elif mode == "field":
                if not field_ckpt.exists():
                    raise FileNotFoundError(
                        f"field mode checkpoint missing for seed={seed}: {field_ckpt}"
                    )
            elif mode == "track":
                if not track_ckpt.exists():
                    raise FileNotFoundError(
                        f"track mode checkpoint missing for seed={seed}: {track_ckpt}"
                    )
            elif mode == "fused":
                if not field_ckpt.exists() or not track_ckpt.exists():
                    raise FileNotFoundError(
                        f"fused mode checkpoints missing for seed={seed}: {field_ckpt}, {track_ckpt}"
                    )

            if not args.skip_benchmark:
                cmd = [
                    args.python_bin,
                    str(strict_script),
                    "--config",
                    str(seed_cfg_path),
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
                    cmd.extend(["--checkpoint", str(single_ckpt)])
                elif mode == "field":
                    cmd.extend(["--field_checkpoint", str(field_ckpt)])
                elif mode == "track":
                    cmd.extend(["--track_checkpoint", str(track_ckpt)])
                elif mode == "fused":
                    cmd.extend(
                        [
                            "--field_checkpoint",
                            str(field_ckpt),
                            "--track_checkpoint",
                            str(track_ckpt),
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
                "seed": int(seed),
                "mode": mode,
                "config": str(seed_cfg_path),
                "single_checkpoint": str(single_ckpt),
                "field_checkpoint": str(field_ckpt),
                "track_checkpoint": str(track_ckpt),
                "benchmark_dir": str(bench_dir),
                "gate_pass": bool(gate.get("pass", False)),
            }
            for metric in METRICS:
                row[metric] = _to_float(model_row[metric])
            run_rows.append(row)

    runs_csv = output_root / "multiseed_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "seed",
            "mode",
            "gate_pass",
            *METRICS,
            "config",
            "single_checkpoint",
            "field_checkpoint",
            "track_checkpoint",
            "benchmark_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)

    agg: dict[str, Any] = {
        "base_config": str(base_config),
        "seeds": [int(s) for s in args.seeds],
        "modes": [str(m) for m in args.inference_modes],
        "n_runs": len(run_rows),
        "by_mode": {},
    }
    for mode in args.inference_modes:
        mode_rows = [r for r in run_rows if r["mode"] == str(mode)]
        mode_summary: dict[str, Any] = {
            "n_runs": len(mode_rows),
            "gate_pass_rate": float(sum(1 for r in mode_rows if r["gate_pass"]) / max(1, len(mode_rows))),
            "metrics": {},
        }
        for metric in METRICS:
            values = [float(r[metric]) for r in mode_rows]
            if not values:
                mode_summary["metrics"][metric] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
                continue
            mean = sum(values) / max(1, len(values))
            std = (sum((x - mean) ** 2 for x in values) / max(1, len(values))) ** 0.5
            mode_summary["metrics"][metric] = {
                "mean": float(mean),
                "std": float(std),
                "min": float(min(values)),
                "max": float(max(values)),
            }
        agg["by_mode"][str(mode)] = mode_summary

    agg_json = output_root / "multiseed_aggregate.json"
    agg_json.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved run table: {runs_csv}")
    print(f"Saved aggregate json: {agg_json}")
    print("Aggregate (model):")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
