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
    parser.add_argument("--checkpoint_name", default="best.pt")
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

        checkpoint = run_dir / args.checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for seed={seed}: {checkpoint}. "
                "You can change --checkpoint_name."
            )

        bench_dir = run_dir / "strict_benchmark"
        if not args.skip_benchmark:
            cmd = [
                args.python_bin,
                str(strict_script),
                "--config",
                str(seed_cfg_path),
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
            "seed": int(seed),
            "config": str(seed_cfg_path),
            "checkpoint": str(checkpoint),
            "benchmark_dir": str(bench_dir),
            "gate_pass": bool(gate.get("pass", False)),
        }
        for metric in METRICS:
            row[metric] = _to_float(model_row[metric])
        run_rows.append(row)

    runs_csv = output_root / "multiseed_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["seed", "gate_pass", *METRICS, "config", "checkpoint", "benchmark_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)

    agg: dict[str, Any] = {
        "base_config": str(base_config),
        "seeds": [int(s) for s in args.seeds],
        "n_runs": len(run_rows),
        "metrics": {},
        "gate_pass_rate": float(sum(1 for r in run_rows if r["gate_pass"]) / max(1, len(run_rows))),
    }
    for metric in METRICS:
        values = [float(r[metric]) for r in run_rows]
        mean = sum(values) / max(1, len(values))
        std = (sum((x - mean) ** 2 for x in values) / max(1, len(values))) ** 0.5
        agg["metrics"][metric] = {
            "mean": float(mean),
            "std": float(std),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    agg_json = output_root / "multiseed_aggregate.json"
    agg_json.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved run table: {runs_csv}")
    print(f"Saved aggregate json: {agg_json}")
    print("Aggregate (model):")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
