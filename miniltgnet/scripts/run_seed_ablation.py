from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniltgnet.cli import evaluate, prepare, train
from miniltgnet.config import load_config


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniLTGNet seed sweep and ablation experiments.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "miniltgnet/configs/lite_plus.yaml",
            "miniltgnet/configs/ablation_no_extreme.yaml",
            "miniltgnet/configs/ablation_low_traj.yaml",
            "miniltgnet/configs/ablation_no_ema.yaml",
        ],
        help="Config list to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 2025, 7], help="Seed list.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Evaluate split.")
    parser.add_argument("--skip_prepare", action="store_true", help="Skip prepare step.")
    parser.add_argument(
        "--output_csv",
        default="outputs/miniltg/seed_ablation/summary.csv",
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--output_json",
        default="outputs/miniltg/seed_ablation/summary.json",
        help="Summary JSON output path.",
    )
    args = parser.parse_args()

    work_dir = Path("outputs/miniltg/seed_ablation")
    cfg_dir = work_dir / "_generated_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    prepared_once: set[str] = set()
    rows: list[dict] = []
    for cfg_path in args.configs:
        base_cfg = load_config(cfg_path)
        exp_base = str(base_cfg["experiment"]["name"])
        for seed in args.seeds:
            run_cfg = copy.deepcopy(base_cfg)
            run_cfg["experiment"]["seed"] = int(seed)
            run_cfg["experiment"]["resume_checkpoint"] = None
            run_cfg["experiment"]["name"] = f"{exp_base}_s{seed}"
            out_dir = work_dir / _slug(exp_base) / f"seed_{seed}"
            run_cfg["experiment"]["output_dir"] = out_dir.as_posix()

            run_cfg_path = cfg_dir / f"{_slug(exp_base)}_seed{seed}.yaml"
            _write_yaml(run_cfg_path, run_cfg)

            cfg_key = str(Path(cfg_path).resolve())
            if not args.skip_prepare and cfg_key not in prepared_once:
                print(f"[prepare] {cfg_path}")
                prepare(str(run_cfg_path))
                prepared_once.add(cfg_key)

            print(f"[train] config={cfg_path} seed={seed}")
            train(str(run_cfg_path))

            best_ckpt = out_dir / "best.pt"
            if not best_ckpt.exists():
                raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")
            print(f"[evaluate] {best_ckpt}")
            metrics = evaluate(str(run_cfg_path), str(best_ckpt), split=args.split)

            row = {
                "config": cfg_path,
                "experiment": exp_base,
                "seed": seed,
                "run_config": str(run_cfg_path),
                "checkpoint": str(best_ckpt),
                **metrics,
            }
            rows.append(row)

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        headers = list(rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Saved CSV summary: {out_csv}")
    print(f"Saved JSON summary: {out_json}")


if __name__ == "__main__":
    main()
