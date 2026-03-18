from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltg_net.cli import train
from ltg_net.config import load_config


def _set_nested(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LTG-Net ablation studies")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default="outputs/ablation")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = {
        "baseline": {},
        "no_adv": {"loss.lambda_adv": 0.0},
        "no_spec": {"loss.lambda_spec": 0.0},
        "no_phys": {"loss.lambda_phys": 0.0},
        "no_traj": {"loss.lambda_traj": 0.0},
        "no_event_mod": {"model.event_modulation.enabled": False},
    }

    for case_name, edits in cases.items():
        cfg = load_config(args.config)
        for key, value in edits.items():
            _set_nested(cfg, key, value)
        cfg["experiment"]["name"] = f"{base_cfg['experiment']['name']}_{case_name}"
        cfg["experiment"]["output_dir"] = str(outdir / case_name)
        cfg_path = outdir / f"{case_name}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
        train(config=str(cfg_path))


if __name__ == "__main__":
    main()
