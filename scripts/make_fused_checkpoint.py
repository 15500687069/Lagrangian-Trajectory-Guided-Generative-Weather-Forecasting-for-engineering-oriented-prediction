from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltg_net.config import load_config
from ltg_net.data.datamodule import build_dataloaders
from ltg_net.models import LTGNet

TRAJ_PREFIXES = ("traj_predictor.",)
ENCODER_PREFIX = "encoder."


def _resolve_device(cfg: dict[str, Any], requested: str | None) -> torch.device:
    if requested:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    from_cfg = str(cfg.get("experiment", {}).get("device", "cuda"))
    if from_cfg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_state_dict(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _build_model(cfg: dict[str, Any], device: torch.device) -> LTGNet:
    dataloaders = build_dataloaders(cfg)
    sample = next(iter(dataloaders["train"]))
    in_channels = int(sample["x_hist"].shape[2])
    return LTGNet(cfg=cfg, in_channels=in_channels).to(device)


def _compose_fused_state_dict(
    base_model: LTGNet,
    field_state: dict[str, torch.Tensor],
    track_state: dict[str, torch.Tensor],
    fused_encoder_source: str,
) -> dict[str, torch.Tensor]:
    fused = dict(base_model.state_dict())

    # 1) Baseline: copy as much as possible from field checkpoint.
    for k in list(fused.keys()):
        if k in field_state:
            fused[k] = field_state[k]

    # 2) Optional: overwrite encoder from track checkpoint.
    if fused_encoder_source == "track":
        for k in list(fused.keys()):
            if k.startswith(ENCODER_PREFIX) and k in track_state:
                fused[k] = track_state[k]

    # 3) Always overwrite trajectory branch from track checkpoint.
    for k in list(fused.keys()):
        if k.startswith(TRAJ_PREFIXES) and k in track_state:
            fused[k] = track_state[k]

    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a fused LTG checkpoint from field+track checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--field_checkpoint", required=True)
    parser.add_argument("--track_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fused_encoder_source", default="field", choices=["field", "track"])
    parser.add_argument("--device", default="", choices=["", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(cfg, requested=(args.device or None))
    print(f"Device: {device.type}")

    model = _build_model(cfg=cfg, device=device)
    field_state = _load_state_dict(args.field_checkpoint, device=device)
    track_state = _load_state_dict(args.track_checkpoint, device=device)
    fused_state = _compose_fused_state_dict(
        base_model=model,
        field_state=field_state,
        track_state=track_state,
        fused_encoder_source=str(args.fused_encoder_source),
    )

    # Validate fused state can be loaded by current architecture.
    model.load_state_dict(fused_state, strict=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": fused_state,
        "meta": {
            "type": "fused_field_track",
            "field_checkpoint": str(args.field_checkpoint),
            "track_checkpoint": str(args.track_checkpoint),
            "fused_encoder_source": str(args.fused_encoder_source),
            "config": str(args.config),
        },
    }
    torch.save(payload, out_path)
    print(f"Saved fused checkpoint: {out_path}")


if __name__ == "__main__":
    main()
