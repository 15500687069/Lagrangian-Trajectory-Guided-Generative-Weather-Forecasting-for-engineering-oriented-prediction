from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch


def _load_ckpt(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a dict: {path}")
    return ckpt


def _pick_state(ckpt: dict[str, Any], use_ema_if_available: bool) -> dict[str, torch.Tensor]:
    if use_ema_if_available and ckpt.get("model_ema") is not None:
        return ckpt["model_ema"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    raise KeyError("Checkpoint missing `model` state.")


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        out[k] = v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
    return out


def _matches_prefix(key: str, prefixes: list[str]) -> bool:
    return any(key.startswith(p) for p in prefixes)


def _blend_tensor(base: torch.Tensor, donor: torch.Tensor, alpha: float) -> torch.Tensor:
    if not isinstance(base, torch.Tensor) or not isinstance(donor, torch.Tensor):
        return donor
    if base.shape != donor.shape:
        return donor
    if torch.is_floating_point(base) and torch.is_floating_point(donor):
        b = base.to(torch.float32)
        d = donor.to(torch.float32)
        out = (1.0 - alpha) * b + alpha * d
        return out.to(dtype=base.dtype)
    return donor


def _apply_module_op(
    state: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor] | None,
    prefixes: list[str],
    mode: str,
    alpha: float,
) -> tuple[dict[str, torch.Tensor], int]:
    if donor is None or mode == "none":
        return state, 0
    updated = 0
    for k in list(state.keys()):
        if not _matches_prefix(k, prefixes):
            continue
        if k not in donor:
            continue
        if mode == "swap":
            state[k] = donor[k].detach().clone()
            updated += 1
        elif mode == "blend":
            state[k] = _blend_tensor(state[k], donor[k], alpha=float(alpha))
            updated += 1
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    return state, updated


def _normalize_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in {"none", "swap", "blend"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose a checkpoint by metric-specific module stitching.")
    parser.add_argument("--base_checkpoint", required=True, help="Overall/balanced best checkpoint.")
    parser.add_argument("--track_checkpoint", default="", help="Track-best checkpoint (for traj modules).")
    parser.add_argument("--spectral_checkpoint", default="", help="Spectral-best checkpoint (for field/hf modules).")
    parser.add_argument("--acc_checkpoint", default="", help="ACC-best checkpoint (for encoder, optional).")
    parser.add_argument("--output", required=True, help="Output stitched checkpoint path.")
    parser.add_argument("--use_ema_if_available", action="store_true")

    parser.add_argument("--traj_mode", default="swap", choices=["none", "swap", "blend"])
    parser.add_argument("--traj_alpha", type=float, default=0.60)
    parser.add_argument("--field_mode", default="blend", choices=["none", "swap", "blend"])
    parser.add_argument("--field_alpha", type=float, default=0.25)
    parser.add_argument("--hf_mode", default="swap", choices=["none", "swap", "blend"])
    parser.add_argument("--hf_alpha", type=float, default=0.50)
    parser.add_argument("--encoder_mode", default="none", choices=["none", "swap", "blend"])
    parser.add_argument("--encoder_alpha", type=float, default=0.10)
    args = parser.parse_args()

    device = torch.device("cpu")
    base_ckpt_path = Path(args.base_checkpoint)
    track_ckpt_path = Path(args.track_checkpoint) if args.track_checkpoint else None
    spectral_ckpt_path = Path(args.spectral_checkpoint) if args.spectral_checkpoint else None
    acc_ckpt_path = Path(args.acc_checkpoint) if args.acc_checkpoint else None

    for p in [base_ckpt_path, track_ckpt_path, spectral_ckpt_path, acc_ckpt_path]:
        if p is not None and not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    base_ckpt = _load_ckpt(base_ckpt_path, device)
    base_state = _pick_state(base_ckpt, use_ema_if_available=bool(args.use_ema_if_available))
    state = _clone_state(base_state)

    track_state = None
    spectral_state = None
    acc_state = None
    if track_ckpt_path is not None:
        track_state = _pick_state(
            _load_ckpt(track_ckpt_path, device), use_ema_if_available=bool(args.use_ema_if_available)
        )
    if spectral_ckpt_path is not None:
        spectral_state = _pick_state(
            _load_ckpt(spectral_ckpt_path, device), use_ema_if_available=bool(args.use_ema_if_available)
        )
    if acc_ckpt_path is not None:
        acc_state = _pick_state(_load_ckpt(acc_ckpt_path, device), use_ema_if_available=bool(args.use_ema_if_available))

    traj_mode = _normalize_mode(args.traj_mode)
    field_mode = _normalize_mode(args.field_mode)
    hf_mode = _normalize_mode(args.hf_mode)
    encoder_mode = _normalize_mode(args.encoder_mode)

    if traj_mode != "none" and track_state is None:
        raise ValueError("traj_mode is enabled but --track_checkpoint is not provided.")
    if field_mode != "none" and spectral_state is None:
        raise ValueError("field_mode is enabled but --spectral_checkpoint is not provided.")
    if hf_mode != "none" and spectral_state is None:
        raise ValueError("hf_mode is enabled but --spectral_checkpoint is not provided.")
    if encoder_mode != "none" and acc_state is None:
        raise ValueError("encoder_mode is enabled but --acc_checkpoint is not provided.")

    summary: dict[str, Any] = {
        "base_checkpoint": str(base_ckpt_path),
        "track_checkpoint": str(track_ckpt_path) if track_ckpt_path is not None else "",
        "spectral_checkpoint": str(spectral_ckpt_path) if spectral_ckpt_path is not None else "",
        "acc_checkpoint": str(acc_ckpt_path) if acc_ckpt_path is not None else "",
        "ops": [],
        "use_ema_if_available": bool(args.use_ema_if_available),
    }

    state, n = _apply_module_op(
        state=state,
        donor=track_state,
        prefixes=["traj.", "traj_refiner."],
        mode=traj_mode,
        alpha=float(args.traj_alpha),
    )
    summary["ops"].append(
        {"module": "traj", "mode": traj_mode, "alpha": float(args.traj_alpha), "updated_keys": int(n)}
    )

    state, n = _apply_module_op(
        state=state,
        donor=spectral_state,
        prefixes=["field_step."],
        mode=field_mode,
        alpha=float(args.field_alpha),
    )
    summary["ops"].append(
        {"module": "field_step", "mode": field_mode, "alpha": float(args.field_alpha), "updated_keys": int(n)}
    )

    state, n = _apply_module_op(
        state=state,
        donor=spectral_state,
        prefixes=["hf_refiner."],
        mode=hf_mode,
        alpha=float(args.hf_alpha),
    )
    summary["ops"].append({"module": "hf_refiner", "mode": hf_mode, "alpha": float(args.hf_alpha), "updated_keys": int(n)})

    state, n = _apply_module_op(
        state=state,
        donor=acc_state,
        prefixes=["encoder."],
        mode=encoder_mode,
        alpha=float(args.encoder_alpha),
    )
    summary["ops"].append(
        {"module": "encoder", "mode": encoder_mode, "alpha": float(args.encoder_alpha), "updated_keys": int(n)}
    )

    out_ckpt = copy.deepcopy(base_ckpt)
    out_ckpt["model"] = _clone_state(state)
    if out_ckpt.get("model_ema") is not None:
        out_ckpt["model_ema"] = _clone_state(state)
    out_ckpt["metric_stitch"] = summary

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out_path)
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved stitched checkpoint: {out_path}")
    print("Stitch summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        sys.exit(1)
