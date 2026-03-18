from __future__ import annotations

import argparse
import copy
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


def _normalize_weights(weights: list[float], n: int) -> list[float]:
    if not weights:
        return [1.0 / n for _ in range(n)]
    if len(weights) != n:
        raise ValueError(f"weights length ({len(weights)}) must equal checkpoints length ({n}).")
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("Sum of weights must be positive.")
    return [float(w) / s for w in weights]


def _weighted_average_states(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    keys = list(states[0].keys())
    for st in states[1:]:
        if list(st.keys()) != keys:
            raise ValueError("State dict keys mismatch across checkpoints.")
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        ref = states[0][k]
        if not isinstance(ref, torch.Tensor):
            out[k] = copy.deepcopy(ref)
            continue
        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for st, w in zip(states, weights):
                v = st[k].to(dtype=torch.float32)
                acc.add_(v, alpha=float(w))
            out[k] = acc.to(dtype=ref.dtype)
        else:
            out[k] = ref.clone()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a checkpoint soup from multiple checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to average.")
    parser.add_argument("--weights", nargs="*", type=float, default=[], help="Optional weights.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--use_ema_if_available", action="store_true", help="Average EMA weights when available.")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt_paths = [Path(p) for p in args.checkpoints]
    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    weights = _normalize_weights(args.weights, len(ckpt_paths))
    ckpts = [_load_ckpt(p, device) for p in ckpt_paths]

    state_key = "model_ema" if args.use_ema_if_available and all(
        ("model_ema" in c and c["model_ema"] is not None) for c in ckpts
    ) else "model"
    if not all(state_key in c and isinstance(c[state_key], dict) for c in ckpts):
        raise ValueError(f"All checkpoints must have state dict key `{state_key}`.")

    states = [c[state_key] for c in ckpts]
    soup_state = _weighted_average_states(states, weights)

    out_ckpt = copy.deepcopy(ckpts[-1])
    out_ckpt["model"] = soup_state
    if "model_ema" in out_ckpt and out_ckpt["model_ema"] is not None:
        out_ckpt["model_ema"] = soup_state
    out_ckpt["soup"] = {
        "checkpoints": [str(p) for p in ckpt_paths],
        "weights": weights,
        "state_key": state_key,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out_path)

    print(f"Saved soup checkpoint: {out_path}")
    print(f"state_key: {state_key}")
    for p, w in zip(ckpt_paths, weights):
        print(f"  {p}  weight={w:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        sys.exit(1)
