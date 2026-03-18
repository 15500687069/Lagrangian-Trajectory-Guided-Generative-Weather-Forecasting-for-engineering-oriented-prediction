from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniltgnet.config import load_config
from miniltgnet.data import build_dataloaders
from miniltgnet.metrics import acc, extreme_f1, rmse, spectral_distance, track_mae
from miniltgnet.model import build_model
from miniltgnet.trainer import move_batch_to_device


def _resolve_device(cfg: dict[str, Any]) -> torch.device:
    requested = cfg["experiment"].get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if deterministic and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_ckpt(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not dict: {path}")
    return ckpt


def _select_model_state(ckpt: dict[str, Any], use_ema: bool) -> dict[str, torch.Tensor]:
    if use_ema and "model_ema" in ckpt and ckpt["model_ema"] is not None:
        return ckpt["model_ema"]
    if "model" in ckpt:
        return ckpt["model"]
    # raw state dict fallback
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore[return-value]
    raise ValueError("Cannot find model state in checkpoint.")


def _build_model(cfg: dict[str, Any], dls: dict[str, torch.utils.data.DataLoader], device: torch.device) -> torch.nn.Module:
    sample = next(iter(dls["train"]))
    in_channels = int(sample["x_hist"].shape[2])
    return build_model(cfg, in_channels=in_channels).to(device)


def _persistence(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    horizon = int(batch["y_future"].shape[1])
    last_field = batch["x_hist"][:, -1].float()
    last_traj = batch["traj_hist"][:, -1].float()
    field_pred = last_field.unsqueeze(1).repeat(1, horizon, 1, 1, 1)
    traj_pred = last_traj.unsqueeze(1).repeat(1, horizon, 1, 1)
    return field_pred, traj_pred


def _blend_traj(model_traj: torch.Tensor, pers_traj: torch.Tensor, alpha_model: float) -> torch.Tensor:
    if alpha_model >= 1.0:
        return model_traj
    a = float(max(0.0, min(1.0, alpha_model)))
    lat = a * model_traj[..., 0] + (1.0 - a) * pers_traj[..., 0]
    dlon = (model_traj[..., 1] - pers_traj[..., 1] + 180.0) % 360.0 - 180.0
    lon = (pers_traj[..., 1] + a * dlon) % 360.0
    return torch.stack([lat, lon], dim=-1)


def _fft_high_blend(model_field: torch.Tensor, pers_field: torch.Tensor, beta: float, k_ratio: float) -> torch.Tensor:
    # model_field/pers_field: [B,T,C,H,W]
    if beta <= 0.0:
        return model_field
    b = float(max(0.0, min(1.0, beta)))
    kr = float(max(0.05, min(0.95, k_ratio)))

    pf = torch.fft.rfft2(model_field, dim=(-2, -1), norm="ortho")
    pp = torch.fft.rfft2(pers_field, dim=(-2, -1), norm="ortho")
    h = model_field.shape[-2]
    w2 = pf.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=model_field.device),
        torch.arange(w2, device=model_field.device),
        indexing="ij",
    )
    ky = torch.minimum(yy, h - yy).float() / max(1.0, float(h // 2))
    kx = xx.float() / max(1.0, float(w2 - 1))
    radius = torch.sqrt(kx.square() + ky.square())
    high = (radius >= kr).to(pf.real.dtype)[None, None, None, :, :]
    # Keep low frequency from model; blend high frequency towards persistence.
    p_out = pf + b * high * (pp - pf)
    out = torch.fft.irfft2(p_out, s=(model_field.shape[-2], model_field.shape[-1]), dim=(-2, -1), norm="ortho")
    return out


def _apply_variant_state(
    base_state: dict[str, torch.Tensor],
    aux_state: dict[str, torch.Tensor],
    variant: str,
    field_interp_w: float,
) -> dict[str, torch.Tensor]:
    state = {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in base_state.items()}

    if variant in {"traj_splice", "combo_f20", "combo_f35"}:
        for k in state.keys():
            if k.startswith("traj.") and k in aux_state:
                state[k] = aux_state[k].clone()

    if variant == "field_interp_f20":
        w = float(field_interp_w)
        for k in state.keys():
            if k.startswith("field_step.") and k in aux_state and torch.is_floating_point(state[k]):
                state[k] = (1.0 - w) * state[k] + w * aux_state[k].to(state[k].dtype)

    if variant == "combo_f20":
        w = 0.20
        for k in state.keys():
            if k.startswith("field_step.") and k in aux_state and torch.is_floating_point(state[k]):
                state[k] = (1.0 - w) * state[k] + w * aux_state[k].to(state[k].dtype)

    if variant == "combo_f35":
        w = 0.35
        for k in state.keys():
            if k.startswith("field_step.") and k in aux_state and torch.is_floating_point(state[k]):
                state[k] = (1.0 - w) * state[k] + w * aux_state[k].to(state[k].dtype)

    return state


@torch.no_grad()
def _evaluate_variant(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    state_dict: dict[str, torch.Tensor],
    alpha_field: float,
    alpha_traj: float,
    high_beta: float,
    high_k_ratio: float,
    max_batches: int,
    spectral_wavenumbers: int,
) -> dict[str, float]:
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    meter: dict[str, list[float]] = defaultdict(list)

    for step_idx, batch in enumerate(tqdm(dataloader, desc="search-eval", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        pred_field = out["field_pred"]
        pred_traj = out["traj_pred"]
        tgt_field = batch["y_future"].float()
        tgt_traj = batch["traj_future"].float()
        pers_field, pers_traj = _persistence(batch)

        af = float(max(0.0, min(1.0, alpha_field)))
        if af < 1.0:
            pred_field = af * pred_field + (1.0 - af) * pers_field

        pred_field = _fft_high_blend(pred_field, pers_field, beta=high_beta, k_ratio=high_k_ratio)
        pred_traj = _blend_traj(pred_traj, pers_traj, alpha_model=alpha_traj)

        # Field metrics on full horizon, consistent with strict benchmark/evaluate.
        b, t, c, h, w = pred_field.shape
        pf = pred_field.reshape(b * t, c, h, w)
        tf = tgt_field.reshape(b * t, c, h, w)

        meter["rmse"].append(float(rmse(pf, tf).item()))
        meter["acc"].append(float(acc(pf, tf).item()))
        meter["track_mae"].append(float(track_mae(pred_traj, tgt_traj).item()))
        meter["extreme_f1"].append(float(extreme_f1(pf, tf).item()))
        meter["spectral_distance"].append(float(spectral_distance(pf, tf, max_wavenumber=spectral_wavenumbers).item()))

    return {k: float(sum(v) / max(1, len(v))) for k, v in meter.items()}


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search splice + fusion variants to improve track_mae/spectral_distance.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_checkpoint", required=True, help="Base (field-strong) checkpoint.")
    parser.add_argument("--aux_checkpoint", required=True, help="Aux (track/spectral-strong) checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_batches", type=int, default=0, help="Limit batches for faster search (0=all).")
    parser.add_argument("--output_dir", default="outputs/miniltg/fusion_search")
    parser.add_argument("--field_interp_w", type=float, default=0.2, help="For field_interp_f20 mode.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["experiment"]["seed"])
    _seed_everything(seed, deterministic=bool(cfg["experiment"].get("deterministic", False)))
    device = _resolve_device(cfg)
    dls = build_dataloaders(cfg)
    spectral_wavenumbers = int(cfg.get("evaluation", {}).get("spectral_wavenumbers", 32))
    use_ema = bool(cfg.get("evaluation", {}).get("use_ema_for_eval", True))

    model = _build_model(cfg, dls, device)
    base_ckpt = _load_ckpt(args.base_checkpoint, device)
    aux_ckpt = _load_ckpt(args.aux_checkpoint, device)
    base_state = _select_model_state(base_ckpt, use_ema=use_ema)
    aux_state = _select_model_state(aux_ckpt, use_ema=use_ema)

    state_variants = [
        "base",
        "traj_splice",
        "field_interp_f20",
        "combo_f20",
        "combo_f35",
    ]
    fusion_variants = [
        {"name": "plain", "alpha_field": 1.0, "alpha_traj": 1.0, "high_beta": 0.0, "high_k_ratio": 0.45},
        {"name": "traj90", "alpha_field": 1.0, "alpha_traj": 0.90, "high_beta": 0.0, "high_k_ratio": 0.45},
        {"name": "field99", "alpha_field": 0.99, "alpha_traj": 1.0, "high_beta": 0.0, "high_k_ratio": 0.45},
        {"name": "both99_90", "alpha_field": 0.99, "alpha_traj": 0.90, "high_beta": 0.0, "high_k_ratio": 0.45},
        {"name": "hf12", "alpha_field": 1.0, "alpha_traj": 1.0, "high_beta": 0.12, "high_k_ratio": 0.45},
        {"name": "hf12_traj90", "alpha_field": 1.0, "alpha_traj": 0.90, "high_beta": 0.12, "high_k_ratio": 0.45},
    ]

    rows: list[dict[str, Any]] = []
    baseline: dict[str, float] | None = None

    for sname in state_variants:
        if sname == "base":
            state = {k: v.clone() for k, v in base_state.items()}
        else:
            state = _apply_variant_state(
                base_state=base_state,
                aux_state=aux_state,
                variant=sname,
                field_interp_w=float(args.field_interp_w),
            )

        for fcfg in fusion_variants:
            vname = f"{sname}__{fcfg['name']}"
            metrics = _evaluate_variant(
                model=model,
                dataloader=dls[args.split],
                device=device,
                state_dict=state,
                alpha_field=float(fcfg["alpha_field"]),
                alpha_traj=float(fcfg["alpha_traj"]),
                high_beta=float(fcfg["high_beta"]),
                high_k_ratio=float(fcfg["high_k_ratio"]),
                max_batches=int(args.max_batches),
                spectral_wavenumbers=spectral_wavenumbers,
            )
            row: dict[str, Any] = {
                "variant": vname,
                "state_variant": sname,
                "fusion_variant": fcfg["name"],
                "alpha_field": float(fcfg["alpha_field"]),
                "alpha_traj": float(fcfg["alpha_traj"]),
                "high_beta": float(fcfg["high_beta"]),
                "high_k_ratio": float(fcfg["high_k_ratio"]),
            }
            row.update(metrics)
            rows.append(row)
            if vname == "base__plain":
                baseline = metrics
            print(f"[{vname}] {metrics}")

    if baseline is None:
        raise RuntimeError("Baseline `base__plain` not evaluated.")

    for r in rows:
        r["delta_rmse"] = float(baseline["rmse"] - r["rmse"])
        r["delta_acc"] = float(r["acc"] - baseline["acc"])
        r["delta_extreme_f1"] = float(r["extreme_f1"] - baseline["extreme_f1"])
        r["delta_track_mae"] = float(baseline["track_mae"] - r["track_mae"])
        r["delta_spectral_distance"] = float(baseline["spectral_distance"] - r["spectral_distance"])
        # Strict-keep means no degradation on main field indicators.
        r["strict_keep"] = bool(
            (r["rmse"] <= baseline["rmse"] + 1e-6)
            and (r["acc"] >= baseline["acc"] - 1e-6)
            and (r["extreme_f1"] >= baseline["extreme_f1"] - 1e-6)
        )
        # Weak-score focuses on both weak items.
        r["weak_score"] = float(r["delta_track_mae"] + r["delta_spectral_distance"])
        # Balanced score penalizes field metric drops.
        penalty = max(0.0, -r["delta_rmse"]) * 8.0 + max(0.0, -r["delta_acc"]) * 120.0 + max(0.0, -r["delta_extreme_f1"]) * 20.0
        r["balanced_score"] = float(r["weak_score"] - penalty)

    strict_rows = [r for r in rows if r["strict_keep"] and r["delta_track_mae"] > 0 and r["delta_spectral_distance"] > 0]
    if strict_rows:
        best = max(strict_rows, key=lambda x: x["weak_score"])
        best_policy = "strict_keep"
    else:
        best = max(rows, key=lambda x: x["balanced_score"])
        best_policy = "balanced_fallback"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(out_dir / "fusion_search_results.csv", rows)
    with (out_dir / "fusion_search_best.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": best_policy,
                "baseline": baseline,
                "best": best,
                "count_variants": len(rows),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== Baseline (base__plain) ===")
    print(baseline)
    print("\n=== Best Variant ===")
    print({"policy": best_policy, **best})
    print(f"Saved: {out_dir / 'fusion_search_results.csv'}")
    print(f"Saved: {out_dir / 'fusion_search_best.json'}")


if __name__ == "__main__":
    main()
