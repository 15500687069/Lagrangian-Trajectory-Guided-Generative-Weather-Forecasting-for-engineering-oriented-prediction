from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltg_net.config import load_config
from ltg_net.data.datamodule import build_dataloaders
from ltg_net.models import LTGNet
from ltg_net.train.loops import move_batch_to_device
from ltg_net.utils.metrics import acc, extreme_f1, rmse, spectral_metric, track_mae


METRICS = ["rmse", "acc", "track_mae", "extreme_f1", "spectral_distance"]
LOWER_BETTER = {"rmse", "track_mae", "spectral_distance"}
LEGACY_GATE_METRICS = ["rmse", "acc", "extreme_f1"]
FIELD_GATE_METRICS = ["rmse", "acc", "extreme_f1", "spectral_distance"]
TRACK_GATE_METRICS = ["track_mae"]
TRAJ_PREFIXES = ("traj_predictor.",)
ENCODER_PREFIX = "encoder."


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


def _resolve_device(cfg: dict[str, Any], requested: str | None) -> torch.device:
    if requested:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    from_cfg = str(cfg["experiment"].get("device", "cuda"))
    if from_cfg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(cfg: dict[str, Any], dataloaders: dict[str, torch.utils.data.DataLoader], device: torch.device) -> LTGNet:
    sample = next(iter(dataloaders["train"]))
    in_channels = int(sample["x_hist"].shape[2])
    model = LTGNet(cfg=cfg, in_channels=in_channels).to(device)
    return model


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


def _compose_fused_state_dict(
    base_model: LTGNet,
    field_state: dict[str, torch.Tensor],
    track_state: dict[str, torch.Tensor],
    fused_encoder_source: str = "field",
) -> dict[str, torch.Tensor]:
    fused = dict(base_model.state_dict())
    # Start from field checkpoint.
    for k in list(fused.keys()):
        if k in field_state:
            fused[k] = field_state[k]
    # Optionally overwrite encoder from track checkpoint.
    if fused_encoder_source == "track":
        for k in list(fused.keys()):
            if k.startswith(ENCODER_PREFIX) and k in track_state:
                fused[k] = track_state[k]
    # Always overwrite trajectory branch from track checkpoint.
    for k in list(fused.keys()):
        if k.startswith(TRAJ_PREFIXES) and k in track_state:
            fused[k] = track_state[k]
    return fused


def _resolve_model_state(
    *,
    model: LTGNet,
    device: torch.device,
    inference_mode: str,
    checkpoint: str,
    field_checkpoint: str,
    track_checkpoint: str,
    fused_encoder_source: str,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    mode = str(inference_mode)
    meta: dict[str, str] = {"inference_mode": mode}
    if mode == "single":
        state = _load_state_dict(checkpoint, device=device)
        meta["checkpoint"] = str(checkpoint)
        return state, meta

    if mode == "field":
        field_path = field_checkpoint or checkpoint
        if not field_path:
            raise ValueError("field mode requires --field_checkpoint or --checkpoint.")
        state = _load_state_dict(field_path, device=device)
        meta["field_checkpoint"] = str(field_path)
        return state, meta

    if mode == "track":
        track_path = track_checkpoint or checkpoint
        if not track_path:
            raise ValueError("track mode requires --track_checkpoint or --checkpoint.")
        state = _load_state_dict(track_path, device=device)
        meta["track_checkpoint"] = str(track_path)
        return state, meta

    if mode == "fused":
        if not field_checkpoint or not track_checkpoint:
            raise ValueError("fused mode requires both --field_checkpoint and --track_checkpoint.")
        field_state = _load_state_dict(field_checkpoint, device=device)
        track_state = _load_state_dict(track_checkpoint, device=device)
        state = _compose_fused_state_dict(
            base_model=model,
            field_state=field_state,
            track_state=track_state,
            fused_encoder_source=str(fused_encoder_source),
        )
        meta["field_checkpoint"] = str(field_checkpoint)
        meta["track_checkpoint"] = str(track_checkpoint)
        meta["fused_encoder_source"] = str(fused_encoder_source)
        return state, meta

    raise ValueError(f"Unsupported inference_mode: {mode}")


def _select_lead_indices(forecast_steps: int, leads: list[int] | None) -> list[int]:
    if not leads:
        return list(range(forecast_steps))
    out: list[int] = []
    for lead in leads:
        idx = int(lead) - 1
        if idx < 0 or idx >= forecast_steps:
            raise ValueError(f"Lead step out of range: {lead}, forecast_steps={forecast_steps}")
        out.append(idx)
    return sorted(set(out))


def _subset_leads_field(field: torch.Tensor, lead_indices: list[int]) -> torch.Tensor:
    return field[:, lead_indices]


def _subset_leads_traj(traj: torch.Tensor, lead_indices: list[int]) -> torch.Tensor:
    return traj[:, lead_indices]


def _predict_persistence(batch: dict[str, torch.Tensor], horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    last_field = batch["x_hist"][:, -1].float()
    last_traj = batch["traj_hist"][:, -1].float()
    field_pred = last_field.unsqueeze(1).repeat(1, horizon, 1, 1, 1)
    traj_pred = last_traj.unsqueeze(1).repeat(1, horizon, 1, 1)
    return field_pred, traj_pred


def _clamp_geo(state: torch.Tensor) -> torch.Tensor:
    lat = state[..., 0].clamp(-90.0, 90.0)
    lon = state[..., 1] % 360.0
    return torch.stack([lat, lon], dim=-1)


def _predict_linear(batch: dict[str, torch.Tensor], horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_hist = batch["x_hist"].float()
    traj_hist = batch["traj_hist"].float()
    if x_hist.shape[1] < 2:
        return _predict_persistence(batch, horizon=horizon)

    prev_field = x_hist[:, -1]
    delta_field = x_hist[:, -1] - x_hist[:, -2]
    field_steps: list[torch.Tensor] = []
    for _ in range(horizon):
        prev_field = prev_field + delta_field
        field_steps.append(prev_field)
    field_pred = torch.stack(field_steps, dim=1)

    prev_traj = traj_hist[:, -1]
    delta_lat = traj_hist[:, -1, :, 0] - traj_hist[:, -2, :, 0]
    delta_lon = (traj_hist[:, -1, :, 1] - traj_hist[:, -2, :, 1] + 180.0) % 360.0 - 180.0
    traj_steps: list[torch.Tensor] = []
    for _ in range(horizon):
        lat = prev_traj[..., 0] + delta_lat
        lon = prev_traj[..., 1] + delta_lon
        prev_traj = _clamp_geo(torch.stack([lat, lon], dim=-1))
        traj_steps.append(prev_traj)
    traj_pred = torch.stack(traj_steps, dim=1)
    return field_pred, traj_pred


def _parse_float_grid(text: str) -> list[float]:
    out: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def _parse_int_grid(text: str) -> list[int]:
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def _blend_traj(model_traj: torch.Tensor, pers_traj: torch.Tensor, alpha_model: float) -> torch.Tensor:
    a = float(max(0.0, min(1.0, alpha_model)))
    if a >= 1.0:
        return model_traj
    lat = a * model_traj[..., 0] + (1.0 - a) * pers_traj[..., 0]
    dlon = (model_traj[..., 1] - pers_traj[..., 1] + 180.0) % 360.0 - 180.0
    lon = (pers_traj[..., 1] + a * dlon) % 360.0
    return torch.stack([lat, lon], dim=-1)


def _fft_high_blend(model_field: torch.Tensor, pers_field: torch.Tensor, beta: float, k_ratio: float) -> torch.Tensor:
    b = float(max(0.0, min(1.0, beta)))
    if b <= 0.0:
        return model_field
    kr = float(max(0.05, min(0.95, k_ratio)))

    mf = torch.fft.rfft2(model_field, dim=(-2, -1), norm="ortho")
    pf = torch.fft.rfft2(pers_field, dim=(-2, -1), norm="ortho")
    h = model_field.shape[-2]
    w2 = mf.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=model_field.device),
        torch.arange(w2, device=model_field.device),
        indexing="ij",
    )
    ky = torch.minimum(yy, h - yy).float() / max(1.0, float(h // 2))
    kx = xx.float() / max(1.0, float(w2 - 1))
    radius = torch.sqrt(kx.square() + ky.square())
    high = (radius >= kr).to(mf.real.dtype)[None, None, None, :, :]
    out_f = mf + b * high * (pf - mf)
    return torch.fft.irfft2(out_f, s=(model_field.shape[-2], model_field.shape[-1]), dim=(-2, -1), norm="ortho")


def _apply_model_calibration(
    model_field: torch.Tensor,
    model_traj: torch.Tensor,
    batch: dict[str, torch.Tensor],
    params: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not params:
        return model_field, model_traj
    horizon = int(model_field.shape[1])
    pers_field, pers_traj = _predict_persistence(batch, horizon=horizon)

    alpha_field = float(params.get("alpha_field", 1.0))
    alpha_traj = float(params.get("alpha_traj", 1.0))
    beta_high = float(params.get("beta_high", 0.0))
    k_ratio = float(params.get("k_ratio", 0.55))
    prefix_leads = int(params.get("prefix_leads", 0))
    alpha_field_prefix = float(params.get("alpha_field_prefix", alpha_field))
    alpha_traj_prefix = float(params.get("alpha_traj_prefix", alpha_traj))

    af = float(max(0.0, min(1.0, alpha_field)))
    field = af * model_field + (1.0 - af) * pers_field
    field = _fft_high_blend(field, pers_field, beta=beta_high, k_ratio=k_ratio)

    traj = _blend_traj(model_traj, pers_traj, alpha_model=alpha_traj)
    if prefix_leads > 0:
        lp = int(min(prefix_leads, horizon))
        if lp > 0:
            afp = float(max(0.0, min(1.0, alpha_field_prefix)))
            atp = float(max(0.0, min(1.0, alpha_traj_prefix)))
            field[:, :lp] = afp * model_field[:, :lp] + (1.0 - afp) * pers_field[:, :lp]
            field[:, :lp] = _fft_high_blend(field[:, :lp], pers_field[:, :lp], beta=beta_high, k_ratio=k_ratio)
            traj[:, :lp] = _blend_traj(model_traj[:, :lp], pers_traj[:, :lp], alpha_model=atp)
    return field, traj


@torch.no_grad()
def _compute_climatology(
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_field = None
    sum_traj = None
    count = 0
    for step_idx, batch in enumerate(tqdm(train_loader, desc="climatology-fit", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        y_future = batch["y_future"].to(device).float()
        traj_future = batch["traj_future"].to(device).float()
        bsz = int(y_future.shape[0])
        if sum_field is None:
            sum_field = y_future.sum(dim=0)
            sum_traj = traj_future.sum(dim=0)
        else:
            sum_field = sum_field + y_future.sum(dim=0)
            sum_traj = sum_traj + traj_future.sum(dim=0)
        count += bsz
    if count <= 0 or sum_field is None or sum_traj is None:
        raise RuntimeError("No data available to compute climatology baseline.")
    return sum_field / float(count), sum_traj / float(count)


def _predict_climatology(
    batch: dict[str, torch.Tensor],
    clim_field: torch.Tensor,
    clim_traj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = int(batch["x_hist"].shape[0])
    field_pred = clim_field.unsqueeze(0).repeat(bsz, 1, 1, 1, 1)
    traj_pred = clim_traj.unsqueeze(0).repeat(bsz, 1, 1, 1)
    return field_pred, traj_pred


def _sample_metric_bundle(
    pred_field: torch.Tensor,
    target_field: torch.Tensor,
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    max_wavenumber: int,
    extreme_quantile: float,
    max_quantile_elements: int,
) -> dict[str, float]:
    return {
        "rmse": float(rmse(pred_field, target_field).item()),
        "acc": float(acc(pred_field, target_field).item()),
        "track_mae": float(track_mae(pred_traj, target_traj).item()),
        "extreme_f1": float(
            extreme_f1(
                pred_field,
                target_field,
                quantile=extreme_quantile,
                max_quantile_elements=max_quantile_elements,
            ).item()
        ),
        "spectral_distance": float(spectral_metric(pred_field[:, 0], target_field[:, 0], max_wavenumber).item()),
    }


@torch.no_grad()
def _evaluate_model_candidate(
    *,
    model: LTGNet,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    lead_indices: list[int],
    max_batches: int,
    max_wavenumber: int,
    extreme_quantile: float,
    max_quantile_elements: int,
    params: dict[str, Any] | None,
) -> dict[str, float]:
    meter: dict[str, list[float]] = {k: [] for k in METRICS}
    for step_idx, batch in enumerate(tqdm(dataloader, desc="calib-eval", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        pred_field = out["field_pred"].float()
        pred_traj = out["traj_pred"].float()
        pred_field, pred_traj = _apply_model_calibration(pred_field, pred_traj, batch, params=params)
        pred_field = _subset_leads_field(pred_field, lead_indices)
        pred_traj = _subset_leads_traj(pred_traj, lead_indices)
        target_field = _subset_leads_field(batch["y_future"].float(), lead_indices)
        target_traj = _subset_leads_traj(batch["traj_future"].float(), lead_indices)
        bsz = int(target_field.shape[0])
        for i in range(bsz):
            bundle = _sample_metric_bundle(
                pred_field=pred_field[i : i + 1],
                target_field=target_field[i : i + 1],
                pred_traj=pred_traj[i : i + 1],
                target_traj=target_traj[i : i + 1],
                max_wavenumber=max_wavenumber,
                extreme_quantile=extreme_quantile,
                max_quantile_elements=max_quantile_elements,
            )
            for k, v in bundle.items():
                meter[k].append(float(v))
    return {k: float(np.mean(meter[k])) if meter[k] else float("nan") for k in METRICS}


@torch.no_grad()
def _evaluate_reference_candidate(
    *,
    reference: str,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    lead_indices: list[int],
    max_batches: int,
    max_wavenumber: int,
    extreme_quantile: float,
    max_quantile_elements: int,
    clim_field: torch.Tensor | None = None,
    clim_traj: torch.Tensor | None = None,
) -> dict[str, float]:
    meter: dict[str, list[float]] = {k: [] for k in METRICS}
    ref = str(reference)
    for step_idx, batch in enumerate(tqdm(dataloader, desc=f"calib-ref-{ref}", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        horizon = int(batch["y_future"].shape[1])
        if ref == "persistence":
            pred_field, pred_traj = _predict_persistence(batch, horizon=horizon)
        elif ref == "linear":
            pred_field, pred_traj = _predict_linear(batch, horizon=horizon)
        elif ref == "climatology":
            if clim_field is None or clim_traj is None:
                raise RuntimeError("Climatology tensors are required for reference=climatology.")
            pred_field, pred_traj = _predict_climatology(batch, clim_field=clim_field, clim_traj=clim_traj)
        else:
            raise ValueError(f"Unsupported reference method: {ref}")
        pred_field = _subset_leads_field(pred_field, lead_indices)
        pred_traj = _subset_leads_traj(pred_traj, lead_indices)
        target_field = _subset_leads_field(batch["y_future"].float(), lead_indices)
        target_traj = _subset_leads_traj(batch["traj_future"].float(), lead_indices)
        bsz = int(target_field.shape[0])
        for i in range(bsz):
            bundle = _sample_metric_bundle(
                pred_field=pred_field[i : i + 1],
                target_field=target_field[i : i + 1],
                pred_traj=pred_traj[i : i + 1],
                target_traj=target_traj[i : i + 1],
                max_wavenumber=max_wavenumber,
                extreme_quantile=extreme_quantile,
                max_quantile_elements=max_quantile_elements,
            )
            for k, v in bundle.items():
                meter[k].append(float(v))
    return {k: float(np.mean(meter[k])) if meter[k] else float("nan") for k in METRICS}


def _gate_margin(model_value: float, ref_value: float, metric: str) -> float:
    if metric in LOWER_BETTER:
        return float(ref_value - model_value)
    return float(model_value - ref_value)


def _resolve_gate_metrics(gate_profile: str, inference_mode: str, gate_metrics_text: str) -> list[str]:
    if gate_metrics_text:
        metrics = [m.strip() for m in str(gate_metrics_text).split(",") if m.strip()]
    elif gate_profile == "legacy":
        metrics = list(LEGACY_GATE_METRICS)
    elif gate_profile == "all":
        metrics = list(METRICS)
    elif gate_profile == "by_mode":
        metrics = list(TRACK_GATE_METRICS if str(inference_mode) == "track" else FIELD_GATE_METRICS)
    else:
        raise ValueError(f"Unsupported gate profile: {gate_profile}")

    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"Unsupported gate metrics: {unknown}. valid={METRICS}")
    return metrics


def _fit_model_calibration(
    *,
    model: LTGNet,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    calib_split: str,
    reference: str,
    device: torch.device,
    lead_indices: list[int],
    max_batches: int,
    max_wavenumber: int,
    extreme_quantile: float,
    max_quantile_elements: int,
    gate_metrics: list[str],
    alpha_field_grid: list[float],
    alpha_traj_grid: list[float],
    beta_high_grid: list[float],
    k_ratio_grid: list[float],
    prefix_leads_grid: list[int],
    alpha_field_prefix_grid: list[float],
    alpha_traj_prefix_grid: list[float],
    max_candidates: int,
    seed: int,
    rmse_tol: float,
    acc_tol: float,
    f1_tol: float,
    track_tol: float,
    spectral_tol: float,
    clim_field: torch.Tensor | None = None,
    clim_traj: torch.Tensor | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float], dict[str, float], dict[str, float]]:
    loader = dataloaders[str(calib_split)]
    baseline_metrics = _evaluate_model_candidate(
        model=model,
        dataloader=loader,
        device=device,
        lead_indices=lead_indices,
        max_batches=max_batches,
        max_wavenumber=max_wavenumber,
        extreme_quantile=extreme_quantile,
        max_quantile_elements=max_quantile_elements,
        params=None,
    )
    reference_metrics = _evaluate_reference_candidate(
        reference=reference,
        dataloader=loader,
        device=device,
        lead_indices=lead_indices,
        max_batches=max_batches,
        max_wavenumber=max_wavenumber,
        extreme_quantile=extreme_quantile,
        max_quantile_elements=max_quantile_elements,
        clim_field=clim_field,
        clim_traj=clim_traj,
    )

    candidates: list[dict[str, Any]] = []
    first_k_ratio = float(k_ratio_grid[0]) if k_ratio_grid else 0.55
    for alpha_field in alpha_field_grid:
        for alpha_traj in alpha_traj_grid:
            for beta_high in beta_high_grid:
                for k_ratio in k_ratio_grid:
                    if float(beta_high) == 0.0 and float(k_ratio) != first_k_ratio:
                        continue
                    for prefix_leads in prefix_leads_grid:
                        field_prefix_grid = [1.0] if int(prefix_leads) <= 0 else alpha_field_prefix_grid
                        traj_prefix_grid = [1.0] if int(prefix_leads) <= 0 else alpha_traj_prefix_grid
                        for alpha_field_prefix in field_prefix_grid:
                            for alpha_traj_prefix in traj_prefix_grid:
                                candidates.append(
                                    {
                                        "alpha_field": float(alpha_field),
                                        "alpha_traj": float(alpha_traj),
                                        "beta_high": float(beta_high),
                                        "k_ratio": float(k_ratio),
                                        "prefix_leads": int(prefix_leads),
                                        "alpha_field_prefix": float(alpha_field_prefix),
                                        "alpha_traj_prefix": float(alpha_traj_prefix),
                                    }
                                )

    # De-duplicate.
    seen: set[tuple[Any, ...]] = set()
    unique_candidates: list[dict[str, Any]] = []
    for c in candidates:
        key = (
            c["alpha_field"],
            c["alpha_traj"],
            c["beta_high"],
            c["k_ratio"],
            c["prefix_leads"],
            c["alpha_field_prefix"],
            c["alpha_traj_prefix"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(c)
    candidates = unique_candidates

    if max_candidates > 0 and len(candidates) > max_candidates:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(candidates), size=int(max_candidates), replace=False)
        candidates = [candidates[int(i)] for i in idx]

    rows: list[dict[str, Any]] = []
    best_feasible: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None

    for c in tqdm(candidates, desc="calib-search", leave=False):
        metrics = _evaluate_model_candidate(
            model=model,
            dataloader=loader,
            device=device,
            lead_indices=lead_indices,
            max_batches=max_batches,
            max_wavenumber=max_wavenumber,
            extreme_quantile=extreme_quantile,
            max_quantile_elements=max_quantile_elements,
            params=c,
        )
        margins = {m: _gate_margin(metrics[m], reference_metrics[m], m) for m in gate_metrics}
        gate_feasible = all(float(v) >= 0.0 for v in margins.values())

        delta_rmse = float(metrics["rmse"] - baseline_metrics["rmse"])
        delta_acc = float(metrics["acc"] - baseline_metrics["acc"])
        delta_f1 = float(metrics["extreme_f1"] - baseline_metrics["extreme_f1"])
        delta_track = float(metrics["track_mae"] - baseline_metrics["track_mae"])
        delta_spec = float(metrics["spectral_distance"] - baseline_metrics["spectral_distance"])
        preserve = (
            delta_rmse <= float(rmse_tol)
            and delta_acc >= -float(acc_tol)
            and delta_f1 >= -float(f1_tol)
            and delta_track <= float(track_tol)
            and delta_spec <= float(spectral_tol)
        )

        # Normalize by reference magnitude for robust ranking.
        objective = 0.0
        for m in gate_metrics:
            denom = abs(float(reference_metrics[m])) + 1e-8
            nm = float(margins[m]) / denom
            objective += nm
            if nm < 0.0:
                objective += 2.0 * nm
        if not preserve:
            objective -= 5.0

        row = dict(c)
        for m in METRICS:
            row[m] = float(metrics[m])
            row[f"ref_{m}"] = float(reference_metrics[m])
            row[f"margin_{m}"] = float(_gate_margin(metrics[m], reference_metrics[m], m))
        row["delta_rmse_vs_base"] = delta_rmse
        row["delta_acc_vs_base"] = delta_acc
        row["delta_extreme_f1_vs_base"] = delta_f1
        row["delta_track_mae_vs_base"] = delta_track
        row["delta_spectral_distance_vs_base"] = delta_spec
        row["gate_feasible"] = bool(gate_feasible)
        row["preserve_feasible"] = bool(preserve)
        row["objective"] = float(objective)
        rows.append(row)

        candidate = {"params": dict(c), "metrics": dict(metrics), "objective": float(objective), "preserve": bool(preserve)}
        if gate_feasible and preserve:
            if best_feasible is None or float(candidate["objective"]) > float(best_feasible["objective"]):
                best_feasible = candidate
        if best_any is None or float(candidate["objective"]) > float(best_any["objective"]):
            best_any = candidate

    chosen = best_feasible or best_any
    if chosen is None:
        raise RuntimeError("Calibration search produced no candidates.")
    return (
        dict(chosen["params"]),
        rows,
        dict(baseline_metrics),
        dict(reference_metrics),
        dict(chosen["metrics"]),
    )


def _bootstrap_ci_mean(values: np.ndarray, n_bootstrap: int, alpha: float, rng: np.random.Generator) -> tuple[float, float]:
    if values.size <= 1 or n_bootstrap <= 0:
        mean = float(values.mean()) if values.size > 0 else float("nan")
        return mean, mean
    n = int(values.size)
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha * 0.5))
    hi = float(np.quantile(means, 1.0 - alpha * 0.5))
    return lo, hi


def _bootstrap_p_better(
    model_values: np.ndarray,
    ref_values: np.ndarray,
    metric: str,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    if model_values.size != ref_values.size:
        raise ValueError(f"Length mismatch for metric={metric}: {model_values.size} vs {ref_values.size}")
    if metric in LOWER_BETTER:
        diff = ref_values - model_values
    else:
        diff = model_values - ref_values
    delta_mean = float(diff.mean())
    if diff.size <= 1 or n_bootstrap <= 0:
        p_better = float((diff > 0.0).mean())
        return delta_mean, p_better, delta_mean, delta_mean
    n = int(diff.size)
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    boot_means = diff[idx].mean(axis=1)
    p_better = float((boot_means > 0.0).mean())
    ci_lo = float(np.quantile(boot_means, alpha * 0.5))
    ci_hi = float(np.quantile(boot_means, 1.0 - alpha * 0.5))
    return delta_mean, p_better, ci_lo, ci_hi


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_report(
    cfg_path: str,
    inference_mode: str,
    ckpt_path: str | None,
    field_ckpt_path: str | None,
    track_ckpt_path: str | None,
    split: str,
    methods: list[str],
    leads: list[int],
    summary_rows: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# LTG Strict Benchmark Report")
    lines.append("")
    lines.append("## Protocol")
    lines.append(f"- config: `{cfg_path}`")
    lines.append(f"- inference_mode: `{inference_mode}`")
    if ckpt_path:
        lines.append(f"- checkpoint: `{ckpt_path}`")
    if field_ckpt_path:
        lines.append(f"- field_checkpoint: `{field_ckpt_path}`")
    if track_ckpt_path:
        lines.append(f"- track_checkpoint: `{track_ckpt_path}`")
    lines.append(f"- split: `{split}`")
    lines.append(f"- methods: `{methods}`")
    lines.append(f"- lead_steps: `{leads}`")
    lines.append("")
    lines.append("## Summary")
    for row in summary_rows:
        lines.append(
            f"- `{row['method']}`: rmse={row['rmse']:.6f}, acc={row['acc']:.6f}, "
            f"track_mae={row['track_mae']:.6f}, extreme_f1={row['extreme_f1']:.6f}, "
            f"spectral_distance={row['spectral_distance']:.6f}"
        )
    lines.append("")
    lines.append("## Skill vs Reference")
    ref = gate.get("reference")
    lines.append(f"- reference: `{ref}`")
    for row in skill_rows:
        lines.append(
            f"- `{row['method']}` / `{row['metric']}`: delta={row['delta_mean']:.6f}, "
            f"p_better={row['p_better']:.3f}, ci=[{row['delta_ci_lower']:.6f}, {row['delta_ci_upper']:.6f}]"
        )
    lines.append("")
    lines.append("## Gate")
    lines.append(f"- rule: `{gate['rule']}`")
    if "gate_profile" in gate:
        lines.append(f"- gate_profile: `{gate['gate_profile']}`")
    if "gate_metrics" in gate:
        lines.append(f"- gate_metrics: `{gate['gate_metrics']}`")
    lines.append(f"- strict_p_threshold: `{gate['strict_p_threshold']}`")
    mean_checks = gate.get("mean_checks")
    if isinstance(mean_checks, dict) and mean_checks:
        lines.append(f"- mean_checks: `{mean_checks}`")
    p_checks = gate.get("p_checks")
    if isinstance(p_checks, dict) and p_checks:
        lines.append(f"- p_checks: `{p_checks}`")
    lines.append(f"- result: **{'PASS' if gate['pass'] else 'FAIL'}**")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict benchmark for LTG-Net with baseline methods.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="", help="Model checkpoint path (required when method includes model).")
    parser.add_argument(
        "--inference_mode",
        default="single",
        choices=["single", "field", "track", "fused"],
        help="single=one checkpoint; field/track=single-branch ckpt; fused=field ckpt + track ckpt stitching.",
    )
    parser.add_argument("--field_checkpoint", default="", help="Checkpoint used by field mode or fused mode.")
    parser.add_argument("--track_checkpoint", default="", help="Checkpoint used by track mode or fused mode.")
    parser.add_argument(
        "--fused_encoder_source",
        default="field",
        choices=["field", "track"],
        help="In fused mode, choose encoder source.",
    )
    parser.add_argument(
        "--save_fused_checkpoint",
        default="",
        help="Optional path to save materialized fused checkpoint state_dict.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["model", "persistence", "linear", "climatology"],
        choices=["model", "persistence", "linear", "climatology"],
    )
    parser.add_argument("--leads", nargs="*", type=int, default=None, help="1-based lead steps to evaluate.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--ci_alpha", type=float, default=0.05)
    parser.add_argument("--reference", default="persistence")
    parser.add_argument("--strict_p_threshold", type=float, default=0.95)
    parser.add_argument(
        "--gate_profile",
        default="legacy",
        choices=["legacy", "by_mode", "all"],
        help="legacy=rmse/acc/extreme_f1; by_mode=track uses track_mae while others use field metrics; all=all five metrics.",
    )
    parser.add_argument(
        "--gate_metrics",
        default="",
        help="Optional explicit gate metrics, comma-separated, e.g. rmse,acc,extreme_f1 or track_mae.",
    )
    parser.add_argument(
        "--calibrate_on_split",
        default="none",
        choices=["none", "train", "val", "test"],
        help="Search inference calibration parameters on this split before final evaluation.",
    )
    parser.add_argument("--calibration_json", default="", help="Use an existing calibration JSON.")
    parser.add_argument("--calib_alpha_field_grid", default="1.0,0.95,0.9")
    parser.add_argument("--calib_alpha_traj_grid", default="1.0,0.9,0.8")
    parser.add_argument("--calib_beta_high_grid", default="0.0,0.12")
    parser.add_argument("--calib_k_ratio_grid", default="0.55,0.65")
    parser.add_argument("--calib_prefix_leads_grid", default="0,2")
    parser.add_argument("--calib_alpha_field_prefix_grid", default="1.0,0.6")
    parser.add_argument("--calib_alpha_traj_prefix_grid", default="1.0,0.7")
    parser.add_argument("--calib_max_batches", type=int, default=0, help="Max batches for calibration split (0=use max_batches).")
    parser.add_argument("--calib_max_candidates", type=int, default=120, help="Randomly sample at most this many candidates.")
    parser.add_argument("--calib_rmse_tol", type=float, default=0.005)
    parser.add_argument("--calib_acc_tol", type=float, default=0.005)
    parser.add_argument("--calib_f1_tol", type=float, default=0.010)
    parser.add_argument("--calib_track_tol", type=float, default=0.300)
    parser.add_argument("--calib_spectral_tol", type=float, default=0.030)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="", choices=["", "cpu", "cuda"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--climatology_train_max_batches", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/ltg/strict_benchmark")
    args = parser.parse_args()

    methods = [str(m) for m in args.methods]
    if "model" in methods:
        mode = str(args.inference_mode)
        if mode == "single" and not args.checkpoint:
            raise ValueError("--checkpoint is required when methods include model and inference_mode=single.")
        if mode == "field" and not (args.field_checkpoint or args.checkpoint):
            raise ValueError("field mode requires --field_checkpoint (or fallback --checkpoint).")
        if mode == "track" and not (args.track_checkpoint or args.checkpoint):
            raise ValueError("track mode requires --track_checkpoint (or fallback --checkpoint).")
        if mode == "fused" and (not args.field_checkpoint or not args.track_checkpoint):
            raise ValueError("fused mode requires both --field_checkpoint and --track_checkpoint.")
    if args.reference not in methods:
        raise ValueError(f"--reference must be included in methods. got reference={args.reference}, methods={methods}")
    gate_metrics = _resolve_gate_metrics(
        gate_profile=str(args.gate_profile),
        inference_mode=str(args.inference_mode),
        gate_metrics_text=str(args.gate_metrics),
    )

    cfg = load_config(args.config)
    _seed_everything(int(args.seed), deterministic=bool(args.deterministic))
    device = _resolve_device(cfg, requested=(args.device or None))
    print(f"Device: {device.type}")

    dataloaders = build_dataloaders(cfg)
    split_loader = dataloaders[args.split]
    sample = next(iter(split_loader))
    horizon = int(sample["y_future"].shape[1])
    lead_indices = _select_lead_indices(horizon, args.leads)
    lead_steps = [idx + 1 for idx in lead_indices]

    model = None
    model_meta: dict[str, str] = {"inference_mode": str(args.inference_mode)}
    if "model" in methods:
        model = _build_model(cfg, dataloaders=dataloaders, device=device)
        state, model_meta = _resolve_model_state(
            model=model,
            device=device,
            inference_mode=str(args.inference_mode),
            checkpoint=str(args.checkpoint),
            field_checkpoint=str(args.field_checkpoint),
            track_checkpoint=str(args.track_checkpoint),
            fused_encoder_source=str(args.fused_encoder_source),
        )
        model.load_state_dict(state)
        model.eval()
        if args.save_fused_checkpoint:
            out_fused = Path(args.save_fused_checkpoint)
            out_fused.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": state, "meta": model_meta}, out_fused)
            print(f"Saved fused checkpoint: {out_fused}")

    clim_field = None
    clim_traj = None
    if "climatology" in methods:
        clim_field, clim_traj = _compute_climatology(
            train_loader=dataloaders["train"],
            device=device,
            max_batches=int(args.climatology_train_max_batches),
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_wavenumber = int(cfg["loss"]["spectral_wavenumbers"])
    extreme_quantile = float(cfg["loss"].get("field_extreme_quantile", 0.98))
    max_quantile_elements = int(cfg["loss"].get("field_extreme_max_elements", 5_000_000))

    model_calibration: dict[str, Any] | None = None
    calibration_baseline: dict[str, float] | None = None
    calibration_reference: dict[str, float] | None = None
    calibration_chosen: dict[str, float] | None = None
    if "model" in methods:
        if args.calibration_json:
            cpath = Path(args.calibration_json)
            if not cpath.exists():
                raise FileNotFoundError(f"Calibration JSON not found: {cpath}")
            payload = json.loads(cpath.read_text(encoding="utf-8"))
            model_calibration = dict(payload.get("chosen_params", payload))
            if "baseline_metrics" in payload and isinstance(payload["baseline_metrics"], dict):
                calibration_baseline = {k: float(v) for k, v in payload["baseline_metrics"].items() if k in METRICS}
            if "reference_metrics" in payload and isinstance(payload["reference_metrics"], dict):
                calibration_reference = {k: float(v) for k, v in payload["reference_metrics"].items() if k in METRICS}
            if "chosen_metrics" in payload and isinstance(payload["chosen_metrics"], dict):
                calibration_chosen = {k: float(v) for k, v in payload["chosen_metrics"].items() if k in METRICS}
        elif str(args.calibrate_on_split) != "none":
            alpha_field_grid = _parse_float_grid(args.calib_alpha_field_grid)
            alpha_traj_grid = _parse_float_grid(args.calib_alpha_traj_grid)
            beta_high_grid = _parse_float_grid(args.calib_beta_high_grid)
            k_ratio_grid = _parse_float_grid(args.calib_k_ratio_grid)
            prefix_leads_grid = _parse_int_grid(args.calib_prefix_leads_grid)
            alpha_field_prefix_grid = _parse_float_grid(args.calib_alpha_field_prefix_grid)
            alpha_traj_prefix_grid = _parse_float_grid(args.calib_alpha_traj_prefix_grid)
            if not alpha_field_grid or not alpha_traj_grid or not beta_high_grid or not k_ratio_grid:
                raise ValueError("Calibration float grids must be non-empty.")
            if not prefix_leads_grid or not alpha_field_prefix_grid or not alpha_traj_prefix_grid:
                raise ValueError("Calibration prefix grids must be non-empty.")

            calib_max_batches = int(args.calib_max_batches) if int(args.calib_max_batches) > 0 else int(args.max_batches)
            chosen_params, calib_rows, calib_baseline, calib_reference, calib_chosen = _fit_model_calibration(
                model=model,
                dataloaders=dataloaders,
                calib_split=str(args.calibrate_on_split),
                reference=str(args.reference),
                device=device,
                lead_indices=lead_indices,
                max_batches=calib_max_batches,
                max_wavenumber=max_wavenumber,
                extreme_quantile=extreme_quantile,
                max_quantile_elements=max_quantile_elements,
                gate_metrics=gate_metrics,
                alpha_field_grid=alpha_field_grid,
                alpha_traj_grid=alpha_traj_grid,
                beta_high_grid=beta_high_grid,
                k_ratio_grid=k_ratio_grid,
                prefix_leads_grid=prefix_leads_grid,
                alpha_field_prefix_grid=alpha_field_prefix_grid,
                alpha_traj_prefix_grid=alpha_traj_prefix_grid,
                max_candidates=int(args.calib_max_candidates),
                seed=int(args.seed),
                rmse_tol=float(args.calib_rmse_tol),
                acc_tol=float(args.calib_acc_tol),
                f1_tol=float(args.calib_f1_tol),
                track_tol=float(args.calib_track_tol),
                spectral_tol=float(args.calib_spectral_tol),
                clim_field=clim_field,
                clim_traj=clim_traj,
            )
            model_calibration = dict(chosen_params)
            calibration_baseline = dict(calib_baseline)
            calibration_reference = dict(calib_reference)
            calibration_chosen = dict(calib_chosen)
            calib_csv = out_dir / "model_calibration_search.csv"
            _write_csv(
                calib_csv,
                calib_rows,
                fieldnames=[
                    "alpha_field",
                    "alpha_traj",
                    "beta_high",
                    "k_ratio",
                    "prefix_leads",
                    "alpha_field_prefix",
                    "alpha_traj_prefix",
                    *METRICS,
                    *[f"ref_{m}" for m in METRICS],
                    *[f"margin_{m}" for m in METRICS],
                    "delta_rmse_vs_base",
                    "delta_acc_vs_base",
                    "delta_extreme_f1_vs_base",
                    "delta_track_mae_vs_base",
                    "delta_spectral_distance_vs_base",
                    "gate_feasible",
                    "preserve_feasible",
                    "objective",
                ],
            )
            calib_json = out_dir / "model_calibration.json"
            calib_payload = {
                "config": args.config,
                "calibrate_on_split": str(args.calibrate_on_split),
                "reference": str(args.reference),
                "gate_profile": str(args.gate_profile),
                "gate_metrics": gate_metrics,
                "chosen_params": model_calibration,
                "baseline_metrics": calibration_baseline,
                "reference_metrics": calibration_reference,
                "chosen_metrics": calibration_chosen,
            }
            calib_json.write_text(json.dumps(calib_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved calibration search CSV: {calib_csv}")
            print(f"Saved calibration JSON: {calib_json}")

    metric_values: dict[str, dict[str, list[float]]] = {
        m: {k: [] for k in METRICS} for m in methods
    }
    lead_metric_values: dict[str, dict[int, dict[str, list[float]]]] = {
        m: {lead: {k: [] for k in METRICS} for lead in lead_steps} for m in methods
    }

    with torch.no_grad():
        pbar = tqdm(split_loader, desc=f"strict-{args.split}", leave=False)
        for step_idx, batch in enumerate(pbar):
            if args.max_batches > 0 and step_idx >= int(args.max_batches):
                break
            batch = move_batch_to_device(batch, device)
            target_field = _subset_leads_field(batch["y_future"].float(), lead_indices)
            target_traj = _subset_leads_traj(batch["traj_future"].float(), lead_indices)

            pred_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            if "model" in methods:
                assert model is not None
                out = model(batch)
                model_field = out["field_pred"].float()
                model_traj = out["traj_pred"].float()
                model_field, model_traj = _apply_model_calibration(
                    model_field=model_field,
                    model_traj=model_traj,
                    batch=batch,
                    params=model_calibration,
                )
                pred_cache["model"] = (
                    _subset_leads_field(model_field, lead_indices),
                    _subset_leads_traj(model_traj, lead_indices),
                )
            if "persistence" in methods:
                pf, pt = _predict_persistence(batch, horizon=horizon)
                pred_cache["persistence"] = (_subset_leads_field(pf, lead_indices), _subset_leads_traj(pt, lead_indices))
            if "linear" in methods:
                pf, pt = _predict_linear(batch, horizon=horizon)
                pred_cache["linear"] = (_subset_leads_field(pf, lead_indices), _subset_leads_traj(pt, lead_indices))
            if "climatology" in methods:
                assert clim_field is not None and clim_traj is not None
                pf, pt = _predict_climatology(batch, clim_field=clim_field, clim_traj=clim_traj)
                pred_cache["climatology"] = (
                    _subset_leads_field(pf, lead_indices),
                    _subset_leads_traj(pt, lead_indices),
                )

            bsz = int(target_field.shape[0])
            for method in methods:
                pred_field, pred_traj = pred_cache[method]
                for i in range(bsz):
                    pf_i = pred_field[i : i + 1]
                    tf_i = target_field[i : i + 1]
                    pt_i = pred_traj[i : i + 1]
                    tt_i = target_traj[i : i + 1]
                    bundle = _sample_metric_bundle(
                        pred_field=pf_i,
                        target_field=tf_i,
                        pred_traj=pt_i,
                        target_traj=tt_i,
                        max_wavenumber=max_wavenumber,
                        extreme_quantile=extreme_quantile,
                        max_quantile_elements=max_quantile_elements,
                    )
                    for metric, value in bundle.items():
                        metric_values[method][metric].append(float(value))

                    for li, lead in enumerate(lead_steps):
                        pf_l = pf_i[:, li : li + 1]
                        tf_l = tf_i[:, li : li + 1]
                        pt_l = pt_i[:, li : li + 1]
                        tt_l = tt_i[:, li : li + 1]
                        lead_bundle = _sample_metric_bundle(
                            pred_field=pf_l,
                            target_field=tf_l,
                            pred_traj=pt_l,
                            target_traj=tt_l,
                            max_wavenumber=max_wavenumber,
                            extreme_quantile=extreme_quantile,
                            max_quantile_elements=max_quantile_elements,
                        )
                        for metric, value in lead_bundle.items():
                            lead_metric_values[method][lead][metric].append(float(value))

    rng = np.random.default_rng(int(args.seed))
    summary_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    for method in methods:
        row = {"method": method, "split": args.split}
        for metric in METRICS:
            arr = np.asarray(metric_values[method][metric], dtype=np.float64)
            mean = float(arr.mean()) if arr.size > 0 else float("nan")
            row[metric] = mean
            lo, hi = _bootstrap_ci_mean(
                arr,
                n_bootstrap=int(args.bootstrap),
                alpha=float(args.ci_alpha),
                rng=rng,
            )
            ci_rows.append(
                {
                    "method": method,
                    "split": args.split,
                    "metric": metric,
                    "mean": mean,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_samples": int(arr.size),
                }
            )
        summary_rows.append(row)

        for lead in lead_steps:
            lead_row = {"method": method, "split": args.split, "lead": int(lead)}
            for metric in METRICS:
                arr = np.asarray(lead_metric_values[method][lead][metric], dtype=np.float64)
                lead_row[metric] = float(arr.mean()) if arr.size > 0 else float("nan")
            lead_rows.append(lead_row)

    skill_rows: list[dict[str, Any]] = []
    lead_skill_rows: list[dict[str, Any]] = []
    ref = str(args.reference)
    ref_metrics = {k: np.asarray(metric_values[ref][k], dtype=np.float64) for k in METRICS}
    for method in methods:
        if method == ref:
            continue
        for metric in METRICS:
            model_arr = np.asarray(metric_values[method][metric], dtype=np.float64)
            ref_arr = ref_metrics[metric]
            delta_mean, p_better, lo, hi = _bootstrap_p_better(
                model_values=model_arr,
                ref_values=ref_arr,
                metric=metric,
                n_bootstrap=int(args.bootstrap),
                alpha=float(args.ci_alpha),
                rng=rng,
            )
            skill_rows.append(
                {
                    "method": method,
                    "reference": ref,
                    "metric": metric,
                    "delta_mean": delta_mean,
                    "p_better": p_better,
                    "delta_ci_lower": lo,
                    "delta_ci_upper": hi,
                }
            )

        for lead in lead_steps:
            for metric in METRICS:
                model_arr = np.asarray(lead_metric_values[method][lead][metric], dtype=np.float64)
                ref_arr = np.asarray(lead_metric_values[ref][lead][metric], dtype=np.float64)
                delta_mean, p_better, lo, hi = _bootstrap_p_better(
                    model_values=model_arr,
                    ref_values=ref_arr,
                    metric=metric,
                    n_bootstrap=int(args.bootstrap),
                    alpha=float(args.ci_alpha),
                    rng=rng,
                )
                lead_skill_rows.append(
                    {
                        "method": method,
                        "reference": ref,
                        "lead": int(lead),
                        "metric": metric,
                        "delta_mean": delta_mean,
                        "p_better": p_better,
                        "delta_ci_lower": lo,
                        "delta_ci_upper": hi,
                    }
                )

    summary_by_method = {r["method"]: r for r in summary_rows}
    model_row = summary_by_method.get("model")
    ref_row = summary_by_method.get(ref)
    p_map = {(r["method"], r["metric"]): float(r["p_better"]) for r in skill_rows}
    base_gate = False
    strict_gate = False
    mean_checks: dict[str, bool] = {}
    p_checks: dict[str, bool] = {}
    if model_row is not None and ref_row is not None:
        for metric in gate_metrics:
            mv = float(model_row[metric])
            rv = float(ref_row[metric])
            if metric in LOWER_BETTER:
                mean_ok = mv < rv
            elif metric == "acc":
                mean_ok = mv > rv
            else:
                mean_ok = mv >= rv
            mean_checks[metric] = bool(mean_ok)
            p_checks[metric] = bool(p_map.get(("model", metric), 0.0) >= float(args.strict_p_threshold))
        base_gate = all(mean_checks.values()) if mean_checks else False
        strict_gate = base_gate and all(p_checks.values()) if p_checks else False

    rule_parts: list[str] = []
    for metric in gate_metrics:
        if metric in LOWER_BETTER:
            rule_parts.append(f"model {metric} < reference {metric}")
        elif metric == "acc":
            rule_parts.append("model acc > reference acc")
        else:
            rule_parts.append(f"model {metric} >= reference {metric}")
    rule_text = " and ".join(rule_parts)
    p_text = f"p_better({','.join(gate_metrics)}) >= strict_p_threshold"

    gate = {
        "reference": ref,
        "gate_profile": str(args.gate_profile),
        "gate_metrics": list(gate_metrics),
        "rule": f"{rule_text} and {p_text}",
        "strict_p_threshold": float(args.strict_p_threshold),
        "mean_checks": mean_checks,
        "p_checks": p_checks,
        "base_pass": bool(base_gate),
        "pass": bool(strict_gate),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "strict_summary.csv"
    ci_csv = out_dir / "strict_ci.csv"
    lead_csv = out_dir / "strict_lead_summary.csv"
    skill_csv = out_dir / f"strict_skill_vs_{ref}.csv"
    lead_skill_csv = out_dir / f"strict_lead_skill_vs_{ref}.csv"
    result_json = out_dir / "strict_results.json"
    report_md = out_dir / "strict_report.md"

    _write_csv(
        summary_csv,
        summary_rows,
        fieldnames=["method", "split", *METRICS],
    )
    _write_csv(
        ci_csv,
        ci_rows,
        fieldnames=["method", "split", "metric", "mean", "ci_lower", "ci_upper", "n_samples"],
    )
    _write_csv(
        lead_csv,
        lead_rows,
        fieldnames=["method", "split", "lead", *METRICS],
    )
    _write_csv(
        skill_csv,
        skill_rows,
        fieldnames=["method", "reference", "metric", "delta_mean", "p_better", "delta_ci_lower", "delta_ci_upper"],
    )
    _write_csv(
        lead_skill_csv,
        lead_skill_rows,
        fieldnames=[
            "method",
            "reference",
            "lead",
            "metric",
            "delta_mean",
            "p_better",
            "delta_ci_lower",
            "delta_ci_upper",
        ],
    )

    result_payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "inference_mode": str(args.inference_mode),
        "field_checkpoint": str(args.field_checkpoint),
        "track_checkpoint": str(args.track_checkpoint),
        "model_meta": model_meta,
        "split": args.split,
        "methods": methods,
        "leads": lead_steps,
        "bootstrap": int(args.bootstrap),
        "ci_alpha": float(args.ci_alpha),
        "gate_profile": str(args.gate_profile),
        "gate_metrics": list(gate_metrics),
        "model_calibration": model_calibration,
        "calibration_baseline_metrics": calibration_baseline,
        "calibration_reference_metrics": calibration_reference,
        "calibration_chosen_metrics": calibration_chosen,
        "summary": summary_rows,
        "ci": ci_rows,
        "lead_summary": lead_rows,
        "skill_vs_reference": skill_rows,
        "lead_skill_vs_reference": lead_skill_rows,
        "gate": gate,
    }
    result_json.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md.write_text(
        _build_report(
            cfg_path=args.config,
            inference_mode=str(args.inference_mode),
            ckpt_path=args.checkpoint if args.checkpoint else None,
            field_ckpt_path=args.field_checkpoint if args.field_checkpoint else None,
            track_ckpt_path=args.track_checkpoint if args.track_checkpoint else None,
            split=args.split,
            methods=methods,
            leads=lead_steps,
            summary_rows=summary_rows,
            skill_rows=skill_rows,
            gate=gate,
        ),
        encoding="utf-8",
    )

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved CI CSV: {ci_csv}")
    print(f"Saved lead CSV: {lead_csv}")
    print(f"Saved skill CSV: {skill_csv}")
    print(f"Saved lead-skill CSV: {lead_skill_csv}")
    print(f"Saved JSON: {result_json}")
    print(f"Saved report: {report_md}")
    print("Summary:")
    for row in summary_rows:
        print(row)
    print(f"Gate: {'PASS' if gate['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
