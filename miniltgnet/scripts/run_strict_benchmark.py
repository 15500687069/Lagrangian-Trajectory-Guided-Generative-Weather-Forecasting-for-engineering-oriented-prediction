from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniltgnet.config import load_config
from miniltgnet.data import build_dataloaders
from miniltgnet.inference import apply_inference_postprocess
from miniltgnet.metrics import spectral_distance
from miniltgnet.model import build_model
from miniltgnet.trainer import move_batch_to_device


METRICS = ["rmse", "acc", "track_mae", "extreme_f1", "spectral_distance"]
LOWER_BETTER = {"rmse", "track_mae", "spectral_distance"}


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


def _load_model(
    cfg: dict[str, Any],
    dataloaders: dict[str, torch.utils.data.DataLoader],
    checkpoint: str,
    device: torch.device,
) -> torch.nn.Module:
    sample = next(iter(dataloaders["train"]))
    in_channels = int(sample["x_hist"].shape[2])
    model = build_model(cfg, in_channels=in_channels).to(device)
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        use_ema = bool(cfg.get("evaluation", {}).get("use_ema_for_eval", True))
        if use_ema and "model_ema" in ckpt and ckpt["model_ema"] is not None:
            state = ckpt["model_ema"]
        else:
            state = ckpt["model"] if "model" in ckpt else ckpt
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _clamp_geo(state: torch.Tensor) -> torch.Tensor:
    lat = state[..., 0].clamp(-90.0, 90.0)
    lon = state[..., 1] % 360.0
    return torch.stack([lat, lon], dim=-1)


def _predict_persistence(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    horizon = int(batch["y_future"].shape[1])
    last_field = batch["x_hist"][:, -1].float()
    last_traj = batch["traj_hist"][:, -1].float()
    field_pred = last_field.unsqueeze(1).repeat(1, horizon, 1, 1, 1)
    traj_pred = last_traj.unsqueeze(1).repeat(1, horizon, 1, 1)
    return field_pred, traj_pred


def _predict_linear(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    horizon = int(batch["y_future"].shape[1])
    x_hist = batch["x_hist"].float()
    traj_hist = batch["traj_hist"].float()
    if x_hist.shape[1] < 2:
        return _predict_persistence(batch)

    prev_field = x_hist[:, -1]
    delta_field = x_hist[:, -1] - x_hist[:, -2]
    fields: list[torch.Tensor] = []
    for _ in range(horizon):
        prev_field = prev_field + delta_field
        fields.append(prev_field)
    field_pred = torch.stack(fields, dim=1)

    prev_traj = traj_hist[:, -1]
    delta_lat = traj_hist[:, -1, :, 0] - traj_hist[:, -2, :, 0]
    delta_lon = (traj_hist[:, -1, :, 1] - traj_hist[:, -2, :, 1] + 180.0) % 360.0 - 180.0
    trajs: list[torch.Tensor] = []
    for _ in range(horizon):
        lat = prev_traj[..., 0] + delta_lat
        lon = prev_traj[..., 1] + delta_lon
        prev_traj = _clamp_geo(torch.stack([lat, lon], dim=-1))
        trajs.append(prev_traj)
    traj_pred = torch.stack(trajs, dim=1)
    return field_pred, traj_pred


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
    out = torch.fft.irfft2(out_f, s=(model_field.shape[-2], model_field.shape[-1]), dim=(-2, -1), norm="ortho")
    return out


def _apply_model_calibration(
    model_field: torch.Tensor,
    model_traj: torch.Tensor,
    batch: dict[str, torch.Tensor],
    params: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not params:
        return model_field, model_traj
    pers_field, pers_traj = _predict_persistence(batch)
    alpha_field = float(params.get("alpha_field", 1.0))
    alpha_traj = float(params.get("alpha_traj", 1.0))
    beta_high = float(params.get("beta_high", 0.0))
    k_ratio = float(params.get("k_ratio", 0.55))

    af = float(max(0.0, min(1.0, alpha_field)))
    field = af * model_field + (1.0 - af) * pers_field
    field = _fft_high_blend(field, pers_field, beta=beta_high, k_ratio=k_ratio)
    traj = _blend_traj(model_traj, pers_traj, alpha_model=alpha_traj)

    # Optional lead-wise trajectory bias correction (fit on calibration split).
    traj_bias_gamma = float(params.get("traj_bias_gamma", 0.0))
    if abs(traj_bias_gamma) > 1e-8 and "traj_bias_lat" in params and "traj_bias_lon" in params:
        bias_lat = torch.as_tensor(params["traj_bias_lat"], device=traj.device, dtype=traj.dtype)
        bias_lon = torch.as_tensor(params["traj_bias_lon"], device=traj.device, dtype=traj.dtype)
        if bias_lat.ndim == 1 and bias_lon.ndim == 1:
            l = min(int(traj.shape[1]), int(bias_lat.numel()), int(bias_lon.numel()))
            if l > 0:
                traj_lat = traj[:, :l, :, 0] + traj_bias_gamma * bias_lat[:l].view(1, l, 1)
                traj_lon = traj[:, :l, :, 1] + traj_bias_gamma * bias_lon[:l].view(1, l, 1)
                traj[:, :l] = _clamp_geo(torch.stack([traj_lat, traj_lon], dim=-1))

    # Optional spectral transfer correction in Fourier amplitude domain.
    spec_gamma = float(params.get("spec_transfer_gamma", 0.0))
    if abs(spec_gamma) > 1e-8 and "spec_transfer_gain" in params:
        gain = torch.as_tensor(params["spec_transfer_gain"], device=field.device, dtype=field.dtype)
        if gain.ndim == 2 and int(gain.shape[0]) == int(field.shape[2]):
            min_k = int(params.get("spec_transfer_min_k", 2))
            gmin = float(params.get("spec_transfer_gain_clip_min", 0.7))
            gmax = float(params.get("spec_transfer_gain_clip_max", 1.4))
            bt, c, h, w = int(field.shape[0] * field.shape[1]), int(field.shape[2]), int(field.shape[3]), int(field.shape[4])
            ff = torch.fft.rfft2(field.reshape(bt, c, h, w), dim=(-2, -1), norm="ortho")
            w2 = ff.shape[-1]
            yy, xx = torch.meshgrid(
                torch.arange(h, device=field.device),
                torch.arange(w2, device=field.device),
                indexing="ij",
            )
            ky = torch.minimum(yy, h - yy).float()
            kx = xx.float()
            kr = torch.sqrt(kx.square() + ky.square()).round().long()
            kmax = int(gain.shape[1] - 1)
            kr_clamped = kr.clamp(max=max(0, kmax))
            gain_map = gain[:, kr_clamped]  # [C,H,W2]
            gain_map = (1.0 + spec_gamma * (gain_map - 1.0)).clamp(min=gmin, max=gmax)
            if min_k > 0:
                low_mask = (kr < int(min_k)).to(gain_map.dtype)
                gain_map = low_mask[None] * torch.ones_like(gain_map) + (1.0 - low_mask[None]) * gain_map
            ff = ff * gain_map[None]
            field = torch.fft.irfft2(ff, s=(h, w), dim=(-2, -1), norm="ortho").reshape_as(field)
    return field, traj


def _parse_float_grid(text: str) -> list[float]:
    out: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


@torch.no_grad()
def _evaluate_calibration_candidate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
    spectral_wavenumbers: int,
    extreme_quantile: float,
    params: dict[str, Any] | None,
    inference_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    meter: dict[str, list[np.ndarray]] = defaultdict(list)
    for step_idx, batch in enumerate(tqdm(dataloader, desc="calib-eval", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        pred_field = out["field_pred"]
        pred_traj = out["traj_pred"]
        pred_field, pred_traj = apply_inference_postprocess(
            pred_field=pred_field,
            pred_traj=pred_traj,
            batch=batch,
            cfg=inference_config,
        )
        pred_field, pred_traj = _apply_model_calibration(pred_field, pred_traj, batch, params)
        tgt_field = batch["y_future"].float()
        tgt_traj = batch["traj_future"].float()
        bundle = _metric_bundle_per_sample(
            pred_field=pred_field,
            target_field=tgt_field,
            pred_traj=pred_traj,
            target_traj=tgt_traj,
            spectral_wavenumbers=spectral_wavenumbers,
            extreme_quantile=extreme_quantile,
        )
        for k, v in bundle.items():
            meter[k].append(v)

    out_mean: dict[str, float] = {}
    for k in METRICS:
        if meter[k]:
            out_mean[k] = float(np.mean(np.concatenate(meter[k], axis=0)))
        else:
            out_mean[k] = float("nan")
    return out_mean


def _fit_model_calibration(
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    calib_split: str,
    device: torch.device,
    max_batches: int,
    spectral_wavenumbers: int,
    extreme_quantile: float,
    alpha_traj_grid: list[float],
    alpha_field_grid: list[float],
    beta_high_grid: list[float],
    k_ratio_grid: list[float],
    w_track: float,
    w_spec: float,
    rmse_tol: float,
    acc_tol: float,
    f1_tol: float,
    inference_config: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float], dict[str, float]]:
    loader = dataloaders[calib_split]
    baseline_params = {"alpha_traj": 1.0, "alpha_field": 1.0, "beta_high": 0.0, "k_ratio": 0.55}
    baseline = _evaluate_calibration_candidate(
        model=model,
        dataloader=loader,
        device=device,
        max_batches=max_batches,
        spectral_wavenumbers=spectral_wavenumbers,
        extreme_quantile=extreme_quantile,
        params=baseline_params,
        inference_config=inference_config,
    )

    rows: list[dict[str, Any]] = []
    best_feasible: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None

    for alpha_traj in alpha_traj_grid:
        for alpha_field in alpha_field_grid:
            for beta_high in beta_high_grid:
                for k_ratio in k_ratio_grid:
                    if float(beta_high) == 0.0 and float(k_ratio) != float(k_ratio_grid[0]):
                        continue
                    params = {
                        "alpha_traj": float(alpha_traj),
                        "alpha_field": float(alpha_field),
                        "beta_high": float(beta_high),
                        "k_ratio": float(k_ratio),
                    }
                    metrics = _evaluate_calibration_candidate(
                        model=model,
                        dataloader=loader,
                        device=device,
                        max_batches=max_batches,
                        spectral_wavenumbers=spectral_wavenumbers,
                        extreme_quantile=extreme_quantile,
                        params=params,
                        inference_config=inference_config,
                    )
                    gain_track = (baseline["track_mae"] - metrics["track_mae"]) / (abs(baseline["track_mae"]) + 1e-8)
                    gain_spec = (baseline["spectral_distance"] - metrics["spectral_distance"]) / (
                        abs(baseline["spectral_distance"]) + 1e-8
                    )
                    objective = float(w_track * gain_track + w_spec * gain_spec)
                    delta_rmse = float(metrics["rmse"] - baseline["rmse"])
                    delta_acc = float(metrics["acc"] - baseline["acc"])
                    delta_f1 = float(metrics["extreme_f1"] - baseline["extreme_f1"])
                    feasible = bool(
                        delta_rmse <= float(rmse_tol)
                        and delta_acc >= -float(acc_tol)
                        and delta_f1 >= -float(f1_tol)
                    )
                    penalty = (
                        max(0.0, delta_rmse - float(rmse_tol)) * 200.0
                        + max(0.0, (-delta_acc) - float(acc_tol)) * 200.0
                        + max(0.0, (-delta_f1) - float(f1_tol)) * 120.0
                    )
                    score_any = float(objective - penalty)

                    row = {
                        **params,
                        **metrics,
                        "gain_track": float(gain_track),
                        "gain_spectral": float(gain_spec),
                        "objective": float(objective),
                        "delta_rmse": float(delta_rmse),
                        "delta_acc": float(delta_acc),
                        "delta_extreme_f1": float(delta_f1),
                        "feasible": feasible,
                        "score_any": score_any,
                    }
                    rows.append(row)
                    if feasible and (best_feasible is None or float(row["objective"]) > float(best_feasible["objective"])):
                        best_feasible = row
                    if best_any is None or float(row["score_any"]) > float(best_any["score_any"]):
                        best_any = row

    chosen = best_feasible if best_feasible is not None else best_any
    if chosen is None:
        raise RuntimeError("Calibration search produced no candidates.")

    chosen_params = {
        "alpha_traj": float(chosen["alpha_traj"]),
        "alpha_field": float(chosen["alpha_field"]),
        "beta_high": float(chosen["beta_high"]),
        "k_ratio": float(chosen["k_ratio"]),
    }
    chosen_metrics = {
        k: float(chosen[k]) for k in METRICS
    }
    return chosen_params, rows, baseline, chosen_metrics


@torch.no_grad()
def _fit_advanced_residual_corrections(
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    calib_split: str,
    device: torch.device,
    max_batches: int,
    base_params: dict[str, float],
    spectral_wavenumbers: int,
    spec_transfer_min_k: int,
    spec_gain_clip_min: float,
    spec_gain_clip_max: float,
    inference_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loader = dataloaders[calib_split]
    traj_sum_lat: torch.Tensor | None = None
    traj_sum_lon: torch.Tensor | None = None
    traj_count = 0.0

    max_k = int(max(1, spectral_wavenumbers))
    sum_log_pred: torch.Tensor | None = None
    sum_log_tgt: torch.Tensor | None = None
    cnt_k: torch.Tensor | None = None

    for step_idx, batch in enumerate(tqdm(loader, desc="calib-advanced-fit", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        pred_field = out["field_pred"]
        pred_traj = out["traj_pred"]
        pred_field, pred_traj = apply_inference_postprocess(
            pred_field=pred_field,
            pred_traj=pred_traj,
            batch=batch,
            cfg=inference_config,
        )
        pred_field, pred_traj = _apply_model_calibration(pred_field, pred_traj, batch, base_params)
        tgt_field = batch["y_future"].float()
        tgt_traj = batch["traj_future"].float()

        # Trajectory lead-wise bias.
        err_lat = (tgt_traj[..., 0] - pred_traj[..., 0])  # [B,T,O]
        err_lon = (tgt_traj[..., 1] - pred_traj[..., 1] + 180.0) % 360.0 - 180.0
        batch_sum_lat = err_lat.sum(dim=(0, 2))
        batch_sum_lon = err_lon.sum(dim=(0, 2))
        if traj_sum_lat is None:
            traj_sum_lat = batch_sum_lat
            traj_sum_lon = batch_sum_lon
        else:
            traj_sum_lat = traj_sum_lat + batch_sum_lat
            traj_sum_lon = traj_sum_lon + batch_sum_lon
        traj_count += float(err_lat.shape[0] * err_lat.shape[2])

        # Spectral transfer gain by channel and radial wavenumber.
        b, t, c, h, w = pred_field.shape
        pred4 = pred_field.reshape(b * t, c, h, w)
        tgt4 = tgt_field.reshape(b * t, c, h, w)
        pf = torch.fft.rfft2(pred4, dim=(-2, -1), norm="ortho")
        tf = torch.fft.rfft2(tgt4, dim=(-2, -1), norm="ortho")
        pp = pf.real.square() + pf.imag.square()
        tp = tf.real.square() + tf.imag.square()
        w2 = pp.shape[-1]

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w2, device=device),
            indexing="ij",
        )
        ky = torch.minimum(yy, h - yy).float()
        kx = xx.float()
        kr = torch.sqrt(kx.square() + ky.square()).round().long()

        if sum_log_pred is None:
            sum_log_pred = torch.zeros(c, max_k + 1, device=device, dtype=torch.float64)
            sum_log_tgt = torch.zeros(c, max_k + 1, device=device, dtype=torch.float64)
            cnt_k = torch.zeros(max_k + 1, device=device, dtype=torch.float64)

        for k in range(max_k + 1):
            mask = (kr == k)
            if not mask.any():
                continue
            pred_band = pp[:, :, mask].mean(dim=-1).mean(dim=0)  # [C]
            tgt_band = tp[:, :, mask].mean(dim=-1).mean(dim=0)  # [C]
            sum_log_pred[:, k] += torch.log1p(pred_band.double())
            sum_log_tgt[:, k] += torch.log1p(tgt_band.double())
            cnt_k[k] += 1.0

    if traj_sum_lat is None or traj_sum_lon is None or traj_count <= 0:
        raise RuntimeError("Advanced calibration failed: no trajectory samples.")
    if sum_log_pred is None or sum_log_tgt is None or cnt_k is None:
        raise RuntimeError("Advanced calibration failed: no spectral samples.")

    bias_lat = (traj_sum_lat / traj_count).detach().cpu().tolist()
    bias_lon = (traj_sum_lon / traj_count).detach().cpu().tolist()

    cnt = cnt_k.clamp_min(1.0)[None, :]
    gain = torch.exp((sum_log_tgt / cnt) - (sum_log_pred / cnt)).float()
    gain = gain.clamp(min=float(spec_gain_clip_min), max=float(spec_gain_clip_max))
    min_k = int(max(0, spec_transfer_min_k))
    if min_k > 0:
        gain[:, :min_k] = 1.0

    return {
        "traj_bias_lat": bias_lat,
        "traj_bias_lon": bias_lon,
        "spec_transfer_gain": gain.detach().cpu().tolist(),
        "spec_transfer_min_k": int(min_k),
        "spec_transfer_gain_clip_min": float(spec_gain_clip_min),
        "spec_transfer_gain_clip_max": float(spec_gain_clip_max),
    }


def _build_climatology(
    train_loader: torch.utils.data.DataLoader,
    max_batches: int = 0,
) -> torch.Tensor:
    count = 0
    sum_y: torch.Tensor | None = None
    for step_idx, batch in enumerate(train_loader):
        if max_batches > 0 and step_idx >= max_batches:
            break
        y = batch["y_future"].float()  # [B,T,C,H,W]
        batch_sum = y.sum(dim=0)  # [T,C,H,W]
        if sum_y is None:
            sum_y = batch_sum
        else:
            sum_y = sum_y + batch_sum
        count += int(y.shape[0])
    if sum_y is None or count == 0:
        raise RuntimeError("Failed to build climatology: empty training loader.")
    return (sum_y / float(count)).contiguous()


def _safe_quantile_1d(values: torch.Tensor, q: float, max_elements: int = 1_000_000) -> torch.Tensor:
    flat = values.reshape(-1)
    if max_elements > 0 and flat.numel() > max_elements:
        step = max(1, flat.numel() // max_elements)
        flat = flat[::step]
        if flat.numel() > max_elements:
            flat = flat[:max_elements]
    if not torch.is_floating_point(flat) or flat.dtype in {torch.float16, torch.bfloat16}:
        flat = flat.float()
    q = min(1.0, max(0.0, float(q)))
    return torch.quantile(flat, q)


def _rmse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    dims = tuple(range(1, pred.ndim))
    values = torch.sqrt(torch.mean((pred - target) ** 2, dim=dims))
    return values.detach().cpu().numpy().astype(np.float64)


def _acc_per_sample(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred_anom = pred - pred.mean(dim=(-2, -1), keepdim=True)
    tgt_anom = target - target.mean(dim=(-2, -1), keepdim=True)
    num = torch.sum(pred_anom * tgt_anom, dim=(-2, -1))
    den = torch.sqrt(torch.sum(pred_anom**2, dim=(-2, -1)) * torch.sum(tgt_anom**2, dim=(-2, -1)) + 1e-8)
    ratio = num / den
    dims = tuple(range(1, ratio.ndim))
    values = ratio if not dims else ratio.mean(dim=dims)
    return values.detach().cpu().numpy().astype(np.float64)


def _track_mae_per_sample(pred_traj: torch.Tensor, target_traj: torch.Tensor) -> np.ndarray:
    lat_err = torch.abs(pred_traj[..., 0] - target_traj[..., 0])
    lon_diff = (pred_traj[..., 1] - target_traj[..., 1] + 180.0) % 360.0 - 180.0
    lon_err = torch.abs(lon_diff)
    dims = tuple(range(1, lat_err.ndim))
    lat_mean = lat_err if not dims else lat_err.mean(dim=dims)
    lon_mean = lon_err if not dims else lon_err.mean(dim=dims)
    values = 0.5 * (lat_mean + lon_mean)
    return values.detach().cpu().numpy().astype(np.float64)


def _extreme_f1_per_sample(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> np.ndarray:
    b = int(pred.shape[0])
    out = np.zeros(b, dtype=np.float64)
    for i in range(b):
        p = pred[i]
        t = target[i]
        threshold = _safe_quantile_1d(t, q=quantile, max_elements=1_000_000)
        pred_evt = p >= threshold
        tgt_evt = t >= threshold
        tp = (pred_evt & tgt_evt).sum().float()
        fp = (pred_evt & ~tgt_evt).sum().float()
        fn = (~pred_evt & tgt_evt).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        out[i] = float(f1.detach().cpu().item())
    return out


def _spectral_distance_per_sample(
    pred_field: torch.Tensor,
    target_field: torch.Tensor,
    max_wavenumber: int,
) -> np.ndarray:
    b = int(pred_field.shape[0])
    out = np.zeros(b, dtype=np.float64)
    for i in range(b):
        if pred_field.ndim == 5:
            # [B,T,C,H,W] -> [1,T*C,H,W]
            _, t, c, h, w = pred_field.shape
            pf = pred_field[i].reshape(1, t * c, h, w)
            tf = target_field[i].reshape(1, t * c, h, w)
        else:
            # [B,C,H,W] -> [1,C,H,W]
            pf = pred_field[i : i + 1]
            tf = target_field[i : i + 1]
        sd = spectral_distance(pf, tf, max_wavenumber=max_wavenumber)
        out[i] = float(sd.detach().cpu().item())
    return out


def _metric_bundle_per_sample(
    pred_field: torch.Tensor,
    target_field: torch.Tensor,
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    spectral_wavenumbers: int,
    extreme_quantile: float,
) -> dict[str, np.ndarray]:
    return {
        "rmse": _rmse_per_sample(pred_field, target_field),
        "acc": _acc_per_sample(pred_field, target_field),
        "track_mae": _track_mae_per_sample(pred_traj, target_traj),
        "extreme_f1": _extreme_f1_per_sample(pred_field, target_field, quantile=extreme_quantile),
        "spectral_distance": _spectral_distance_per_sample(
            pred_field=pred_field,
            target_field=target_field,
            max_wavenumber=spectral_wavenumbers,
        ),
    }


def _concat_samples(samples: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k in METRICS:
        chunks = samples.get(k, [])
        if not chunks:
            out[k] = np.zeros(0, dtype=np.float64)
        else:
            out[k] = np.concatenate(chunks, axis=0).astype(np.float64)
    return out


@torch.no_grad()
def _collect_method_samples(
    method: str,
    predict_fn: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    leads: list[int],
    max_batches: int,
    spectral_wavenumbers: int,
    extreme_quantile: float,
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    overall_lists: dict[str, list[np.ndarray]] = defaultdict(list)
    lead_lists: dict[int, dict[str, list[np.ndarray]]] = {
        int(lead): defaultdict(list) for lead in leads
    }

    for step_idx, batch in enumerate(tqdm(dataloader, desc=f"strict:{method}", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        pred_field, pred_traj = predict_fn(batch)
        target_field = batch["y_future"].float()
        target_traj = batch["traj_future"].float()

        overall = _metric_bundle_per_sample(
            pred_field=pred_field,
            target_field=target_field,
            pred_traj=pred_traj,
            target_traj=target_traj,
            spectral_wavenumbers=spectral_wavenumbers,
            extreme_quantile=extreme_quantile,
        )
        for k, v in overall.items():
            overall_lists[k].append(v)

        horizon = int(target_field.shape[1])
        for lead in leads:
            li = int(lead) - 1
            if li < 0 or li >= horizon:
                continue
            pf = pred_field[:, li]
            tf = target_field[:, li]
            pt = pred_traj[:, li]
            tt = target_traj[:, li]
            lead_metrics = _metric_bundle_per_sample(
                pred_field=pf,
                target_field=tf,
                pred_traj=pt,
                target_traj=tt,
                spectral_wavenumbers=spectral_wavenumbers,
                extreme_quantile=extreme_quantile,
            )
            for k, v in lead_metrics.items():
                lead_lists[int(lead)][k].append(v)

    overall_out = _concat_samples(overall_lists)
    lead_out: dict[int, dict[str, np.ndarray]] = {}
    for lead in leads:
        lead_out[int(lead)] = _concat_samples(lead_lists[int(lead)])
    return overall_out, lead_out


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1 or n_bootstrap <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(n_bootstrap), arr.size))
    means = np.mean(arr[idx], axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return mean, low, high


def _paired_improvement_stats(
    method_values: np.ndarray,
    ref_values: np.ndarray,
    metric: str,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> dict[str, float]:
    m = np.asarray(method_values, dtype=np.float64).reshape(-1)
    r = np.asarray(ref_values, dtype=np.float64).reshape(-1)
    n = int(min(m.size, r.size))
    if n == 0:
        return {
            "delta_better": float("nan"),
            "rel_gain_pct": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_better": float("nan"),
        }
    m = m[:n]
    r = r[:n]
    if metric in LOWER_BETTER:
        diffs = r - m  # > 0 means method better
        rel_gain_pct = float(((np.mean(r) - np.mean(m)) / (abs(np.mean(r)) + 1e-8)) * 100.0)
    else:
        diffs = m - r  # > 0 means method better
        rel_gain_pct = float(((np.mean(m) - np.mean(r)) / (abs(np.mean(r)) + 1e-8)) * 100.0)

    delta = float(np.mean(diffs))
    if n == 1 or n_bootstrap <= 1:
        return {
            "delta_better": delta,
            "rel_gain_pct": rel_gain_pct,
            "ci_low": delta,
            "ci_high": delta,
            "p_better": float(1.0 if delta > 0 else 0.0),
        }

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    boot_means = np.mean(diffs[idx], axis=1)
    ci_low = float(np.quantile(boot_means, alpha / 2.0))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    p_better = float(np.mean(boot_means > 0.0))
    return {
        "delta_better": delta,
        "rel_gain_pct": rel_gain_pct,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_better": p_better,
    }


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_report(
    config_path: str,
    split: str,
    leads: list[int],
    methods: list[str],
    n_samples: int,
    reference_method: str,
    p_threshold: float,
    summary_rows: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# MiniLTG Strict Benchmark Report")
    lines.append("")
    lines.append("## Protocol")
    lines.append(f"- config: `{config_path}`")
    lines.append(f"- split: `{split}`")
    lines.append(f"- lead_steps: `{leads}`")
    lines.append(f"- methods: `{methods}`")
    lines.append(f"- samples: `{n_samples}`")
    lines.append(f"- reference: `{reference_method}`")
    lines.append("")
    lines.append("## Overall Summary")
    for row in summary_rows:
        lines.append(
            f"- `{row['method']}`: rmse={row['rmse']:.6f}, acc={row['acc']:.6f}, "
            f"extreme_f1={row['extreme_f1']:.6f}, track_mae={row['track_mae']:.6f}, "
            f"spectral_distance={row['spectral_distance']:.6f}"
        )

    by_method = {r["method"]: r for r in summary_rows}
    if "model" in by_method and reference_method in by_method:
        m = by_method["model"]
        r = by_method[reference_method]
        mean_gate = (
            float(m["rmse"]) < float(r["rmse"])
            and float(m["acc"]) > float(r["acc"])
            and float(m["extreme_f1"]) >= float(r["extreme_f1"])
        )

        skill_lookup = {(x["method"], x["metric"]): x for x in skill_rows}
        rmse_sig = skill_lookup.get(("model", "rmse"), {}).get("p_better", float("nan"))
        acc_sig = skill_lookup.get(("model", "acc"), {}).get("p_better", float("nan"))
        f1_sig = skill_lookup.get(("model", "extreme_f1"), {}).get("p_better", float("nan"))
        strict_gate = (
            mean_gate
            and (not np.isnan(rmse_sig) and float(rmse_sig) >= p_threshold)
            and (not np.isnan(acc_sig) and float(acc_sig) >= p_threshold)
            and (not np.isnan(f1_sig) and float(f1_sig) >= p_threshold)
        )

        lines.append("")
        lines.append("## Gate")
        lines.append(
            f"- mean-rule: `model rmse < {reference_method} rmse` and "
            f"`model acc > {reference_method} acc` and "
            f"`model extreme_f1 >= {reference_method} extreme_f1`"
        )
        lines.append(f"- mean-result: **{'PASS' if mean_gate else 'FAIL'}**")
        lines.append(f"- strict p-threshold: `{p_threshold:.2f}`")
        lines.append(
            f"- p_better: rmse={float(rmse_sig):.3f}, acc={float(acc_sig):.3f}, extreme_f1={float(f1_sig):.3f}"
        )
        lines.append(f"- strict-result: **{'PASS' if strict_gate else 'FAIL'}**")

    lines.append("")
    lines.append("## Notes")
    lines.append("- This is an internal strict benchmark under a fixed local protocol.")
    lines.append("- Claiming SOTA requires public-dataset, public-protocol, and peer baseline parity.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict benchmark for MiniLTG with CIs and paired significance.")
    parser.add_argument("--config", required=True, help="MiniLTG config path.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path for model method.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["model", "persistence", "linear", "climatology"],
        choices=["model", "persistence", "linear", "climatology"],
    )
    parser.add_argument("--leads", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], help="Lead steps.")
    parser.add_argument("--max_batches", type=int, default=0, help="Limit eval batches (0=all).")
    parser.add_argument("--climatology_max_batches", type=int, default=0, help="Train batches for climatology.")
    parser.add_argument("--extreme_quantile", type=float, default=0.99, help="Extreme-event quantile.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap samples for CI/stat tests.")
    parser.add_argument("--ci_alpha", type=float, default=0.05, help="CI alpha.")
    parser.add_argument("--reference", default="persistence", choices=["persistence", "linear", "climatology"])
    parser.add_argument("--strict_p_threshold", type=float, default=0.95, help="Gate p_better threshold.")
    parser.add_argument(
        "--calibrate_on_split",
        default="none",
        choices=["none", "train", "val", "test"],
        help="Search model calibration parameters on this split (no retraining).",
    )
    parser.add_argument(
        "--calibration_json",
        default="",
        help="Use existing calibration JSON (basic + optional advanced residual fields).",
    )
    parser.add_argument("--calib_alpha_traj_grid", default="1.0,0.9,0.8,0.7,0.6")
    parser.add_argument("--calib_alpha_field_grid", default="1.0,0.995,0.99")
    parser.add_argument("--calib_beta_high_grid", default="0.0,0.08,0.16,0.24")
    parser.add_argument("--calib_k_ratio_grid", default="0.45,0.55,0.65")
    parser.add_argument("--calib_w_track", type=float, default=0.55)
    parser.add_argument("--calib_w_spec", type=float, default=0.45)
    parser.add_argument("--calib_rmse_tol", type=float, default=0.0015)
    parser.add_argument("--calib_acc_tol", type=float, default=0.0015)
    parser.add_argument("--calib_f1_tol", type=float, default=0.0020)
    parser.add_argument("--calib_max_batches", type=int, default=0, help="Max batches for calibration split (0=use max_batches).")
    parser.add_argument(
        "--calib_enable_advanced",
        action="store_true",
        help="Enable advanced residual calibration (trajectory bias + spectral transfer).",
    )
    parser.add_argument("--calib_traj_bias_gamma_grid", default="0.0,0.5,1.0")
    parser.add_argument("--calib_spec_transfer_gamma_grid", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--calib_spec_transfer_min_k", type=int, default=2)
    parser.add_argument("--calib_spec_gain_clip_min", type=float, default=0.70)
    parser.add_argument("--calib_spec_gain_clip_max", type=float, default=1.40)
    parser.add_argument("--output_dir", default="outputs/miniltg/strict_benchmark")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    seed = int(cfg["experiment"]["seed"])
    _seed_everything(seed, deterministic=bool(cfg["experiment"].get("deterministic", False)))
    device = _resolve_device(cfg)
    dataloaders = build_dataloaders(cfg)
    spectral_wavenumbers = int(cfg.get("evaluation", {}).get("spectral_wavenumbers", 32))
    leads = sorted({int(x) for x in args.leads if int(x) > 0})
    methods = list(dict.fromkeys(args.methods))

    predictors: dict[str, Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]] = {}
    model_calibration: dict[str, Any] | None = None
    if "model" in methods:
        if not args.checkpoint:
            raise ValueError("`--checkpoint` is required when method includes `model`.")
        model = _load_model(cfg, dataloaders, args.checkpoint, device)
        if args.calibration_json:
            cpath = Path(args.calibration_json)
            if not cpath.exists():
                raise FileNotFoundError(f"Calibration JSON not found: {cpath}")
            with cpath.open("r", encoding="utf-8") as f:
                cobj = json.load(f)
            if "chosen_params" in cobj:
                cobj = cobj["chosen_params"]
            model_calibration = {
                "alpha_traj": float(cobj.get("alpha_traj", 1.0)),
                "alpha_field": float(cobj.get("alpha_field", 1.0)),
                "beta_high": float(cobj.get("beta_high", 0.0)),
                "k_ratio": float(cobj.get("k_ratio", 0.55)),
            }
            for k in ["traj_bias_lat", "traj_bias_lon", "spec_transfer_gain"]:
                if k in cobj and cobj[k] is not None:
                    model_calibration[k] = cobj[k]
            for k in ["traj_bias_gamma", "spec_transfer_gamma", "spec_transfer_gain_clip_min", "spec_transfer_gain_clip_max"]:
                if k in cobj and cobj[k] is not None:
                    model_calibration[k] = float(cobj[k])
            if "spec_transfer_min_k" in cobj and cobj["spec_transfer_min_k"] is not None:
                model_calibration["spec_transfer_min_k"] = int(cobj["spec_transfer_min_k"])
        elif args.calibrate_on_split != "none":
            alpha_traj_grid = _parse_float_grid(args.calib_alpha_traj_grid)
            alpha_field_grid = _parse_float_grid(args.calib_alpha_field_grid)
            beta_high_grid = _parse_float_grid(args.calib_beta_high_grid)
            k_ratio_grid = _parse_float_grid(args.calib_k_ratio_grid)
            if not alpha_traj_grid or not alpha_field_grid or not beta_high_grid or not k_ratio_grid:
                raise ValueError("Calibration grids must be non-empty.")
            calib_max_batches = int(args.calib_max_batches) if int(args.calib_max_batches) > 0 else int(args.max_batches)
            chosen_params, calib_rows, calib_baseline, calib_chosen = _fit_model_calibration(
                model=model,
                dataloaders=dataloaders,
                calib_split=args.calibrate_on_split,
                device=device,
                max_batches=calib_max_batches,
                spectral_wavenumbers=spectral_wavenumbers,
                extreme_quantile=float(args.extreme_quantile),
                alpha_traj_grid=alpha_traj_grid,
                alpha_field_grid=alpha_field_grid,
                beta_high_grid=beta_high_grid,
                k_ratio_grid=k_ratio_grid,
                w_track=float(args.calib_w_track),
                w_spec=float(args.calib_w_spec),
                rmse_tol=float(args.calib_rmse_tol),
                acc_tol=float(args.calib_acc_tol),
                f1_tol=float(args.calib_f1_tol),
                inference_config=cfg,
            )
            model_calibration = dict(chosen_params)

            advanced_rows: list[dict[str, Any]] = []
            advanced_payload: dict[str, Any] | None = None
            if bool(args.calib_enable_advanced):
                traj_gamma_grid = _parse_float_grid(args.calib_traj_bias_gamma_grid)
                spec_gamma_grid = _parse_float_grid(args.calib_spec_transfer_gamma_grid)
                if not traj_gamma_grid or not spec_gamma_grid:
                    raise ValueError("Advanced calibration gamma grids must be non-empty.")

                adv_fit = _fit_advanced_residual_corrections(
                    model=model,
                    dataloaders=dataloaders,
                    calib_split=args.calibrate_on_split,
                    device=device,
                    max_batches=calib_max_batches,
                    base_params=chosen_params,
                    spectral_wavenumbers=spectral_wavenumbers,
                    spec_transfer_min_k=int(args.calib_spec_transfer_min_k),
                    spec_gain_clip_min=float(args.calib_spec_gain_clip_min),
                    spec_gain_clip_max=float(args.calib_spec_gain_clip_max),
                    inference_config=cfg,
                )

                best_adv_feasible: dict[str, Any] | None = None
                best_adv_any: dict[str, Any] | None = None
                for traj_gamma in traj_gamma_grid:
                    for spec_gamma in spec_gamma_grid:
                        params = {
                            **chosen_params,
                            **adv_fit,
                            "traj_bias_gamma": float(traj_gamma),
                            "spec_transfer_gamma": float(spec_gamma),
                        }
                        metrics = _evaluate_calibration_candidate(
                            model=model,
                            dataloader=dataloaders[args.calibrate_on_split],
                            device=device,
                            max_batches=calib_max_batches,
                            spectral_wavenumbers=spectral_wavenumbers,
                            extreme_quantile=float(args.extreme_quantile),
                            params=params,
                            inference_config=cfg,
                        )
                        gain_track = (calib_baseline["track_mae"] - metrics["track_mae"]) / (
                            abs(calib_baseline["track_mae"]) + 1e-8
                        )
                        gain_spec = (calib_baseline["spectral_distance"] - metrics["spectral_distance"]) / (
                            abs(calib_baseline["spectral_distance"]) + 1e-8
                        )
                        objective = float(float(args.calib_w_track) * gain_track + float(args.calib_w_spec) * gain_spec)
                        delta_rmse = float(metrics["rmse"] - calib_baseline["rmse"])
                        delta_acc = float(metrics["acc"] - calib_baseline["acc"])
                        delta_f1 = float(metrics["extreme_f1"] - calib_baseline["extreme_f1"])
                        feasible = bool(
                            delta_rmse <= float(args.calib_rmse_tol)
                            and delta_acc >= -float(args.calib_acc_tol)
                            and delta_f1 >= -float(args.calib_f1_tol)
                        )
                        penalty = (
                            max(0.0, delta_rmse - float(args.calib_rmse_tol)) * 200.0
                            + max(0.0, (-delta_acc) - float(args.calib_acc_tol)) * 200.0
                            + max(0.0, (-delta_f1) - float(args.calib_f1_tol)) * 120.0
                        )
                        score_any = float(objective - penalty)

                        row = {
                            "traj_bias_gamma": float(traj_gamma),
                            "spec_transfer_gamma": float(spec_gamma),
                            **metrics,
                            "gain_track": float(gain_track),
                            "gain_spectral": float(gain_spec),
                            "objective": float(objective),
                            "delta_rmse": float(delta_rmse),
                            "delta_acc": float(delta_acc),
                            "delta_extreme_f1": float(delta_f1),
                            "feasible": feasible,
                            "score_any": score_any,
                        }
                        advanced_rows.append(row)
                        if feasible and (
                            best_adv_feasible is None or float(row["objective"]) > float(best_adv_feasible["objective"])
                        ):
                            best_adv_feasible = row
                        if best_adv_any is None or float(row["score_any"]) > float(best_adv_any["score_any"]):
                            best_adv_any = row

                adv_choice = best_adv_feasible if best_adv_feasible is not None else best_adv_any
                if adv_choice is None:
                    raise RuntimeError("Advanced calibration search produced no candidates.")

                model_calibration = {
                    **chosen_params,
                    **adv_fit,
                    "traj_bias_gamma": float(adv_choice["traj_bias_gamma"]),
                    "spec_transfer_gamma": float(adv_choice["spec_transfer_gamma"]),
                }
                calib_chosen = {k: float(adv_choice[k]) for k in METRICS}
                advanced_payload = {
                    "enabled": True,
                    "fitted_residuals": adv_fit,
                    "selected": {
                        "traj_bias_gamma": float(adv_choice["traj_bias_gamma"]),
                        "spec_transfer_gamma": float(adv_choice["spec_transfer_gamma"]),
                    },
                    "used_feasible_solution": bool(best_adv_feasible is not None),
                }

            calib_csv = out_dir / "model_calibration_search.csv"
            _save_csv(calib_csv, calib_rows)
            if advanced_rows:
                calib_adv_csv = out_dir / "model_calibration_advanced_search.csv"
                _save_csv(calib_adv_csv, advanced_rows)
            calib_json = out_dir / "model_calibration.json"
            with calib_json.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "calibrate_on_split": args.calibrate_on_split,
                        "chosen_params": model_calibration,
                        "baseline_metrics": calib_baseline,
                        "chosen_metrics": calib_chosen,
                        "weights": {"track": float(args.calib_w_track), "spectral": float(args.calib_w_spec)},
                        "constraints": {
                            "rmse_tol": float(args.calib_rmse_tol),
                            "acc_tol": float(args.calib_acc_tol),
                            "f1_tol": float(args.calib_f1_tol),
                        },
                        "grids": {
                            "alpha_traj": alpha_traj_grid,
                            "alpha_field": alpha_field_grid,
                            "beta_high": beta_high_grid,
                            "k_ratio": k_ratio_grid,
                        },
                        "advanced": advanced_payload,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Saved calibration search CSV: {calib_csv}")
            if advanced_rows:
                print(f"Saved advanced calibration CSV: {calib_adv_csv}")
            print(f"Saved calibration JSON: {calib_json}")

        @torch.no_grad()
        def _predict_model(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            out = model(batch)
            pred_field = out["field_pred"]
            pred_traj = out["traj_pred"]
            pred_field, pred_traj = apply_inference_postprocess(
                pred_field=pred_field,
                pred_traj=pred_traj,
                batch=batch,
                cfg=cfg,
            )
            pred_field, pred_traj = _apply_model_calibration(
                model_field=pred_field,
                model_traj=pred_traj,
                batch=batch,
                params=model_calibration,
            )
            return pred_field, pred_traj

        predictors["model"] = _predict_model

    if "persistence" in methods:
        predictors["persistence"] = _predict_persistence
    if "linear" in methods:
        predictors["linear"] = _predict_linear
    if "climatology" in methods:
        clim = _build_climatology(dataloaders["train"], max_batches=int(args.climatology_max_batches)).to(device)

        def _predict_climatology(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            b = int(batch["y_future"].shape[0])
            field_pred = clim.unsqueeze(0).expand(b, -1, -1, -1, -1).contiguous()
            traj_pred = batch["traj_hist"][:, -1].unsqueeze(1).repeat(1, int(clim.shape[0]), 1, 1)
            return field_pred, traj_pred

        predictors["climatology"] = _predict_climatology

    method_samples: dict[str, dict[str, np.ndarray]] = {}
    method_lead_samples: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for method in methods:
        overall_samples, lead_samples = _collect_method_samples(
            method=method,
            predict_fn=predictors[method],
            dataloader=dataloaders[args.split],
            device=device,
            leads=leads,
            max_batches=int(args.max_batches),
            spectral_wavenumbers=spectral_wavenumbers,
            extreme_quantile=float(args.extreme_quantile),
        )
        method_samples[method] = overall_samples
        method_lead_samples[method] = lead_samples

    summary_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    n_samples = 0
    for method in methods:
        row = {"method": method, "split": args.split}
        for metric in METRICS:
            values = method_samples[method][metric]
            n_samples = max(n_samples, int(values.size))
            mean, low, high = _bootstrap_mean_ci(
                values=values,
                n_bootstrap=int(args.bootstrap),
                alpha=float(args.ci_alpha),
                seed=seed + hash((method, metric)) % 10_000,
            )
            row[metric] = mean
            ci_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n": int(values.size),
                }
            )
        summary_rows.append(row)

    lead_rows: list[dict[str, Any]] = []
    for method in methods:
        for lead in leads:
            row = {"method": method, "split": args.split, "lead": int(lead)}
            for metric in METRICS:
                values = method_lead_samples[method][int(lead)][metric]
                row[metric] = float(np.mean(values)) if values.size > 0 else float("nan")
            lead_rows.append(row)

    ref = args.reference
    if ref not in method_samples:
        raise ValueError(f"Reference method `{ref}` not included in methods: {methods}")

    skill_rows: list[dict[str, Any]] = []
    for method in methods:
        if method == ref:
            continue
        for metric in METRICS:
            stats = _paired_improvement_stats(
                method_values=method_samples[method][metric],
                ref_values=method_samples[ref][metric],
                metric=metric,
                n_bootstrap=int(args.bootstrap),
                alpha=float(args.ci_alpha),
                seed=seed + hash((method, metric, "skill")) % 10_000,
            )
            skill_rows.append(
                {
                    "method": method,
                    "reference": ref,
                    "metric": metric,
                    "delta_better": stats["delta_better"],
                    "rel_gain_pct": stats["rel_gain_pct"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "p_better": stats["p_better"],
                }
            )

    lead_skill_rows: list[dict[str, Any]] = []
    for method in methods:
        if method == ref:
            continue
        for lead in leads:
            for metric in METRICS:
                stats = _paired_improvement_stats(
                    method_values=method_lead_samples[method][int(lead)][metric],
                    ref_values=method_lead_samples[ref][int(lead)][metric],
                    metric=metric,
                    n_bootstrap=max(500, int(args.bootstrap) // 2),
                    alpha=float(args.ci_alpha),
                    seed=seed + hash((method, lead, metric, "lead_skill")) % 10_000,
                )
                lead_skill_rows.append(
                    {
                        "method": method,
                        "reference": ref,
                        "lead": int(lead),
                        "metric": metric,
                        "delta_better": stats["delta_better"],
                        "rel_gain_pct": stats["rel_gain_pct"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "p_better": stats["p_better"],
                    }
                )

    summary_csv = out_dir / "strict_summary.csv"
    ci_csv = out_dir / "strict_ci.csv"
    lead_csv = out_dir / "strict_lead_summary.csv"
    skill_csv = out_dir / f"strict_skill_vs_{ref}.csv"
    lead_skill_csv = out_dir / f"strict_lead_skill_vs_{ref}.csv"
    result_json = out_dir / "strict_results.json"
    report_md = out_dir / "strict_report.md"

    _save_csv(summary_csv, summary_rows)
    _save_csv(ci_csv, ci_rows)
    _save_csv(lead_csv, lead_rows)
    _save_csv(skill_csv, skill_rows)
    _save_csv(lead_skill_csv, lead_skill_rows)

    with result_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "split": args.split,
                "leads": leads,
                "methods": methods,
                "reference": ref,
                "n_samples": n_samples,
                "model_calibration": model_calibration,
                "bootstrap": int(args.bootstrap),
                "ci_alpha": float(args.ci_alpha),
                "extreme_quantile": float(args.extreme_quantile),
                "summary": summary_rows,
                "ci": ci_rows,
                "lead_summary": lead_rows,
                "skill_vs_reference": skill_rows,
                "lead_skill_vs_reference": lead_skill_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    report_md.write_text(
        _build_report(
            config_path=args.config,
            split=args.split,
            leads=leads,
            methods=methods,
            n_samples=n_samples,
            reference_method=ref,
            p_threshold=float(args.strict_p_threshold),
            summary_rows=summary_rows,
            skill_rows=skill_rows,
        ),
        encoding="utf-8",
    )

    print(f"Device: {device}")
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


if __name__ == "__main__":
    main()
