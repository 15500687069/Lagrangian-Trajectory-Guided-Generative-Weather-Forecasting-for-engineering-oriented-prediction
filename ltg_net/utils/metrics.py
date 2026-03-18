from __future__ import annotations

import torch

from ltg_net.utils.spectra import spectral_distance


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def acc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_anom = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target_anom = target - target.mean(dim=(-2, -1), keepdim=True)
    numerator = torch.sum(pred_anom * target_anom, dim=(-2, -1))
    denominator = torch.sqrt(
        torch.sum(pred_anom**2, dim=(-2, -1)) * torch.sum(target_anom**2, dim=(-2, -1)) + 1e-8
    )
    return (numerator / denominator).mean()


def track_mae(pred_traj: torch.Tensor, target_traj: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred_traj - target_traj))


def _safe_quantile(values: torch.Tensor, quantile: float, max_elements: int) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return torch.tensor(float("nan"), device=values.device, dtype=torch.float32)

    q = min(1.0, max(0.0, float(quantile)))
    if max_elements > 0 and flat.numel() > max_elements:
        stride = max(1, flat.numel() // max_elements)
        flat = flat[::stride]
        if flat.numel() > max_elements:
            flat = flat[:max_elements]

    if not torch.is_floating_point(flat) or flat.dtype in {torch.float16, torch.bfloat16}:
        flat = flat.float()
    return torch.quantile(flat, q)


def extreme_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.98,
    max_quantile_elements: int = 5_000_000,
) -> torch.Tensor:
    thresh = _safe_quantile(target, quantile, max_quantile_elements)
    pred_evt = pred >= thresh
    tgt_evt = target >= thresh
    tp = (pred_evt & tgt_evt).sum().float()
    fp = (pred_evt & ~tgt_evt).sum().float()
    fn = (~pred_evt & tgt_evt).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2.0 * precision * recall / (precision + recall + 1e-8)


def spectral_metric(pred: torch.Tensor, target: torch.Tensor, max_wavenumber: int = 64) -> torch.Tensor:
    return spectral_distance(pred, target, max_wavenumber=max_wavenumber)
