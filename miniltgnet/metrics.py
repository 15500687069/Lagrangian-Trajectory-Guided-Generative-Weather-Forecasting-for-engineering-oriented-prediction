from __future__ import annotations

import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def acc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_anom = pred - pred.mean(dim=(-2, -1), keepdim=True)
    tgt_anom = target - target.mean(dim=(-2, -1), keepdim=True)
    num = torch.sum(pred_anom * tgt_anom, dim=(-2, -1))
    den = torch.sqrt(torch.sum(pred_anom**2, dim=(-2, -1)) * torch.sum(tgt_anom**2, dim=(-2, -1)) + 1e-8)
    return (num / den).mean()


def track_mae(pred_traj: torch.Tensor, target_traj: torch.Tensor) -> torch.Tensor:
    lat_err = torch.abs(pred_traj[..., 0] - target_traj[..., 0])
    lon_diff = (pred_traj[..., 1] - target_traj[..., 1] + 180.0) % 360.0 - 180.0
    lon_err = torch.abs(lon_diff)
    return 0.5 * (lat_err.mean() + lon_err.mean())


def _safe_quantile(values: torch.Tensor, q: float, max_elements: int = 1_000_000) -> torch.Tensor:
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


def extreme_f1(pred: torch.Tensor, target: torch.Tensor, quantile: float = 0.99) -> torch.Tensor:
    threshold = _safe_quantile(target, q=quantile, max_elements=1_000_000)
    pred_evt = pred >= threshold
    tgt_evt = target >= threshold
    tp = (pred_evt & tgt_evt).sum().float()
    fp = (pred_evt & ~tgt_evt).sum().float()
    fn = (~pred_evt & tgt_evt).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2.0 * precision * recall / (precision + recall + 1e-8)


def radial_power_spectrum_2d(field: torch.Tensor, max_wavenumber: int | None = None) -> torch.Tensor:
    # field: [B, H, W]
    fft = torch.fft.rfft2(field, norm="ortho")
    power = (fft.real**2 + fft.imag**2).mean(dim=0)
    h, w2 = power.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=field.device),
        torch.arange(w2, device=field.device),
        indexing="ij",
    )
    ky = torch.minimum(yy, h - yy).float()
    kx = xx.float()
    kr = torch.sqrt(kx**2 + ky**2)
    kr_int = kr.round().long()
    if max_wavenumber is None:
        max_wavenumber = int(kr_int.max().item())
    max_wavenumber = min(max_wavenumber, int(kr_int.max().item()))
    spectrum = torch.zeros(max_wavenumber + 1, device=field.device)
    for k in range(max_wavenumber + 1):
        mask = kr_int == k
        if mask.any():
            spectrum[k] = power[mask].mean()
    return spectrum


def spectral_distance(pred: torch.Tensor, target: torch.Tensor, max_wavenumber: int = 32) -> torch.Tensor:
    # pred/target: [B,C,H,W]
    values = []
    for c in range(pred.shape[1]):
        sp = radial_power_spectrum_2d(pred[:, c], max_wavenumber=max_wavenumber)
        st = radial_power_spectrum_2d(target[:, c], max_wavenumber=max_wavenumber)
        values.append(torch.mean(torch.abs(torch.log1p(sp) - torch.log1p(st))))
    return torch.stack(values).mean()
