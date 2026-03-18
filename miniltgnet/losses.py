from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from miniltgnet.metrics import spectral_distance as spectral_metric_distance


def _safe_quantile(values: torch.Tensor, quantile: float, max_elements: int) -> torch.Tensor:
    flat = values.reshape(-1)
    if max_elements > 0 and flat.numel() > max_elements:
        step = max(1, flat.numel() // max_elements)
        flat = flat[::step]
        if flat.numel() > max_elements:
            flat = flat[:max_elements]
    if not torch.is_floating_point(flat) or flat.dtype in {torch.float16, torch.bfloat16}:
        flat = flat.float()
    q = min(1.0, max(0.0, float(quantile)))
    return torch.quantile(flat, q)


def field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float,
    l2_weight: float,
    extreme_weight: float = 0.0,
    extreme_quantile: float = 0.99,
    extreme_max_elements: int = 1_000_000,
) -> torch.Tensor:
    l1 = F.l1_loss(pred, target)
    l2 = F.mse_loss(pred, target)
    loss = l1_weight * l1 + l2_weight * l2
    if extreme_weight > 0:
        threshold = _safe_quantile(target, extreme_quantile, extreme_max_elements)
        mask = (target >= threshold).float()
        tail = (torch.abs(pred - target) * mask).sum() / (mask.sum() + 1e-6)
        loss = loss + extreme_weight * tail
    return loss


def spectral_high_freq_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    high_freq_power: float = 1.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    # Supports [B,T,C,H,W] or [B,C,H,W].
    if pred.ndim == 5:
        b, t, c, h, w = pred.shape
        pred = pred.reshape(b * t, c, h, w)
        target = target.reshape(b * t, c, h, w)
    if pred.ndim != 4:
        raise ValueError(f"Unsupported tensor shape for spectral loss: {tuple(pred.shape)}")

    pf = torch.fft.rfft2(pred, norm="ortho")
    tf = torch.fft.rfft2(target, norm="ortho")
    pp = pf.real.square() + pf.imag.square()
    tp = tf.real.square() + tf.imag.square()

    h = pp.shape[-2]
    w2 = pp.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=pp.device),
        torch.arange(w2, device=pp.device),
        indexing="ij",
    )
    ky = torch.minimum(yy, h - yy).float() / max(1.0, float(h // 2))
    kx = xx.float() / max(1.0, float(w2 - 1))
    kr = torch.sqrt(kx.square() + ky.square()).clamp_min(eps)
    weight = kr.pow(float(high_freq_power))
    weight = weight / (weight.mean() + eps)

    diff = torch.abs(torch.log1p(pp) - torch.log1p(tp))
    return (diff * weight[None, None]).mean()


def _flatten_spatiotemporal(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5:
        b, t, c, h, w = x.shape
        return x.reshape(b * t, c, h, w)
    if x.ndim == 4:
        return x
    raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")


def spectral_multiband_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    low_cut: float = 0.25,
    high_cut: float = 0.65,
    low_weight: float = 0.1,
    mid_weight: float = 0.3,
    high_weight: float = 0.6,
) -> torch.Tensor:
    pred4 = _flatten_spatiotemporal(pred)
    tgt4 = _flatten_spatiotemporal(target)
    pf = torch.fft.rfft2(pred4, norm="ortho")
    tf = torch.fft.rfft2(tgt4, norm="ortho")
    pp = pf.real.square() + pf.imag.square()
    tp = tf.real.square() + tf.imag.square()
    diff = torch.abs(torch.log1p(pp) - torch.log1p(tp))

    h = pp.shape[-2]
    w2 = pp.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=pp.device),
        torch.arange(w2, device=pp.device),
        indexing="ij",
    )
    ky = torch.minimum(yy, h - yy).float() / max(1.0, float(h // 2))
    kx = xx.float() / max(1.0, float(w2 - 1))
    kr = torch.sqrt(kx.square() + ky.square())

    low_mask = (kr <= float(low_cut)).to(diff.dtype)
    mid_mask = ((kr > float(low_cut)) & (kr <= float(high_cut))).to(diff.dtype)
    high_mask = (kr > float(high_cut)).to(diff.dtype)

    def _masked_mean(mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp_min(1.0)
        return (diff * mask[None, None]).sum() / (diff.shape[0] * diff.shape[1] * denom)

    low = _masked_mean(low_mask)
    mid = _masked_mean(mid_mask)
    high = _masked_mean(high_mask)
    return float(low_weight) * low + float(mid_weight) * mid + float(high_weight) * high


def gradient_structure_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred4 = _flatten_spatiotemporal(pred)
    tgt4 = _flatten_spatiotemporal(target)
    pred_dx = pred4[:, :, :, 1:] - pred4[:, :, :, :-1]
    tgt_dx = tgt4[:, :, :, 1:] - tgt4[:, :, :, :-1]
    pred_dy = pred4[:, :, 1:, :] - pred4[:, :, :-1, :]
    tgt_dy = tgt4[:, :, 1:, :] - tgt4[:, :, :-1, :]
    return 0.5 * (F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy))


def laplacian_structure_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred4 = _flatten_spatiotemporal(pred)
    tgt4 = _flatten_spatiotemporal(target)
    channels = pred4.shape[1]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=pred4.device,
        dtype=pred4.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(channels, 1, 1, 1)
    pred_lap = F.conv2d(pred4, kernel, padding=1, groups=channels)
    tgt_lap = F.conv2d(tgt4, kernel, padding=1, groups=channels)
    return F.l1_loss(pred_lap, tgt_lap)


def _lon_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b + 180.0) % 360.0 - 180.0


def trajectory_geodesic_loss(pred_traj: torch.Tensor, target_traj: torch.Tensor, scale_km: float = 100.0) -> torch.Tensor:
    lat1 = torch.deg2rad(pred_traj[..., 0])
    lon1 = torch.deg2rad(pred_traj[..., 1])
    lat2 = torch.deg2rad(target_traj[..., 0])
    lon2 = torch.deg2rad(target_traj[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2).square() + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).square()
    a = a.clamp(0.0, 1.0)
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt((1.0 - a).clamp_min(1e-8)))
    dist_km = 6371.0 * c
    return dist_km.mean() / max(1e-6, float(scale_km))


def trajectory_heading_loss(pred_traj: torch.Tensor, target_traj: torch.Tensor) -> torch.Tensor:
    if pred_traj.shape[1] <= 1:
        return torch.zeros((), device=pred_traj.device, dtype=pred_traj.dtype)
    pred_dlat = pred_traj[:, 1:, :, 0] - pred_traj[:, :-1, :, 0]
    tgt_dlat = target_traj[:, 1:, :, 0] - target_traj[:, :-1, :, 0]
    pred_dlon = _lon_diff(pred_traj[:, 1:, :, 1], pred_traj[:, :-1, :, 1])
    tgt_dlon = _lon_diff(target_traj[:, 1:, :, 1], target_traj[:, :-1, :, 1])
    cos_lat = torch.cos(torch.deg2rad(target_traj[:, 1:, :, 0]))
    pred_v = torch.stack([pred_dlat, pred_dlon * cos_lat], dim=-1)
    tgt_v = torch.stack([tgt_dlat, tgt_dlon * cos_lat], dim=-1)
    dot = (pred_v * tgt_v).sum(dim=-1)
    den = torch.sqrt(pred_v.square().sum(dim=-1) * tgt_v.square().sum(dim=-1) + 1e-8)
    return (1.0 - (dot / den).clamp(-1.0, 1.0)).mean()


class MiniCompositeLoss(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        lcfg = cfg["loss"]
        self.lambda_field = float(lcfg["lambda_field"])
        self.lambda_traj = float(lcfg["lambda_traj"])
        self.lambda_spec = float(lcfg.get("lambda_spec", 0.0))
        self.lambda_field_eff = self.lambda_field
        self.lambda_traj_eff = self.lambda_traj
        self.lambda_spec_eff = self.lambda_spec
        self.traj_lambda_ramp_epochs = int(lcfg.get("traj_lambda_ramp_epochs", 0))
        self.spec_lambda_ramp_epochs = int(lcfg.get("spec_lambda_ramp_epochs", 0))

        self.traj_scale_deg = float(lcfg.get("traj_scale_deg", 20.0))
        self.traj_mae_weight = float(lcfg.get("traj_mae_weight", 0.0))
        self.traj_velocity_weight = float(lcfg.get("traj_velocity_weight", 0.0))
        self.traj_velocity_scale_deg = float(lcfg.get("traj_velocity_scale_deg", self.traj_scale_deg))
        self.traj_smooth_weight = float(lcfg.get("traj_smooth_weight", 0.0))
        self.traj_smooth_scale_deg = float(lcfg.get("traj_smooth_scale_deg", self.traj_velocity_scale_deg))
        self.traj_geo_weight = float(lcfg.get("traj_geo_weight", 0.0))
        self.traj_geo_scale_km = float(lcfg.get("traj_geo_scale_km", 120.0))
        self.traj_heading_weight = float(lcfg.get("traj_heading_weight", 0.0))

        self.field_l1_weight = float(lcfg["field_l1_weight"])
        self.field_l2_weight = float(lcfg["field_l2_weight"])
        self.field_extreme_weight = float(lcfg.get("field_extreme_weight", 0.0))
        self.field_extreme_quantile = float(lcfg.get("field_extreme_quantile", 0.99))
        self.field_extreme_max_elements = int(lcfg.get("field_extreme_max_elements", 1_000_000))
        self.spec_high_freq_power = float(lcfg.get("spec_high_freq_power", 1.5))
        self.spec_full_weight = float(lcfg.get("spec_full_weight", 1.0))
        self.spec_high_weight = float(lcfg.get("spec_high_weight", 0.0))
        self.spec_band_weight = float(lcfg.get("spec_band_weight", 0.0))
        self.spec_band_low_cut = float(lcfg.get("spec_band_low_cut", 0.25))
        self.spec_band_high_cut = float(lcfg.get("spec_band_high_cut", 0.65))
        self.spec_band_low_weight = float(lcfg.get("spec_band_low_weight", 0.1))
        self.spec_band_mid_weight = float(lcfg.get("spec_band_mid_weight", 0.3))
        self.spec_band_high_weight = float(lcfg.get("spec_band_high_weight", 0.6))
        self.spec_grad_weight = float(lcfg.get("spec_grad_weight", 0.0))
        self.spec_lap_weight = float(lcfg.get("spec_lap_weight", 0.0))
        self.spec_max_wavenumber = int(lcfg.get("spec_max_wavenumber", 32))

    @staticmethod
    def _ramp(epoch: int, ramp_epochs: int) -> float:
        if ramp_epochs <= 0:
            return 1.0
        return float(min(1.0, max(0.0, (epoch + 1) / float(ramp_epochs))))

    def set_progress(self, epoch: int, total_epochs: int) -> None:
        del total_epochs
        self.lambda_field_eff = self.lambda_field
        self.lambda_traj_eff = self.lambda_traj * self._ramp(epoch, self.traj_lambda_ramp_epochs)
        self.lambda_spec_eff = self.lambda_spec * self._ramp(epoch, self.spec_lambda_ramp_epochs)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_field = outputs["field_pred"]
        pred_traj = outputs["traj_pred"]
        tgt_field = batch["y_future"].float().to(pred_field.device)
        tgt_traj = batch["traj_future"].float().to(pred_field.device)

        l_field = field_loss(
            pred=pred_field,
            target=tgt_field,
            l1_weight=self.field_l1_weight,
            l2_weight=self.field_l2_weight,
            extreme_weight=self.field_extreme_weight,
            extreme_quantile=self.field_extreme_quantile,
            extreme_max_elements=self.field_extreme_max_elements,
        )
        lat_diff = pred_traj[..., 0] - tgt_traj[..., 0]
        lon_diff = _lon_diff(pred_traj[..., 1], tgt_traj[..., 1])
        traj_err = torch.stack([lat_diff, lon_diff], dim=-1)
        l_traj_mse = torch.mean((traj_err / max(1e-6, self.traj_scale_deg)) ** 2)
        l_traj_mae = torch.mean(torch.abs(traj_err) / max(1e-6, self.traj_scale_deg))

        if pred_traj.shape[1] > 1:
            pred_dlat = pred_traj[:, 1:, :, 0] - pred_traj[:, :-1, :, 0]
            tgt_dlat = tgt_traj[:, 1:, :, 0] - tgt_traj[:, :-1, :, 0]
            pred_dlon = _lon_diff(pred_traj[:, 1:, :, 1], pred_traj[:, :-1, :, 1])
            tgt_dlon = _lon_diff(tgt_traj[:, 1:, :, 1], tgt_traj[:, :-1, :, 1])
            vel_err = torch.stack([pred_dlat - tgt_dlat, pred_dlon - tgt_dlon], dim=-1)
            l_traj_vel = torch.mean((vel_err / max(1e-6, self.traj_velocity_scale_deg)) ** 2)
        else:
            l_traj_vel = torch.zeros((), device=pred_traj.device, dtype=pred_traj.dtype)

        if pred_traj.shape[1] > 2:
            pred_acc_lat = pred_traj[:, 2:, :, 0] - 2.0 * pred_traj[:, 1:-1, :, 0] + pred_traj[:, :-2, :, 0]
            tgt_acc_lat = tgt_traj[:, 2:, :, 0] - 2.0 * tgt_traj[:, 1:-1, :, 0] + tgt_traj[:, :-2, :, 0]
            pred_acc_lon = _lon_diff(pred_traj[:, 2:, :, 1], pred_traj[:, 1:-1, :, 1]) - _lon_diff(
                pred_traj[:, 1:-1, :, 1], pred_traj[:, :-2, :, 1]
            )
            tgt_acc_lon = _lon_diff(tgt_traj[:, 2:, :, 1], tgt_traj[:, 1:-1, :, 1]) - _lon_diff(
                tgt_traj[:, 1:-1, :, 1], tgt_traj[:, :-2, :, 1]
            )
            acc_err = torch.stack([pred_acc_lat - tgt_acc_lat, pred_acc_lon - tgt_acc_lon], dim=-1)
            l_traj_smooth = torch.mean((acc_err / max(1e-6, self.traj_smooth_scale_deg)) ** 2)
        else:
            l_traj_smooth = torch.zeros((), device=pred_traj.device, dtype=pred_traj.dtype)

        if self.traj_geo_weight > 0:
            l_traj_geo = trajectory_geodesic_loss(pred_traj, tgt_traj, scale_km=self.traj_geo_scale_km)
        else:
            l_traj_geo = torch.zeros((), device=pred_traj.device, dtype=pred_traj.dtype)

        if self.traj_heading_weight > 0:
            l_traj_heading = trajectory_heading_loss(pred_traj, tgt_traj)
        else:
            l_traj_heading = torch.zeros((), device=pred_traj.device, dtype=pred_traj.dtype)

        l_traj = (
            l_traj_mse
            + self.traj_mae_weight * l_traj_mae
            + self.traj_velocity_weight * l_traj_vel
            + self.traj_smooth_weight * l_traj_smooth
            + self.traj_geo_weight * l_traj_geo
            + self.traj_heading_weight * l_traj_heading
        )
        if self.lambda_spec_eff > 0:
            if pred_field.ndim == 5:
                b, t, c, h, w = pred_field.shape
                pred4 = pred_field.reshape(b * t, c, h, w)
                tgt4 = tgt_field.reshape(b * t, c, h, w)
            else:
                pred4 = pred_field
                tgt4 = tgt_field

            spec_full = spectral_metric_distance(pred4, tgt4, max_wavenumber=self.spec_max_wavenumber)
            if self.spec_high_weight > 0:
                spec_high = spectral_high_freq_loss(
                    pred=pred_field,
                    target=tgt_field,
                    high_freq_power=self.spec_high_freq_power,
                )
            else:
                spec_high = torch.zeros((), device=pred_field.device, dtype=pred_field.dtype)

            if self.spec_band_weight > 0:
                spec_band = spectral_multiband_loss(
                    pred=pred_field,
                    target=tgt_field,
                    low_cut=self.spec_band_low_cut,
                    high_cut=self.spec_band_high_cut,
                    low_weight=self.spec_band_low_weight,
                    mid_weight=self.spec_band_mid_weight,
                    high_weight=self.spec_band_high_weight,
                )
            else:
                spec_band = torch.zeros((), device=pred_field.device, dtype=pred_field.dtype)

            if self.spec_grad_weight > 0:
                spec_grad = gradient_structure_loss(pred_field, tgt_field)
            else:
                spec_grad = torch.zeros((), device=pred_field.device, dtype=pred_field.dtype)

            if self.spec_lap_weight > 0:
                spec_lap = laplacian_structure_loss(pred_field, tgt_field)
            else:
                spec_lap = torch.zeros((), device=pred_field.device, dtype=pred_field.dtype)

            l_spec = (
                self.spec_full_weight * spec_full
                + self.spec_high_weight * spec_high
                + self.spec_band_weight * spec_band
                + self.spec_grad_weight * spec_grad
                + self.spec_lap_weight * spec_lap
            )
        else:
            l_spec = torch.zeros((), device=pred_field.device, dtype=pred_field.dtype)
            spec_band = torch.zeros((), device=pred_field.device, dtype=pred_field.dtype)
            spec_grad = torch.zeros((), device=pred_field.device, dtype=pred_field.dtype)
            spec_lap = torch.zeros((), device=pred_field.device, dtype=pred_field.dtype)

        total = self.lambda_field_eff * l_field + self.lambda_traj_eff * l_traj + self.lambda_spec_eff * l_spec
        details = {
            "loss_total": total.detach(),
            "loss_field": l_field.detach(),
            "loss_traj": l_traj.detach(),
            "loss_traj_mse": l_traj_mse.detach(),
            "loss_traj_mae": l_traj_mae.detach(),
            "loss_traj_vel": l_traj_vel.detach(),
            "loss_traj_smooth": l_traj_smooth.detach(),
            "loss_traj_geo": l_traj_geo.detach(),
            "loss_traj_heading": l_traj_heading.detach(),
            "loss_spec": l_spec.detach(),
            "loss_spec_band": spec_band.detach(),
            "loss_spec_grad": spec_grad.detach(),
            "loss_spec_lap": spec_lap.detach(),
            "lambda_field_eff": torch.tensor(self.lambda_field_eff, device=pred_field.device),
            "lambda_traj_eff": torch.tensor(self.lambda_traj_eff, device=pred_field.device),
            "lambda_spec_eff": torch.tensor(self.lambda_spec_eff, device=pred_field.device),
        }
        return total, details
