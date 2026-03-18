from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

OMEGA = 7.2921159e-5
EARTH_RADIUS = 6_371_000.0
GRAVITY = 9.81


def _dy_dx(lat_axis: torch.Tensor, lon_axis: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lat = lat_axis.to(device)
    lon = lon_axis.to(device)
    dlat = torch.gradient(lat, edge_order=1)[0].abs().clamp(min=1e-6)
    dlon = torch.gradient(lon, edge_order=1)[0].abs().clamp(min=1e-6)
    dy = dlat * 111_320.0
    dx = (dlon[None, :] * 111_320.0 * torch.cos(torch.deg2rad(lat))[:, None]).abs().clamp(min=1.0)
    return dy, dx


def _diff_x(field: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(field)
    out[..., :, 1:-1] = (field[..., :, 2:] - field[..., :, :-2]) / (dx[:, 2:] + dx[:, :-2] + 1e-8)
    out[..., :, 0] = (field[..., :, 1] - field[..., :, 0]) / (dx[:, 0] + 1e-8)
    out[..., :, -1] = (field[..., :, -1] - field[..., :, -2]) / (dx[:, -1] + 1e-8)
    return out


def _diff_y(field: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(field)
    out[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (dy[2:, None] + dy[:-2, None] + 1e-8)
    out[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / (dy[0] + 1e-8)
    out[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / (dy[-1] + 1e-8)
    return out


def _laplacian(field: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    return _diff_x(_diff_x(field, dx), dx) + _diff_y(_diff_y(field, dy), dy)


def _time_diff(field: torch.Tensor, dt_seconds: float) -> torch.Tensor:
    if field.shape[1] < 2:
        return field[:, :0]
    return (field[:, 1:] - field[:, :-1]) / max(dt_seconds, 1.0)


def _time_mid(field: torch.Tensor) -> torch.Tensor:
    if field.shape[1] < 2:
        return field[:, :0]
    return 0.5 * (field[:, 1:] + field[:, :-1])


def divergence(field_u: torch.Tensor, field_v: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    # field_* [...,H,W], dy [H], dx [H,W]
    du_dx = _diff_x(field_u, dx)
    dv_dy = _diff_y(field_v, dy)
    return du_dx + dv_dy


def vorticity(field_u: torch.Tensor, field_v: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    dv_dx = _diff_x(field_v, dx)
    du_dy = _diff_y(field_u, dy)
    return dv_dx - du_dy


def moist_static_energy(field_t: torch.Tensor, field_z: torch.Tensor, field_q: torch.Tensor) -> torch.Tensor:
    cp = 1004.0
    g = GRAVITY
    lv = 2.5e6
    return cp * field_t + g * field_z + lv * field_q


def strict_pde_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_axis: torch.Tensor,
    lon_axis: torch.Tensor,
    u_channel: int,
    v_channel: int,
    t_channel: int | None,
    z_channel: int | None,
    q_channel: int | None,
    strict_cfg: dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = pred.device
    zero = torch.tensor(0.0, device=device)
    cfg = strict_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return zero, {}
    if pred.shape[1] < 2:
        return zero, {}

    eps = 1e-8
    dy, dx = _dy_dx(lat_axis=lat_axis, lon_axis=lon_axis, device=device)
    dt_seconds = float(cfg.get("time_step_hours", 6.0)) * 3600.0

    w_cont = float(cfg.get("continuity_weight", 1.0))
    w_vort = float(cfg.get("vorticity_weight", 1.0))
    w_thermo = float(cfg.get("thermodynamic_weight", 1.0))
    w_moist = float(cfg.get("moisture_weight", 1.0))
    w_geo = float(cfg.get("geostrophic_weight", 1.0))
    cont_zero_reg = float(cfg.get("continuity_zero_reg", 0.0))
    vort_visc = float(cfg.get("vorticity_viscosity", 0.0))
    add_beta = bool(cfg.get("include_beta_term", True))
    f_min = float(cfg.get("f_min", 1e-5))

    u_p = pred[:, :, u_channel]
    v_p = pred[:, :, v_channel]
    u_t = target[:, :, u_channel]
    v_t = target[:, :, v_channel]

    # 1) Continuity (horizontal divergence) consistency
    div_p = divergence(u_p, v_p, dy, dx)
    div_t = divergence(u_t, v_t, dy, dx)
    div_ref = torch.mean(div_t**2).detach()
    loss_cont = F.mse_loss(div_p, div_t) / (div_ref + eps)
    if cont_zero_reg > 0.0:
        loss_cont = loss_cont + cont_zero_reg * torch.mean(div_p**2) / (div_ref + eps)

    # 2) Vorticity transport PDE residual consistency
    zeta_p = vorticity(u_p, v_p, dy, dx)
    zeta_t = vorticity(u_t, v_t, dy, dx)

    zeta_mid_p = _time_mid(zeta_p)
    zeta_mid_t = _time_mid(zeta_t)
    u_mid_p = _time_mid(u_p)
    v_mid_p = _time_mid(v_p)
    u_mid_t = _time_mid(u_t)
    v_mid_t = _time_mid(v_t)

    res_vort_p = _time_diff(zeta_p, dt_seconds) + u_mid_p * _diff_x(zeta_mid_p, dx) + v_mid_p * _diff_y(zeta_mid_p, dy)
    res_vort_t = _time_diff(zeta_t, dt_seconds) + u_mid_t * _diff_x(zeta_mid_t, dx) + v_mid_t * _diff_y(zeta_mid_t, dy)

    if add_beta:
        lat_rad = torch.deg2rad(lat_axis.to(device))
        beta = (2.0 * OMEGA * torch.cos(lat_rad) / EARTH_RADIUS)[None, None, :, None]
        res_vort_p = res_vort_p + beta * v_mid_p
        res_vort_t = res_vort_t + beta * v_mid_t

    if vort_visc > 0.0:
        res_vort_p = res_vort_p - vort_visc * _laplacian(zeta_mid_p, dy, dx)
        res_vort_t = res_vort_t - vort_visc * _laplacian(zeta_mid_t, dy, dx)

    vort_ref = torch.mean(res_vort_t**2).detach()
    loss_vort = F.mse_loss(res_vort_p, res_vort_t) / (vort_ref + eps)

    # 3) Thermodynamic advection consistency
    loss_thermo = zero
    if t_channel is not None:
        t_p = pred[:, :, t_channel]
        t_t = target[:, :, t_channel]
        res_t_p = _time_diff(t_p, dt_seconds) + u_mid_p * _diff_x(_time_mid(t_p), dx) + v_mid_p * _diff_y(_time_mid(t_p), dy)
        res_t_t = _time_diff(t_t, dt_seconds) + u_mid_t * _diff_x(_time_mid(t_t), dx) + v_mid_t * _diff_y(_time_mid(t_t), dy)
        thermo_ref = torch.mean(torch.abs(res_t_t)).detach()
        loss_thermo = F.l1_loss(res_t_p, res_t_t) / (thermo_ref + eps)

    # 4) Moisture advection consistency
    loss_moist = zero
    if q_channel is not None:
        q_p = pred[:, :, q_channel]
        q_t = target[:, :, q_channel]
        res_q_p = _time_diff(q_p, dt_seconds) + u_mid_p * _diff_x(_time_mid(q_p), dx) + v_mid_p * _diff_y(_time_mid(q_p), dy)
        res_q_t = _time_diff(q_t, dt_seconds) + u_mid_t * _diff_x(_time_mid(q_t), dx) + v_mid_t * _diff_y(_time_mid(q_t), dy)
        moist_ref = torch.mean(torch.abs(res_q_t)).detach()
        loss_moist = F.l1_loss(res_q_p, res_q_t) / (moist_ref + eps)

    # 5) Geostrophic consistency (ageostrophic component match)
    loss_geo = zero
    if z_channel is not None:
        z_p = pred[:, :, z_channel]
        z_t = target[:, :, z_channel]
        lat_rad = torch.deg2rad(lat_axis.to(device))
        f = 2.0 * OMEGA * torch.sin(lat_rad)
        f_sign = torch.where(f >= 0.0, torch.ones_like(f), -torch.ones_like(f))
        f = f_sign * torch.clamp(torch.abs(f), min=f_min)
        f_map = f[None, None, :, None]

        ug_p = -(GRAVITY / f_map) * _diff_y(z_p, dy)
        vg_p = (GRAVITY / f_map) * _diff_x(z_p, dx)
        ug_t = -(GRAVITY / f_map) * _diff_y(z_t, dy)
        vg_t = (GRAVITY / f_map) * _diff_x(z_t, dx)

        ageo_u_p = u_p - ug_p
        ageo_v_p = v_p - vg_p
        ageo_u_t = u_t - ug_t
        ageo_v_t = v_t - vg_t

        ageo_ref = (torch.mean(ageo_u_t**2) + torch.mean(ageo_v_t**2)).detach()
        loss_geo = (F.mse_loss(ageo_u_p, ageo_u_t) + F.mse_loss(ageo_v_p, ageo_v_t)) / (ageo_ref + eps)

    strict_total = (
        w_cont * loss_cont
        + w_vort * loss_vort
        + w_thermo * loss_thermo
        + w_moist * loss_moist
        + w_geo * loss_geo
    )
    details = {
        "strict_continuity": loss_cont.detach(),
        "strict_vorticity": loss_vort.detach(),
        "strict_thermodynamic": loss_thermo.detach(),
        "strict_moisture": loss_moist.detach(),
        "strict_geostrophic": loss_geo.detach(),
        "strict_total": strict_total.detach(),
    }
    return strict_total, details


def physics_consistency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_axis: torch.Tensor,
    lon_axis: torch.Tensor,
    u_channel: int,
    v_channel: int,
    t_channel: int | None,
    z_channel: int | None,
    q_channel: int | None,
    divergence_weight: float,
    moist_static_energy_weight: float,
    strict_cfg: dict[str, Any] | None = None,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # pred/target [B,T,C,H,W]
    device = pred.device
    dy, dx = _dy_dx(lat_axis=lat_axis, lon_axis=lon_axis, device=device)

    div_losses = []
    for ti in range(pred.shape[1]):
        u_p = pred[:, ti, u_channel]
        v_p = pred[:, ti, v_channel]
        u_t = target[:, ti, u_channel]
        v_t = target[:, ti, v_channel]
        div_p = divergence(u_p, v_p, dy, dx)
        div_t = divergence(u_t, v_t, dy, dx)
        ref = torch.mean(div_t**2).detach()
        div_match = F.mse_loss(div_p, div_t) / (ref + 1e-8)
        div_reg = 0.1 * torch.mean(div_p**2) / (ref + 1e-8)
        div_losses.append(div_match + div_reg)
    loss_div = torch.stack(div_losses).mean()

    loss_mse = torch.tensor(0.0, device=device)
    if t_channel is not None and z_channel is not None and q_channel is not None:
        mse_losses = []
        for ti in range(pred.shape[1]):
            mse_p = moist_static_energy(
                pred[:, ti, t_channel],
                pred[:, ti, z_channel],
                pred[:, ti, q_channel],
            )
            mse_t = moist_static_energy(
                target[:, ti, t_channel],
                target[:, ti, z_channel],
                target[:, ti, q_channel],
            )
            ref = torch.mean(torch.abs(mse_t)).detach()
            mse_losses.append(F.l1_loss(mse_p, mse_t) / (ref + 1e-8))
        loss_mse = torch.stack(mse_losses).mean()

    strict_loss, strict_details = strict_pde_loss(
        pred=pred,
        target=target,
        lat_axis=lat_axis,
        lon_axis=lon_axis,
        u_channel=u_channel,
        v_channel=v_channel,
        t_channel=t_channel,
        z_channel=z_channel,
        q_channel=q_channel,
        strict_cfg=strict_cfg,
    )

    total = divergence_weight * loss_div + moist_static_energy_weight * loss_mse + strict_loss
    if not return_details:
        return total

    details = {
        "loss_phys_divergence": loss_div.detach(),
        "loss_phys_mse": loss_mse.detach(),
    }
    for key, value in strict_details.items():
        details[f"loss_phys_{key}"] = value.detach()
    return total, details
