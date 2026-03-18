from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_lat(lat_points: torch.Tensor, lat_axis: torch.Tensor) -> torch.Tensor:
    lat0 = lat_axis[0]
    lat1 = lat_axis[-1]
    if lat0 > lat1:
        return 2.0 * (lat0 - lat_points) / (lat0 - lat1 + 1e-8) - 1.0
    return 2.0 * (lat_points - lat0) / (lat1 - lat0 + 1e-8) - 1.0


def _normalize_lon(lon_points: torch.Tensor, lon_axis: torch.Tensor) -> torch.Tensor:
    lon0 = lon_axis[0]
    lon1 = lon_axis[-1]
    return 2.0 * (lon_points - lon0) / (lon1 - lon0 + 1e-8) - 1.0


def sample_field_at_points(
    field: torch.Tensor,
    points: torch.Tensor,
    lat_axis: torch.Tensor,
    lon_axis: torch.Tensor,
) -> torch.Tensor:
    # field [B,H,W], points [B,O,2] (lat,lon)
    lat_p = points[..., 0]
    lon_p = points[..., 1]
    y = _normalize_lat(lat_p, lat_axis)
    x = _normalize_lon(lon_p, lon_axis)
    grid = torch.stack([x, y], dim=-1).unsqueeze(2)  # [B,O,1,2]
    sampled = F.grid_sample(
        field.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(1).squeeze(-1)


def trajectory_velocity_ms(
    traj: torch.Tensor,
    dt_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # traj [B,T,O,2] lat/lon deg
    dlat = traj[:, 1:, :, 0] - traj[:, :-1, :, 0]
    dlon = traj[:, 1:, :, 1] - traj[:, :-1, :, 1]
    dlon = ((dlon + 180.0) % 360.0) - 180.0
    lat_mid = 0.5 * (traj[:, 1:, :, 0] + traj[:, :-1, :, 0])
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * torch.cos(torch.deg2rad(lat_mid))
    v = dlat * meters_per_deg_lat / dt_seconds
    u = dlon * meters_per_deg_lon / dt_seconds
    return u, v


def advection_consistency_loss(
    traj_pred: torch.Tensor,
    field_pred: torch.Tensor,
    lat_axis: torch.Tensor,
    lon_axis: torch.Tensor,
    u_channel: int,
    v_channel: int,
    dt_seconds: float = 21_600.0,
) -> torch.Tensor:
    # traj_pred [B,T,O,2], field_pred [B,T,C,H,W]
    if traj_pred.shape[1] < 2:
        return torch.tensor(0.0, device=field_pred.device)
    u_traj, v_traj = trajectory_velocity_ms(traj_pred, dt_seconds=dt_seconds)
    losses = []
    for ti in range(traj_pred.shape[1] - 1):
        pts = traj_pred[:, ti]
        u_field = field_pred[:, ti, u_channel]
        v_field = field_pred[:, ti, v_channel]
        u_sample = sample_field_at_points(u_field, pts, lat_axis, lon_axis)
        v_sample = sample_field_at_points(v_field, pts, lat_axis, lon_axis)
        du = u_sample - u_traj[:, ti]
        dv = v_sample - v_traj[:, ti]
        losses.append((du**2 + dv**2).mean())
    return torch.stack(losses).mean()
