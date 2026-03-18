from __future__ import annotations

import math

import torch

EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor,
) -> torch.Tensor:
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(
        dlon / 2.0
    ) ** 2
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a + 1e-8))
    return EARTH_RADIUS_KM * c


def wrap_longitude(lon: torch.Tensor) -> torch.Tensor:
    return lon % 360.0


def lon_lat_to_normalized_grid(
    lat: torch.Tensor,
    lon: torch.Tensor,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> torch.Tensor:
    lat_norm = 2.0 * (lat - lat_min) / (lat_max - lat_min + 1e-8) - 1.0
    lon_norm = 2.0 * (lon - lon_min) / (lon_max - lon_min + 1e-8) - 1.0
    return torch.stack([lon_norm, lat_norm], dim=-1)


def curvature_from_trajectory(traj: torch.Tensor) -> torch.Tensor:
    # traj: [B, T, O, 2] in degree
    v1 = traj[:, 1:-1] - traj[:, :-2]
    v2 = traj[:, 2:] - traj[:, 1:-1]
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    n1 = torch.norm(v1, dim=-1)
    n2 = torch.norm(v2, dim=-1)
    denom = (n1 * n2 + 1e-8).clamp(min=1e-8)
    sin_angle = cross / denom
    kappa = torch.abs(torch.asin(sin_angle.clamp(-0.9999, 0.9999)))
    pad_left = torch.zeros_like(kappa[:, :1])
    pad_right = torch.zeros_like(kappa[:, :1])
    return torch.cat([pad_left, kappa, pad_right], dim=1)


def degree_per_second_to_ms(value_deg: torch.Tensor, latitude_deg: torch.Tensor) -> torch.Tensor:
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = meters_per_degree_lat * torch.cos(torch.deg2rad(latitude_deg))
    u = value_deg[..., 1] * meters_per_degree_lon
    v = value_deg[..., 0] * meters_per_degree_lat
    return torch.stack([u, v], dim=-1)


def angle_to_unit_vector(angle_rad: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(angle_rad), torch.sin(angle_rad)], dim=-1)


def great_circle_bearing(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor,
) -> torch.Tensor:
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)
    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    return torch.atan2(y, x)


def central_diff_2d(field: torch.Tensor, dy: float, dx: float) -> tuple[torch.Tensor, torch.Tensor]:
    # field: [B, H, W]
    grad_y = torch.zeros_like(field)
    grad_x = torch.zeros_like(field)
    grad_y[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dy)
    grad_x[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2.0 * dx)
    grad_y[:, 0] = (field[:, 1] - field[:, 0]) / dy
    grad_y[:, -1] = (field[:, -1] - field[:, -2]) / dy
    grad_x[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / dx
    grad_x[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / dx
    return grad_y, grad_x


def scalar_gradient_magnitude(field: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    gy, gx = central_diff_2d(field, dy, dx)
    return torch.sqrt(gy**2 + gx**2 + 1e-8)

