from __future__ import annotations

import math

import torch
import torch.nn as nn


def spherical_harmonic_features(
    lat: torch.Tensor,
    lon: torch.Tensor,
    num_freqs: int,
) -> torch.Tensor:
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    feats = []
    for k in range(1, num_freqs + 1):
        feats.append(torch.sin(k * lat_rad))
        feats.append(torch.cos(k * lat_rad))
        feats.append(torch.sin(k * lon_rad))
        feats.append(torch.cos(k * lon_rad))
        feats.append(torch.sin(k * lat_rad) * torch.cos(k * lon_rad))
        feats.append(torch.cos(k * lat_rad) * torch.sin(k * lon_rad))
    return torch.stack(feats, dim=-1)


class LearnedTemporalEncoding(nn.Module):
    def __init__(self, max_steps: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_steps, dim)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        return self.embedding(steps)


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb
