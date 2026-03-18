from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ltg_net.models.positional import LearnedTemporalEncoding, spherical_harmonic_features


class SphereSpatiotemporalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        max_history_steps: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        try:
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth, enable_nested_tensor=False)
        except TypeError:
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.temporal_encoding = LearnedTemporalEncoding(max_steps=max_history_steps + 8, dim=hidden_dim)
        self.spatial_proj = nn.Linear(24, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _spatial_encoding(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        h_tokens: int,
        w_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        lat_idx = torch.linspace(0, lat.numel() - 1, h_tokens, device=device).long()
        lon_idx = torch.linspace(0, lon.numel() - 1, w_tokens, device=device).long()
        lat_p = lat[lat_idx]
        lon_p = lon[lon_idx]
        lat_grid, lon_grid = torch.meshgrid(lat_p, lon_p, indexing="ij")
        feat = spherical_harmonic_features(lat_grid, lon_grid, num_freqs=4)  # [H', W', 24]
        feat = self.spatial_proj(feat)
        return feat

    def forward(
        self,
        x_hist: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # x_hist: [B, T, C, H, W]
        b, t, c, h, w = x_hist.shape
        x = x_hist.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        feat = self.patch_embed(x)  # [B, D, T, H', W']
        _, d, _, hp, wp = feat.shape
        tokens = feat.permute(0, 2, 3, 4, 1).reshape(b, t * hp * wp, d)

        spatial = self._spatial_encoding(lat, lon, hp, wp, x_hist.device).reshape(hp * wp, d)
        spatial = spatial[None].repeat(t, 1, 1).reshape(t * hp * wp, d)
        time_ids = torch.arange(t, device=x_hist.device)
        temporal = self.temporal_encoding(time_ids)  # [T, D]
        temporal = temporal[:, None, :].repeat(1, hp * wp, 1).reshape(t * hp * wp, d)

        tokens = tokens + spatial[None] + temporal[None]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        token_map = tokens.reshape(b, t, hp, wp, d)
        z_bg_map = token_map.mean(dim=1).permute(0, 3, 1, 2).contiguous()
        z_bg_global = tokens.mean(dim=1)
        return {
            "z_bg_map": z_bg_map,
            "z_bg_global": z_bg_global,
            "tokens": tokens,
            "grid_shape": torch.tensor([hp, wp], device=x_hist.device),
        }
