from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(self.dropout(F.gelu(self.norm2(h))))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res_blocks: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        cur = in_ch
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(cur, out_ch, dropout))
            cur = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.blocks(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_res_blocks: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        blocks = []
        cur = out_ch + skip_ch
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(cur, out_ch, dropout))
            cur = out_ch
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.blocks(x)


class ConditionalUNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        channel_mults: list[int],
        num_res_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList()
        down_channels = []
        cur = base_channels
        for mult in channel_mults:
            out = base_channels * mult
            self.downs.append(DownBlock(cur, out, num_res_blocks, dropout))
            down_channels.append(out)
            cur = out
        self.mid = nn.Sequential(
            ResidualBlock(cur, cur, dropout),
            ResidualBlock(cur, cur, dropout),
        )
        self.ups = nn.ModuleList()
        for skip_ch in reversed(down_channels):
            out = skip_ch
            self.ups.append(UpBlock(cur, skip_ch, out, num_res_blocks, dropout))
            cur = out
        self.out_norm = nn.GroupNorm(8, cur)
        self.out_conv = nn.Conv2d(cur, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        skips = []
        for block in self.downs:
            x, skip = block(x)
            skips.append(skip)
        x = self.mid(x)
        for block, skip in zip(self.ups, reversed(skips)):
            x = block(x, skip)
        x = self.out_conv(F.gelu(self.out_norm(x)))
        return x


@dataclass
class RasterConfig:
    sigma_deg: float = 2.5
    normalize: bool = True


class TrajectoryRasterizer(nn.Module):
    def __init__(self, cfg: RasterConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RasterConfig()

    def forward(
        self,
        traj: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
    ) -> torch.Tensor:
        # traj [B,T,O,2], lat [H], lon [W]
        b, t, o, _ = traj.shape
        h = lat.numel()
        w = lon.numel()
        lat_grid = lat[None, None, None, :, None].to(traj.device)
        lon_grid = lon[None, None, None, None, :].to(traj.device)
        lat_t = traj[..., 0][:, :, :, None, None]
        lon_t = traj[..., 1][:, :, :, None, None]
        d2 = ((lat_grid - lat_t) ** 2 + (lon_grid - lon_t) ** 2) / (self.cfg.sigma_deg**2 + 1e-8)
        heat = torch.exp(-0.5 * d2)
        heat = heat.sum(dim=2)  # [B,T,H,W]
        if self.cfg.normalize:
            heat = heat / (heat.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        return heat


class TrajectoryConditionedUNetGenerator(nn.Module):
    def __init__(
        self,
        field_channels: int,
        latent_channels: int,
        base_channels: int,
        channel_mults: list[int],
        num_res_blocks: int,
        dropout: float,
        residual_forecast: bool = True,
        trend_scale: float = 0.35,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.residual_forecast = bool(residual_forecast)
        self.trend_scale = float(trend_scale)
        self.residual_scale = float(residual_scale)
        self.rasterizer = TrajectoryRasterizer()
        self.history_proj = nn.Conv2d(field_channels * 2, field_channels, 1)
        self.step_net = ConditionalUNet2D(
            in_channels=field_channels * 3 + latent_channels + 1,
            out_channels=field_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        z_bg_map: torch.Tensor,
        traj_pred: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
    ) -> torch.Tensor:
        # x_hist [B,T_hist,C,H,W], z_bg_map [B,D,H',W'], traj_pred [B,T_future,O,2]
        b, t_future, _, _ = traj_pred.shape
        _, t_hist, c, h, w = x_hist.shape
        bg = F.interpolate(z_bg_map, size=(h, w), mode="bilinear", align_corners=False)
        heat = self.rasterizer(traj_pred, lat, lon)  # [B,T,H,W]
        hist_mean = x_hist.mean(dim=1)
        if t_hist > 1:
            hist_trend = x_hist[:, -1] - x_hist[:, -2]
        else:
            hist_trend = torch.zeros_like(x_hist[:, -1])
        hist_ctx = self.history_proj(torch.cat([hist_mean, hist_trend], dim=1))

        prev = x_hist[:, -1]
        outputs = []
        for ti in range(t_future):
            anchor = x_hist[:, -1] + self.trend_scale * float(ti + 1) * hist_trend
            cond = torch.cat([prev, anchor, hist_ctx, bg, heat[:, ti : ti + 1]], dim=1)
            raw = self.step_net(cond)
            if self.residual_forecast:
                nxt = anchor + self.residual_scale * raw
            else:
                nxt = raw
            outputs.append(nxt)
            prev = nxt
        return torch.stack(outputs, dim=1)
