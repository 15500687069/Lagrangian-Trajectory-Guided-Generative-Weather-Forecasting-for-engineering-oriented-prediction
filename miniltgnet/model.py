from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp_geo(state: torch.Tensor) -> torch.Tensor:
    lat = state[..., 0].clamp(-90.0, 90.0)
    lon = state[..., 1] % 360.0
    return torch.stack([lat, lon], dim=-1)


class MiniTrajectoryPredictor(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int, max_step_deg: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.max_step_deg = float(max_step_deg)
        self.traj_embed = nn.Linear(2, hidden_dim)
        self.ctx_proj = nn.Linear(context_dim, hidden_dim)
        self.encoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z_ctx: torch.Tensor, traj_hist: torch.Tensor, horizon: int) -> torch.Tensor:
        # traj_hist [B,T,O,2] -> pred [B,Tf,O,2]
        b, t, o, _ = traj_hist.shape
        seq = traj_hist.permute(0, 2, 1, 3).reshape(b * o, t, 2)
        emb = self.traj_embed(seq)
        _, h = self.encoder(emb)
        h = h[-1]
        ctx = self.ctx_proj(z_ctx)[:, None, :].repeat(1, o, 1).reshape(b * o, -1)
        h = h + ctx
        state = seq[:, -1]
        outs = []
        for _ in range(horizon):
            inp = self.traj_embed(state)
            h = self.cell(inp, h)
            d = self.delta(h)
            if self.max_step_deg > 0:
                d = self.max_step_deg * torch.tanh(d / max(1e-6, self.max_step_deg))
            state = _clamp_geo(state + d)
            outs.append(state)
        return torch.stack(outs, dim=1).reshape(b, o, horizon, 2).permute(0, 2, 1, 3)


class MiniTrajectoryRefiner(nn.Module):
    def __init__(
        self,
        field_channels: int,
        context_dim: int,
        hidden_dim: int,
        max_refine_step_deg: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_refine_step_deg = float(max_refine_step_deg)
        in_dim = field_channels + context_dim + 5  # local field + ctx + [lat,lon] + prev_delta + local_heat
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta = nn.Linear(hidden_dim, 2)
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        traj_t: torch.Tensor,
        field_local: torch.Tensor,
        heat_local: torch.Tensor,
        z_ctx: torch.Tensor,
        prev_refine_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, o, _ = traj_t.shape
        ctx = z_ctx[:, None, :].expand(b, o, -1)
        x = torch.cat([field_local, ctx, traj_t, prev_refine_delta, heat_local], dim=-1)
        h = self.body(x)
        d = self.delta(h)
        if self.max_refine_step_deg > 0:
            d = self.max_refine_step_deg * torch.tanh(d / max(1e-6, self.max_refine_step_deg))
        g = torch.sigmoid(self.gate(h))
        d = g * d
        traj_refined = _clamp_geo(traj_t + d)
        return traj_refined, d


class MiniFieldStep(nn.Module):
    def __init__(
        self,
        field_channels: int,
        context_dim: int,
        context_channels: int,
        hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ctx_proj = nn.Linear(context_dim, context_channels)
        in_channels = field_channels + field_channels + 1 + context_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, field_channels, 3, padding=1),
        )

    def forward(
        self,
        prev: torch.Tensor,
        hist_mean: torch.Tensor,
        heat: torch.Tensor,
        z_ctx: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = prev.shape
        ctx = self.ctx_proj(z_ctx)[:, :, None, None].expand(b, -1, h, w)
        x = torch.cat([prev, hist_mean, heat, ctx], dim=1)
        return self.net(x)


class MiniHighFreqRefiner(nn.Module):
    def __init__(
        self,
        field_channels: int,
        context_dim: int,
        context_channels: int,
        hidden: int,
        scale: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.ctx_proj = nn.Linear(context_dim, context_channels)
        in_channels = field_channels + field_channels + 1 + context_channels
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.out = nn.Conv2d(hidden, field_channels, 3, padding=1)
        self.gate = nn.Conv2d(hidden, field_channels, 1)

    def forward(self, nxt: torch.Tensor, prev: torch.Tensor, heat: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        b, _, h, w = nxt.shape
        ctx = self.ctx_proj(z_ctx)[:, :, None, None].expand(b, -1, h, w)
        x = torch.cat([nxt, prev, heat, ctx], dim=1)
        h_map = self.pre(x)
        residual = self.out(h_map)
        high = residual - F.avg_pool2d(residual, kernel_size=5, stride=1, padding=2)
        gate = torch.sigmoid(self.gate(h_map))
        return nxt + self.scale * gate * high


class MiniLTGNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        traj_hidden_dim: int,
        traj_max_step_deg: float,
        field_hidden_dim: int,
        context_channels: int,
        raster_sigma_deg: float,
        residual_scale: float,
        dropout: float,
        traj_refine_enabled: bool,
        traj_refine_hidden_dim: int,
        traj_refine_max_deg: float,
        hf_refine_enabled: bool,
        hf_refine_hidden_dim: int,
        hf_refine_scale: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.raster_sigma_deg = float(raster_sigma_deg)
        self.residual_scale = float(residual_scale)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.traj = MiniTrajectoryPredictor(
            context_dim=hidden_dim,
            hidden_dim=traj_hidden_dim,
            max_step_deg=traj_max_step_deg,
            dropout=dropout,
        )
        self.field_step = MiniFieldStep(
            field_channels=in_channels,
            context_dim=hidden_dim,
            context_channels=context_channels,
            hidden=field_hidden_dim,
            dropout=dropout,
        )
        self.traj_refiner = (
            MiniTrajectoryRefiner(
                field_channels=in_channels,
                context_dim=hidden_dim,
                hidden_dim=traj_refine_hidden_dim,
                max_refine_step_deg=traj_refine_max_deg,
                dropout=dropout,
            )
            if traj_refine_enabled
            else None
        )
        self.hf_refiner = (
            MiniHighFreqRefiner(
                field_channels=in_channels,
                context_dim=hidden_dim,
                context_channels=context_channels,
                hidden=hf_refine_hidden_dim,
                scale=hf_refine_scale,
                dropout=dropout,
            )
            if hf_refine_enabled and hf_refine_scale > 0
            else None
        )

    def _traj_to_grid(self, traj_t: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        if lat.ndim == 2:
            lat = lat[0]
        if lon.ndim == 2:
            lon = lon[0]
        lat = lat.to(traj_t.device, traj_t.dtype)
        lon = lon.to(traj_t.device, traj_t.dtype)

        lat_min = torch.minimum(lat[0], lat[-1])
        lat_max = torch.maximum(lat[0], lat[-1])
        lon_min = torch.minimum(lon[0], lon[-1])
        lon_max = torch.maximum(lon[0], lon[-1])

        lat_t = traj_t[..., 0].clamp(lat_min, lat_max)
        lon_t = traj_t[..., 1]
        if float(torch.abs(lon_max - lon_min).item()) < 359.0:
            lon_t = lon_t.clamp(lon_min, lon_max)

        lat_den = lat[-1] - lat[0]
        lon_den = lon[-1] - lon[0]
        y = 2.0 * (lat_t - lat[0]) / (lat_den + 1e-6) - 1.0
        x = 2.0 * (lon_t - lon[0]) / (lon_den + 1e-6) - 1.0
        grid = torch.stack([x, y], dim=-1).unsqueeze(2)
        return grid

    def _sample_at_traj(self, field: torch.Tensor, traj_t: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        # field [B,C,H,W], traj_t [B,O,2] -> [B,O,C]
        grid = self._traj_to_grid(traj_t, lat, lon)
        sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(-1).permute(0, 2, 1).contiguous()

    def _rasterize(self, traj_t: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        # traj_t [B,O,2] -> [B,1,H,W]
        if lat.ndim == 2:
            lat = lat[0]
        if lon.ndim == 2:
            lon = lon[0]
        b, o, _ = traj_t.shape
        h = lat.numel()
        w = lon.numel()
        lat_grid = lat.view(1, 1, h, 1).to(traj_t.device)
        lon_grid = lon.view(1, 1, 1, w).to(traj_t.device)
        lat_t = traj_t[..., 0].view(b, o, 1, 1)
        lon_t = traj_t[..., 1].view(b, o, 1, 1)
        dlon = (lon_grid - lon_t + 180.0) % 360.0 - 180.0
        d2 = ((lat_grid - lat_t) ** 2 + dlon**2) / (self.raster_sigma_deg**2 + 1e-6)
        heat = torch.exp(-0.5 * d2).sum(dim=1, keepdim=True)
        heat = heat / (heat.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return heat

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_hist = batch["x_hist"].float()
        traj_hist = batch["traj_hist"].float()
        lat = batch["lat"]
        lon = batch["lon"]
        horizon = int(batch["y_future"].shape[1]) if "y_future" in batch else int(traj_hist.shape[1])

        hist_last = x_hist[:, -1]
        hist_mean = x_hist.mean(dim=1)
        enc_in = torch.cat([hist_last, hist_mean], dim=1)
        z_ctx = self.encoder(enc_in).flatten(1)
        traj_base = self.traj(z_ctx, traj_hist, horizon)

        prev = hist_last
        traj_carry = torch.zeros_like(traj_base[:, 0])
        prev_refine_delta = torch.zeros_like(traj_base[:, 0])
        outputs = []
        traj_outputs = []
        for ti in range(horizon):
            traj_t = _clamp_geo(traj_base[:, ti] + traj_carry)
            heat = self._rasterize(traj_t, lat, lon)
            delta = self.field_step(prev, hist_mean, heat, z_ctx)
            nxt = prev + self.residual_scale * delta
            if self.hf_refiner is not None:
                nxt = self.hf_refiner(nxt=nxt, prev=prev, heat=heat, z_ctx=z_ctx)
            if self.traj_refiner is not None:
                field_local = self._sample_at_traj(nxt, traj_t, lat, lon)
                heat_local = self._sample_at_traj(heat, traj_t, lat, lon)
                traj_t, refine_delta = self.traj_refiner(
                    traj_t=traj_t,
                    field_local=field_local,
                    heat_local=heat_local,
                    z_ctx=z_ctx,
                    prev_refine_delta=prev_refine_delta,
                )
                prev_refine_delta = refine_delta
                traj_carry = 0.5 * traj_carry + refine_delta
            outputs.append(nxt)
            traj_outputs.append(traj_t)
            prev = nxt
        field_pred = torch.stack(outputs, dim=1)
        traj_pred = torch.stack(traj_outputs, dim=1)
        return {"field_pred": field_pred, "traj_pred": traj_pred}


def build_model(cfg: dict, in_channels: int) -> MiniLTGNet:
    mcfg = cfg["model"]
    return MiniLTGNet(
        in_channels=in_channels,
        hidden_dim=int(mcfg["hidden_dim"]),
        traj_hidden_dim=int(mcfg["traj_hidden_dim"]),
        traj_max_step_deg=float(mcfg["traj_max_step_deg"]),
        field_hidden_dim=int(mcfg["field_hidden_dim"]),
        context_channels=int(mcfg["context_channels"]),
        raster_sigma_deg=float(mcfg["raster_sigma_deg"]),
        residual_scale=float(mcfg["residual_scale"]),
        dropout=float(mcfg.get("dropout", 0.0)),
        traj_refine_enabled=bool(mcfg.get("traj_refine_enabled", False)),
        traj_refine_hidden_dim=int(mcfg.get("traj_refine_hidden_dim", mcfg["traj_hidden_dim"])),
        traj_refine_max_deg=float(mcfg.get("traj_refine_max_deg", 1.2)),
        hf_refine_enabled=bool(mcfg.get("hf_refine_enabled", False)),
        hf_refine_hidden_dim=int(mcfg.get("hf_refine_hidden_dim", mcfg["field_hidden_dim"])),
        hf_refine_scale=float(mcfg.get("hf_refine_scale", 0.0)),
    )
