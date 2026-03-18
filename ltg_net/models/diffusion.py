from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltg_net.models.generator_unet import TrajectoryRasterizer
from ltg_net.models.positional import SinusoidalTimeEncoding


class Autoencoder2D(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(256, latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
        out = self.decoder(z)
        if target_hw is not None and out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


class DenoiserUNet(nn.Module):
    def __init__(self, latent_channels: int, cond_channels: int, hidden: int = 128) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEncoding(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.in_conv = nn.Conv2d(latent_channels + cond_channels, hidden, 3, padding=1)
        self.down1 = nn.Conv2d(hidden, hidden * 2, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(hidden * 2, hidden * 4, 4, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden * 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden * 4, hidden * 4, 3, padding=1),
        )
        self.up2 = nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1)
        self.out = nn.Conv2d(hidden, latent_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        # x_t [N,C,H,W], cond [N,Cc,H,W], time_step [N]
        t_emb = self.time_mlp(time_step)[:, :, None, None]
        h0 = self.in_conv(torch.cat([x_t, cond], dim=1))
        h0 = F.gelu(h0 + t_emb)
        h1 = F.gelu(self.down1(h0))
        h2 = F.gelu(self.down2(h1))
        hm = self.mid(h2)
        u2 = F.gelu(self.up2(hm) + h1)
        u1 = F.gelu(self.up1(u2) + h0)
        return self.out(u1)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bar: torch.Tensor


def make_schedule(
    steps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> DiffusionSchedule:
    betas = torch.linspace(beta_start, beta_end, steps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bar=alpha_bar)


class TrajectoryConditionedLatentDiffusion(nn.Module):
    def __init__(
        self,
        field_channels: int,
        latent_channels: int,
        bg_channels: int,
        diffusion_steps: int,
        beta_start: float,
        beta_end: float,
        residual_forecast: bool = True,
    ) -> None:
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.residual_forecast = bool(residual_forecast)
        self.autoencoder = Autoencoder2D(field_channels, latent_channels)
        self.bg_proj = nn.Conv2d(bg_channels, latent_channels, 1)
        self.rasterizer = TrajectoryRasterizer()
        self.cond_proj = nn.Conv2d(latent_channels + 1, latent_channels, 1)
        self.denoiser = DenoiserUNet(
            latent_channels=latent_channels,
            cond_channels=latent_channels,
            hidden=max(128, latent_channels * 8),
        )
        self.beta_start = beta_start
        self.beta_end = beta_end

    def _make_cond(
        self,
        z_bg_map: torch.Tensor,
        traj_pred: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        latent_hw: tuple[int, int],
    ) -> torch.Tensor:
        # returns [B,T,Cl,Hl,Wl]
        b, t, _, _ = traj_pred.shape
        heat = self.rasterizer(traj_pred, lat, lon)[:, :, None]
        heat = F.interpolate(
            heat.reshape(b * t, 1, heat.shape[-2], heat.shape[-1]),
            size=latent_hw,
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, 1, latent_hw[0], latent_hw[1])
        bg = self.bg_proj(
            F.interpolate(z_bg_map, size=latent_hw, mode="bilinear", align_corners=False)
        )
        bg = bg[:, None].repeat(1, t, 1, 1, 1)
        cond = self.cond_proj(torch.cat([bg.reshape(b * t, -1, *latent_hw), heat.reshape(b * t, 1, *latent_hw)], dim=1))
        return cond.reshape(b, t, -1, *latent_hw)

    def training_step(
        self,
        target_fields: torch.Tensor,
        z_bg_map: torch.Tensor,
        traj_pred: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        history_last: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # target_fields [B,T,C,H,W]
        b, t, c, h, w = target_fields.shape
        if self.residual_forecast and history_last is not None:
            base = history_last[:, None].repeat(1, t, 1, 1, 1)
            train_target = target_fields - base
        else:
            base = torch.zeros_like(target_fields)
            train_target = target_fields

        flat = train_target.reshape(b * t, c, h, w)
        z = self.autoencoder.encode(flat)
        _, cl, hl, wl = z.shape
        ae_recon = self.autoencoder.decode(z, target_hw=(h, w))
        loss_ae = F.l1_loss(ae_recon, flat)

        cond = self._make_cond(z_bg_map, traj_pred, lat, lon, latent_hw=(hl, wl))
        cond_flat = cond.reshape(b * t, -1, hl, wl)

        schedule = make_schedule(
            steps=self.diffusion_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=z.device,
        )
        time_idx = torch.randint(0, self.diffusion_steps, (b * t,), device=z.device)
        alpha_bar = schedule.alpha_bar[time_idx][:, None, None, None]
        noise = torch.randn_like(z)
        x_t = torch.sqrt(alpha_bar) * z + torch.sqrt(1.0 - alpha_bar) * noise
        pred_noise = self.denoiser(x_t, cond_flat, time_idx.float())
        loss = F.mse_loss(pred_noise, noise) + 0.1 * loss_ae

        z_hat = (x_t - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar + 1e-8)
        residual = self.autoencoder.decode(z_hat, target_hw=(h, w)).reshape(b, t, c, h, w)
        recon = residual + base
        return {
            "loss_diffusion": loss,
            "loss_autoencoder": loss_ae,
            "recon_fields": recon,
        }

    @torch.no_grad()
    def sample(
        self,
        z_bg_map: torch.Tensor,
        traj_pred: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        horizon: int,
        field_shape: tuple[int, int, int],
        history_last: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # field_shape = (C,H,W)
        b = z_bg_map.shape[0]
        c, h, w = field_shape
        dummy = torch.zeros((b * horizon, c, h, w), device=z_bg_map.device)
        z = self.autoencoder.encode(dummy)
        _, cl, hl, wl = z.shape
        cond = self._make_cond(z_bg_map, traj_pred, lat, lon, latent_hw=(hl, wl))
        cond = cond.reshape(b * horizon, -1, hl, wl)

        schedule = make_schedule(
            steps=self.diffusion_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=z_bg_map.device,
        )
        x = torch.randn((b * horizon, cl, hl, wl), device=z_bg_map.device)
        for ti in reversed(range(self.diffusion_steps)):
            t_step = torch.full((b * horizon,), ti, device=x.device, dtype=torch.float32)
            eps = self.denoiser(x, cond, t_step)
            alpha = schedule.alphas[ti]
            alpha_bar = schedule.alpha_bar[ti]
            beta = schedule.betas[ti]
            mean = (x - (beta / torch.sqrt(1.0 - alpha_bar + 1e-8)) * eps) / torch.sqrt(alpha + 1e-8)
            if ti > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        residual = self.autoencoder.decode(x, target_hw=(h, w)).reshape(b, horizon, c, h, w)
        if self.residual_forecast and history_last is not None:
            base = history_last[:, None].repeat(1, horizon, 1, 1, 1)
            out = residual + base
        else:
            out = residual
        return out
