from __future__ import annotations

import torch
import torch.nn as nn

from ltg_net.utils.geo import curvature_from_trajectory


class EventAdaptiveModulation(nn.Module):
    def __init__(
        self,
        curvature_scale: float,
        gradient_scale: float,
        dyn_scale: float,
        thermo_scale: float,
        alpha_floor: float,
        alpha_ceiling: float,
    ) -> None:
        super().__init__()
        self.curvature_scale = curvature_scale
        self.gradient_scale = gradient_scale
        self.dyn_scale = dyn_scale
        self.thermo_scale = thermo_scale
        self.alpha_floor = alpha_floor
        self.alpha_ceiling = alpha_ceiling

    def _background_gradient(self, z_bg_map: torch.Tensor) -> torch.Tensor:
        # z_bg_map: [B, D, H, W]
        field = z_bg_map.mean(dim=1)
        gy = torch.zeros_like(field)
        gx = torch.zeros_like(field)
        gy[:, 1:-1] = (field[:, 2:] - field[:, :-2]) * 0.5
        gx[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) * 0.5
        g = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return g.mean(dim=(-1, -2), keepdim=True)

    def _dynamic_score(self, traj: torch.Tensor) -> torch.Tensor:
        # traj: [B, T, O, 2]
        vel = torch.zeros_like(traj)
        vel[:, 1:] = traj[:, 1:] - traj[:, :-1]
        speed = torch.norm(vel, dim=-1)  # [B, T, O]
        return speed.mean(dim=-1, keepdim=True)

    def _thermo_score(self, x_hist: torch.Tensor) -> torch.Tensor:
        # x_hist: [B, T_hist, C, H, W]
        # use temporal std as proxy for thermodynamic instability
        b = x_hist.shape[0]
        std = x_hist.std(dim=1).mean(dim=(1, 2, 3), keepdim=True)
        return std.view(b, 1, 1)

    def forward(
        self,
        traj_pred: torch.Tensor,
        z_bg_map: torch.Tensor,
        x_hist: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _, _ = traj_pred.shape
        curvature = curvature_from_trajectory(traj_pred).mean(dim=-1, keepdim=True)  # [B,T,1]
        bg_grad = self._background_gradient(z_bg_map).repeat(1, t, 1)  # [B,T,1]
        dyn = self._dynamic_score(traj_pred)  # [B,T,1]
        thermo = self._thermo_score(x_hist).repeat(1, t, 1)  # [B,T,1]

        score = (
            self.curvature_scale * curvature
            + self.gradient_scale * bg_grad
            + self.dyn_scale * dyn
            + self.thermo_scale * thermo
        )
        alpha = 1.0 + torch.tanh(score)
        alpha = alpha.clamp(self.alpha_floor, self.alpha_ceiling)
        return alpha[..., None, None]

    @staticmethod
    def modulate(field: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # field [B,T,C,H,W], alpha [B,T,1,1,1]
        return field * alpha
