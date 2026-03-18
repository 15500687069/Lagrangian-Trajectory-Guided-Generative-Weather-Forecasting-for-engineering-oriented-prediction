from __future__ import annotations

import torch
import torch.nn as nn


def _clamp_geo(state: torch.Tensor) -> torch.Tensor:
    lat = state[..., 0].clamp(-90.0, 90.0)
    lon = state[..., 1] % 360.0
    return torch.stack([lat, lon], dim=-1)


class DeterministicTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        num_layers: int,
        max_step_deg: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_step_deg = float(max_step_deg)
        self.traj_embed = nn.Linear(2, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.encoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        z_bg_global: torch.Tensor,
        traj_hist: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        # traj_hist: [B, T, O, 2]
        b, t, o, _ = traj_hist.shape
        ctx = self.context_proj(z_bg_global)  # [B, H]
        traj = traj_hist.permute(0, 2, 1, 3).reshape(b * o, t, 2)
        emb = self.traj_embed(traj)
        _, h = self.encoder_gru(emb)
        h = h[-1]  # [B*O, H]

        ctx_o = ctx[:, None, :].repeat(1, o, 1).reshape(b * o, -1)
        h = h + ctx_o
        state = traj[:, -1]  # [B*O, 2]
        outputs = []
        for _ in range(horizon):
            inp = self.traj_embed(state)
            h = self.decoder_cell(inp, h)
            delta = self.delta_head(h)
            if self.max_step_deg > 0:
                delta = self.max_step_deg * torch.tanh(delta / max(1e-6, self.max_step_deg))
            state = _clamp_geo(state + delta)
            outputs.append(state)
        pred = torch.stack(outputs, dim=1).reshape(b, o, horizon, 2).permute(0, 2, 1, 3)
        return pred


class NeuralSDETrajectoryPredictor(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        diffusion_min: float,
        diffusion_max: float,
        num_samples: int,
        max_step_deg: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.diffusion_min = diffusion_min
        self.diffusion_max = diffusion_max
        self.max_step_deg = float(max_step_deg)
        self.state_proj = nn.Linear(2 + context_dim, hidden_dim)
        self.drift = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.diffusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def _step(self, state: torch.Tensor, context: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([state, context], dim=-1)
        h = self.state_proj(inp)
        drift = self.drift(h)
        sigma = self.diffusion(h)
        sigma = self.diffusion_min + sigma * (self.diffusion_max - self.diffusion_min)
        noise = torch.randn_like(state)
        delta = drift * dt + sigma * noise * (dt ** 0.5)
        if self.max_step_deg > 0:
            delta = self.max_step_deg * torch.tanh(delta / max(1e-6, self.max_step_deg))
        next_state = state + delta
        next_state = _clamp_geo(next_state)
        return next_state, sigma

    def forward(
        self,
        z_bg_global: torch.Tensor,
        traj_hist: torch.Tensor,
        horizon: int,
    ) -> dict[str, torch.Tensor]:
        b, _, o, _ = traj_hist.shape
        context = z_bg_global[:, None, :].repeat(1, o, 1).reshape(b * o, -1)
        init = traj_hist[:, -1].reshape(b * o, 2)
        sample_paths = []
        sample_sigmas = []
        for _ in range(self.num_samples):
            state = init
            states = []
            sigmas = []
            for _ in range(horizon):
                state, sigma = self._step(state, context, dt=1.0)
                states.append(state)
                sigmas.append(sigma)
            sample_paths.append(torch.stack(states, dim=1))
            sample_sigmas.append(torch.stack(sigmas, dim=1))

        paths = torch.stack(sample_paths, dim=0)  # [S, B*O, T, 2]
        sigmas = torch.stack(sample_sigmas, dim=0)
        mean_path = paths.mean(dim=0).reshape(b, o, horizon, 2).permute(0, 2, 1, 3)
        paths = paths.reshape(self.num_samples, b, o, horizon, 2).permute(0, 1, 3, 2, 4)
        sigmas = sigmas.reshape(self.num_samples, b, o, horizon, 2).permute(0, 1, 3, 2, 4)
        return {
            "traj_mean": mean_path,
            "traj_samples": paths,
            "sigma_samples": sigmas,
        }
