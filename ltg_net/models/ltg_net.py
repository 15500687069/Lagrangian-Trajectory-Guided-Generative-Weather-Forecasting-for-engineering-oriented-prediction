from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ltg_net.models.diffusion import TrajectoryConditionedLatentDiffusion
from ltg_net.models.encoder import SphereSpatiotemporalEncoder
from ltg_net.models.event_modulation import EventAdaptiveModulation
from ltg_net.models.generator_unet import TrajectoryConditionedUNetGenerator
from ltg_net.models.trajectory import DeterministicTrajectoryPredictor, NeuralSDETrajectoryPredictor


class LTGNet(nn.Module):
    def __init__(self, cfg: dict[str, Any], in_channels: int) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        enc_cfg = model_cfg["encoder"]
        hidden_dim = int(model_cfg["hidden_dim"])
        history_steps = int(cfg["data"]["history_steps"])

        self.encoder = SphereSpatiotemporalEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            patch_size=int(enc_cfg["patch_size"]),
            depth=int(enc_cfg["depth"]),
            heads=int(enc_cfg["heads"]),
            mlp_ratio=float(enc_cfg["mlp_ratio"]),
            dropout=float(enc_cfg["dropout"]),
            max_history_steps=history_steps,
        )
        traj_cfg = model_cfg["trajectory"]
        self.trajectory_mode = traj_cfg["mode"]
        if self.trajectory_mode == "stochastic":
            self.traj_predictor = NeuralSDETrajectoryPredictor(
                context_dim=hidden_dim,
                hidden_dim=int(traj_cfg["hidden_dim"]),
                diffusion_min=float(traj_cfg["diffusion_min"]),
                diffusion_max=float(traj_cfg["diffusion_max"]),
                num_samples=int(traj_cfg["num_samples"]),
                max_step_deg=float(traj_cfg.get("max_step_deg", 0.0)),
            )
        else:
            self.traj_predictor = DeterministicTrajectoryPredictor(
                context_dim=hidden_dim,
                hidden_dim=int(traj_cfg["hidden_dim"]),
                num_layers=int(traj_cfg["num_layers"]),
                max_step_deg=float(traj_cfg.get("max_step_deg", 0.0)),
            )

        gen_cfg = model_cfg["generator"]
        self.generator_type = gen_cfg["type"]
        if self.generator_type == "latent_diffusion":
            self.generator = TrajectoryConditionedLatentDiffusion(
                field_channels=in_channels,
                latent_channels=max(16, hidden_dim // 8),
                bg_channels=hidden_dim,
                diffusion_steps=int(gen_cfg["diffusion_steps"]),
                beta_start=float(gen_cfg["diffusion_beta_start"]),
                beta_end=float(gen_cfg["diffusion_beta_end"]),
                residual_forecast=bool(gen_cfg.get("residual_forecast", True)),
            )
        else:
            self.generator = TrajectoryConditionedUNetGenerator(
                field_channels=in_channels,
                latent_channels=hidden_dim,
                base_channels=int(gen_cfg["base_channels"]),
                channel_mults=[int(v) for v in gen_cfg["channel_mults"]],
                num_res_blocks=int(gen_cfg["num_res_blocks"]),
                dropout=float(gen_cfg["dropout"]),
                residual_forecast=bool(gen_cfg.get("residual_forecast", True)),
                trend_scale=float(gen_cfg.get("trend_scale", 0.35)),
                residual_scale=float(gen_cfg.get("residual_scale", 1.0)),
            )

        mod_cfg = model_cfg["event_modulation"]
        self.modulation_enabled = bool(mod_cfg["enabled"])
        self.event_modulator = EventAdaptiveModulation(
            curvature_scale=float(mod_cfg["curvature_scale"]),
            gradient_scale=float(mod_cfg["gradient_scale"]),
            dyn_scale=float(mod_cfg["dyn_scale"]),
            thermo_scale=float(mod_cfg["thermo_scale"]),
            alpha_floor=float(mod_cfg["alpha_floor"]),
            alpha_ceiling=float(mod_cfg["alpha_ceiling"]),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_hist = batch["x_hist"].float()
        traj_hist = batch["traj_hist"].float()
        lat = batch["lat"][0].float() if batch["lat"].ndim == 2 else batch["lat"].float()
        lon = batch["lon"][0].float() if batch["lon"].ndim == 2 else batch["lon"].float()
        horizon = int(batch["y_future"].shape[1]) if "y_future" in batch else int(traj_hist.shape[1])

        encoded = self.encoder(x_hist=x_hist, lat=lat, lon=lon)
        z_bg_map = encoded["z_bg_map"]
        z_bg_global = encoded["z_bg_global"]

        if self.trajectory_mode == "stochastic":
            traj_pack = self.traj_predictor(z_bg_global=z_bg_global, traj_hist=traj_hist, horizon=horizon)
            traj_pred = traj_pack["traj_mean"]
        else:
            traj_pack = {}
            traj_pred = self.traj_predictor(z_bg_global=z_bg_global, traj_hist=traj_hist, horizon=horizon)

        aux_losses: dict[str, torch.Tensor] = {}
        history_last = x_hist[:, -1]
        if self.generator_type == "latent_diffusion":
            if self.training and "y_future" in batch:
                diff_out = self.generator.training_step(
                    target_fields=batch["y_future"].float(),
                    z_bg_map=z_bg_map,
                    traj_pred=traj_pred,
                    lat=lat,
                    lon=lon,
                    history_last=history_last,
                )
                field_pred = diff_out["recon_fields"]
                aux_losses["loss_diffusion"] = diff_out["loss_diffusion"]
            else:
                c = x_hist.shape[2]
                h = x_hist.shape[3]
                w = x_hist.shape[4]
                field_pred = self.generator.sample(
                    z_bg_map=z_bg_map,
                    traj_pred=traj_pred,
                    lat=lat,
                    lon=lon,
                    horizon=horizon,
                    field_shape=(c, h, w),
                    history_last=history_last,
                )
        else:
            field_pred = self.generator(
                x_hist=x_hist,
                z_bg_map=z_bg_map,
                traj_pred=traj_pred,
                lat=lat,
                lon=lon,
            )

        if self.modulation_enabled:
            alpha = self.event_modulator(traj_pred=traj_pred, z_bg_map=z_bg_map, x_hist=x_hist)
            field_pred = self.event_modulator.modulate(field_pred, alpha)
        else:
            alpha = torch.ones(
                field_pred.shape[0],
                field_pred.shape[1],
                1,
                1,
                1,
                device=field_pred.device,
            )

        return {
            "field_pred": field_pred,
            "traj_pred": traj_pred,
            "traj_pack": traj_pack,
            "z_bg_map": z_bg_map,
            "z_bg_global": z_bg_global,
            "alpha": alpha,
            **aux_losses,
        }
