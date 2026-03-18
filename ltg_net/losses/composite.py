from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ltg_net.data.normalization import load_stats
from ltg_net.losses.advection import advection_consistency_loss
from ltg_net.losses.field import field_reconstruction_loss
from ltg_net.losses.physics import physics_consistency_loss
from ltg_net.losses.spectral import spectral_consistency_loss
from ltg_net.losses.trajectory import trajectory_supervision_loss, trajectory_uncertainty_regularization


def _find_channel(variables: list[str], candidates: list[str]) -> int | None:
    for i, name in enumerate(variables):
        lname = name.lower()
        for c in candidates:
            if lname == c or lname.startswith(c):
                return i
    return None


class CompositeLoss(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.loss_cfg = cfg["loss"]
        self.variables = list(cfg["data"]["variables"])
        self.current_epoch = 0
        self.compute_inactive_losses = bool(self.loss_cfg.get("compute_inactive_losses", True))

        norm_cfg = self.loss_cfg.get("normalization", {})
        self.normalization_enabled = bool(norm_cfg.get("enabled", False))
        self.norm_ema_decay = float(norm_cfg.get("ema_decay", 0.98))
        self.norm_eps = float(norm_cfg.get("eps", 1e-6))
        self.combine_mode = self.loss_cfg.get("combine_mode", "fixed_weighted")

        self.curriculum_cfg = self.loss_cfg.get("curriculum", {})
        self.curriculum_enabled = bool(self.curriculum_cfg.get("enabled", False))

        self.register_buffer("ema_field", torch.tensor(1.0))
        self.register_buffer("ema_traj", torch.tensor(1.0))
        self.register_buffer("ema_adv", torch.tensor(1.0))
        self.register_buffer("ema_phys", torch.tensor(1.0))
        self.register_buffer("ema_spec", torch.tensor(1.0))
        self.register_buffer("ema_diff", torch.tensor(1.0))

        self.u_channel = _find_channel(self.variables, ["u", "u850", "u10"])
        self.v_channel = _find_channel(self.variables, ["v", "v850", "v10"])
        self.t_channel = _find_channel(self.variables, ["t", "t850", "t2m"])
        self.z_channel = _find_channel(self.variables, ["z", "z500", "z850", "gh"])
        self.q_channel = _find_channel(self.variables, ["q", "q850", "q2m"])

        if self.u_channel is None or self.v_channel is None:
            raise ValueError("u/v channel required for advection and physics losses.")

        stats_path = Path(str(cfg["data"].get("norm_stats_path", "")))
        means = []
        stds = []
        if stats_path.exists():
            stats = load_stats(stats_path, self.variables)
            for var in self.variables:
                means.append(float(stats[var]["mean"]))
                stds.append(float(stats[var]["std"]))
        else:
            means = [0.0 for _ in self.variables]
            stds = [1.0 for _ in self.variables]
        self.register_buffer("channel_means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("channel_stds", torch.tensor(stds, dtype=torch.float32))

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _denormalize_fields(self, field: torch.Tensor) -> torch.Tensor:
        # field: [B, T, C, H, W]
        c = field.shape[2]
        cm = self.channel_means[:c].to(field.device)
        cs = self.channel_stds[:c].to(field.device)
        return field * cs[None, None, :, None, None] + cm[None, None, :, None, None]

    def _curriculum_factor(self, key: str) -> float:
        if not self.curriculum_enabled:
            return 1.0
        start_epoch = int(self.curriculum_cfg.get(f"{key}_start_epoch", 0))
        ramp_epochs = int(self.curriculum_cfg.get("ramp_epochs", 0))
        if self.current_epoch < start_epoch:
            return 0.0
        if ramp_epochs <= 0:
            return 1.0
        factor = (self.current_epoch - start_epoch + 1) / max(1, ramp_epochs)
        return float(min(1.0, max(0.0, factor)))

    def _effective_lambdas(self) -> dict[str, float]:
        lambdas = {
            "field": float(self.loss_cfg["lambda_field"]),
            "traj": float(self.loss_cfg["lambda_traj"]),
            "adv": float(self.loss_cfg["lambda_adv"]) * self._curriculum_factor("adv"),
            "phys": float(self.loss_cfg["lambda_phys"]) * self._curriculum_factor("phys"),
            "spec": float(self.loss_cfg["lambda_spec"]) * self._curriculum_factor("spec"),
            "diff": float(self.loss_cfg.get("lambda_diff", 1.0)),
        }
        return lambdas

    def _normalized(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not self.normalization_enabled:
            return value
        ema = getattr(self, f"ema_{name}")
        if self.training and torch.isfinite(value):
            updated = self.norm_ema_decay * ema + (1.0 - self.norm_ema_decay) * value.detach().clamp(
                min=self.norm_eps
            )
            ema.copy_(updated)
        return value / (ema.detach() + self.norm_eps)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_field = outputs["field_pred"]
        pred_traj = outputs["traj_pred"]
        target_field = batch["y_future"].float().to(pred_field.device)
        target_traj = batch["traj_future"].float().to(pred_field.device)
        pred_field = torch.nan_to_num(pred_field, nan=0.0, posinf=1e6, neginf=-1e6)
        target_field = torch.nan_to_num(target_field, nan=0.0, posinf=1e6, neginf=-1e6)
        pred_traj = torch.nan_to_num(pred_traj, nan=0.0, posinf=0.0, neginf=0.0)
        target_traj = torch.nan_to_num(target_traj, nan=0.0, posinf=0.0, neginf=0.0)
        lambdas = self._effective_lambdas()
        zero = torch.tensor(0.0, device=pred_field.device)

        def _active(weight: float) -> bool:
            return self.compute_inactive_losses or abs(weight) > 0.0

        compute_field = _active(lambdas["field"])
        compute_traj = _active(lambdas["traj"])
        compute_adv = _active(lambdas["adv"])
        compute_phys = _active(lambdas["phys"])
        compute_spec = _active(lambdas["spec"])
        compute_diff = _active(lambdas["diff"])

        need_latlon = compute_adv or compute_phys
        if need_latlon:
            lat = batch["lat"][0].to(pred_field.device) if batch["lat"].ndim == 2 else batch["lat"].to(pred_field.device)
            lon = batch["lon"][0].to(pred_field.device) if batch["lon"].ndim == 2 else batch["lon"].to(pred_field.device)
        else:
            lat = None
            lon = None

        need_phys_field = compute_adv or compute_phys
        if need_phys_field:
            pred_field_phys = self._denormalize_fields(pred_field)
            target_field_phys = self._denormalize_fields(target_field)
        else:
            pred_field_phys = None
            target_field_phys = None

        if compute_field:
            l_field_raw = field_reconstruction_loss(
                pred=pred_field,
                target=target_field,
                l1_weight=float(self.loss_cfg["field_l1_weight"]),
                l2_weight=float(self.loss_cfg["field_l2_weight"]),
                extreme_weight=float(self.loss_cfg.get("field_extreme_weight", 0.0)),
                extreme_quantile=float(self.loss_cfg.get("field_extreme_quantile", 0.98)),
                max_quantile_elements=int(self.loss_cfg.get("field_extreme_max_elements", 2_000_000)),
            )
        else:
            l_field_raw = zero

        if compute_traj:
            l_traj_raw = trajectory_supervision_loss(pred_traj, target_traj)
        else:
            l_traj_raw = zero

        if compute_adv:
            l_adv_raw = advection_consistency_loss(
                traj_pred=pred_traj,
                field_pred=pred_field_phys,
                lat_axis=lat,
                lon_axis=lon,
                u_channel=self.u_channel,
                v_channel=self.v_channel,
            )
        else:
            l_adv_raw = zero

        phys_details: dict[str, torch.Tensor] = {}
        if compute_phys:
            strict_pde_cfg = self.loss_cfg.get("physics", {}).get("strict_pde", {})
            l_phys_out = physics_consistency_loss(
                pred=pred_field_phys,
                target=target_field_phys,
                lat_axis=lat,
                lon_axis=lon,
                u_channel=self.u_channel,
                v_channel=self.v_channel,
                t_channel=self.t_channel,
                z_channel=self.z_channel,
                q_channel=self.q_channel,
                divergence_weight=float(self.loss_cfg["physics"]["divergence_weight"]),
                moist_static_energy_weight=float(self.loss_cfg["physics"]["moist_static_energy_weight"]),
                strict_cfg=strict_pde_cfg,
                return_details=True,
            )
            l_phys_raw, phys_details = l_phys_out
        else:
            l_phys_raw = zero

        if compute_spec:
            l_spec_raw = spectral_consistency_loss(
                pred=pred_field,
                target=target_field,
                max_wavenumber=int(self.loss_cfg["spectral_wavenumbers"]),
            )
        else:
            l_spec_raw = zero

        traj_pack = outputs.get("traj_pack", {})
        sigma_samples = traj_pack.get("sigma_samples")
        if sigma_samples is not None:
            l_unc = trajectory_uncertainty_regularization(sigma_samples=sigma_samples)
        else:
            l_unc = torch.tensor(0.0, device=pred_field.device)

        l_diff_raw = outputs.get("loss_diffusion", torch.tensor(0.0, device=pred_field.device))
        l_diff_raw = l_diff_raw if isinstance(l_diff_raw, torch.Tensor) else torch.tensor(0.0, device=pred_field.device)
        if not compute_diff:
            l_diff_raw = zero

        l_field = self._normalized("field", l_field_raw)
        l_traj = self._normalized("traj", l_traj_raw)
        l_adv = self._normalized("adv", l_adv_raw)
        l_phys = self._normalized("phys", l_phys_raw)
        l_spec = self._normalized("spec", l_spec_raw)
        l_diff = self._normalized("diff", l_diff_raw)

        total_raw = (
            lambdas["field"] * l_field_raw
            + lambdas["traj"] * l_traj_raw
            + lambdas["adv"] * l_adv_raw
            + lambdas["phys"] * l_phys_raw
            + lambdas["spec"] * l_spec_raw
            + 0.1 * l_unc
            + lambdas["diff"] * l_diff_raw
        )
        total_bal = (
            lambdas["field"] * l_field
            + lambdas["traj"] * l_traj
            + lambdas["adv"] * l_adv
            + lambdas["phys"] * l_phys
            + lambdas["spec"] * l_spec
            + 0.1 * l_unc
            + lambdas["diff"] * l_diff
        )
        total = total_bal if self.combine_mode == "normalized_weighted" else total_raw
        details = {
            "loss_total": total.detach(),
            "loss_total_raw": total_raw.detach(),
            "loss_total_bal": total_bal.detach(),
            "loss_field": l_field_raw.detach(),
            "loss_traj": l_traj_raw.detach(),
            "loss_adv": l_adv_raw.detach(),
            "loss_phys": l_phys_raw.detach(),
            "loss_spec": l_spec_raw.detach(),
            "loss_field_bal": l_field.detach(),
            "loss_traj_bal": l_traj.detach(),
            "loss_adv_bal": l_adv.detach(),
            "loss_phys_bal": l_phys.detach(),
            "loss_spec_bal": l_spec.detach(),
            "loss_unc": l_unc.detach(),
            "loss_diff": l_diff_raw.detach(),
            "lambda_adv_eff": torch.tensor(lambdas["adv"], device=pred_field.device),
            "lambda_phys_eff": torch.tensor(lambdas["phys"], device=pred_field.device),
            "lambda_spec_eff": torch.tensor(lambdas["spec"], device=pred_field.device),
            "lambda_diff_eff": torch.tensor(lambdas["diff"], device=pred_field.device),
        }
        details.update(phys_details)
        return total, details
