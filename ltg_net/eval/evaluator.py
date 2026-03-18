from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from tqdm import tqdm

from ltg_net.train.loops import move_batch_to_device
from ltg_net.utils.metrics import acc, extreme_f1, rmse, spectral_metric, track_mae


class LTGEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        max_wavenumber: int = 64,
        max_batches: int = 0,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.max_wavenumber = max_wavenumber
        self.max_batches = int(max_batches)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        meter: dict[str, list[float]] = defaultdict(list)
        for step_idx, batch in enumerate(tqdm(self.dataloader, desc="evaluate", leave=False)):
            if self.max_batches > 0 and step_idx >= self.max_batches:
                break
            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(batch)
            pred_field = outputs["field_pred"]
            pred_traj = outputs["traj_pred"]
            target_field = batch["y_future"].float()
            target_traj = batch["traj_future"].float()

            meter["rmse"].append(float(rmse(pred_field, target_field).item()))
            meter["acc"].append(float(acc(pred_field, target_field).item()))
            meter["track_mae"].append(float(track_mae(pred_traj, target_traj).item()))
            meter["extreme_f1"].append(float(extreme_f1(pred_field, target_field).item()))
            meter["spectral_distance"].append(
                float(spectral_metric(pred_field[:, 0], target_field[:, 0], self.max_wavenumber).item())
            )
        return {k: sum(v) / max(1, len(v)) for k, v in meter.items()}
