from __future__ import annotations

import torch
import torch.nn.functional as F


def trajectory_supervision_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(pred, target)


def trajectory_uncertainty_regularization(
    sigma_samples: torch.Tensor | None,
    min_sigma: float = 0.01,
) -> torch.Tensor:
    if sigma_samples is None:
        return torch.tensor(0.0)
    # discourage collapse to zero variance and exploding diffusion
    low_penalty = torch.relu(min_sigma - sigma_samples).mean()
    high_penalty = torch.relu(sigma_samples - 0.5).mean()
    return low_penalty + high_penalty
