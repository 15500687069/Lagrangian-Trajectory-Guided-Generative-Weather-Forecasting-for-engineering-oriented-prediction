from __future__ import annotations

import torch

from ltg_net.utils.spectra import spectral_distance


def spectral_consistency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_wavenumber: int,
) -> torch.Tensor:
    # pred/target [B,T,C,H,W]
    losses = []
    for ti in range(pred.shape[1]):
        losses.append(spectral_distance(pred[:, ti], target[:, ti], max_wavenumber=max_wavenumber))
    return torch.stack(losses).mean()
