from __future__ import annotations

import torch
import torch.nn.functional as F


def _safe_quantile(values: torch.Tensor, quantile: float, max_elements: int) -> torch.Tensor:
    flat = values.reshape(-1)
    if max_elements > 0 and flat.numel() > max_elements:
        stride = max(1, flat.numel() // max_elements)
        flat = flat[::stride]
        if flat.numel() > max_elements:
            flat = flat[:max_elements]
    if not torch.is_floating_point(flat) or flat.dtype in {torch.float16, torch.bfloat16}:
        flat = flat.float()
    q = min(1.0, max(0.0, float(quantile)))
    return torch.quantile(flat, q)


def field_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float,
    l2_weight: float,
    extreme_weight: float = 0.0,
    extreme_quantile: float = 0.98,
    max_quantile_elements: int = 2_000_000,
) -> torch.Tensor:
    l1 = F.l1_loss(pred, target)
    l2 = F.mse_loss(pred, target)
    loss = l1_weight * l1 + l2_weight * l2
    if extreme_weight > 0:
        thresh = _safe_quantile(target, quantile=extreme_quantile, max_elements=max_quantile_elements)
        mask = (target >= thresh).float()
        # Emphasize tail behavior without destabilizing all-grid optimization.
        extreme_l1 = (torch.abs(pred - target) * mask).sum() / (mask.sum() + 1e-6)
        loss = loss + extreme_weight * extreme_l1
    return loss
