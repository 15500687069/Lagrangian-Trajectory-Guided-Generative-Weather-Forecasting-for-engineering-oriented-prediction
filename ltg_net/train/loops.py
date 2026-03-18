from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def average_meter(values: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    meter: dict[str, list[float]] = defaultdict(list)
    for item in values:
        for k, v in item.items():
            meter[k].append(float(v.item()))
    return {k: sum(vs) / max(1, len(vs)) for k, vs in meter.items()}
