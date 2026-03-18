from __future__ import annotations

import torch

from ltg_net.config import load_config
from ltg_net.losses import CompositeLoss
from ltg_net.models import LTGNet


def _make_batch(cfg: dict, batch_size: int = 2, h: int = 64, w: int = 64) -> dict[str, torch.Tensor]:
    t_hist = int(cfg["data"]["history_steps"])
    t_fut = int(cfg["data"]["forecast_steps"])
    c = len(cfg["data"]["variables"])
    o = int(cfg["data"]["objects_per_step"])
    lat = torch.linspace(60.0, -10.0, h)
    lon = torch.linspace(90.0, 200.0, w)
    return {
        "x_hist": torch.randn(batch_size, t_hist, c, h, w),
        "y_future": torch.randn(batch_size, t_fut, c, h, w),
        "traj_hist": torch.randn(batch_size, t_hist, o, 2).mul(3.0).add(torch.tensor([20.0, 150.0])),
        "traj_future": torch.randn(batch_size, t_fut, o, 2).mul(3.0).add(torch.tensor([20.0, 150.0])),
        "lat": lat[None].repeat(batch_size, 1),
        "lon": lon[None].repeat(batch_size, 1),
        "time_index": torch.arange(batch_size),
    }


def test_stage1_forward_and_loss() -> None:
    cfg = load_config("configs/stage1_regional.yaml")
    model = LTGNet(cfg, in_channels=len(cfg["data"]["variables"]))
    loss_fn = CompositeLoss(cfg)
    batch = _make_batch(cfg)
    outputs = model(batch)
    assert outputs["field_pred"].shape == batch["y_future"].shape
    assert outputs["traj_pred"].shape == batch["traj_future"].shape
    loss, details = loss_fn(outputs, batch)
    assert torch.isfinite(loss)
    assert "loss_total" in details


def test_stage3_diffusion_forward_and_loss() -> None:
    cfg = load_config("configs/stage3_global.yaml")
    model = LTGNet(cfg, in_channels=len(cfg["data"]["variables"]))
    loss_fn = CompositeLoss(cfg)
    batch = _make_batch(cfg, h=48, w=48)
    outputs = model(batch)
    assert outputs["field_pred"].shape == batch["y_future"].shape
    assert outputs["traj_pred"].shape == batch["traj_future"].shape
    loss, details = loss_fn(outputs, batch)
    assert torch.isfinite(loss)
    assert details["loss_diff"] >= 0
