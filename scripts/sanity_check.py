from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltg_net.config import load_config
from ltg_net.data.datamodule import build_dataloaders
from ltg_net.losses import CompositeLoss
from ltg_net.models import LTGNet


def _resolve_device(cfg: dict) -> torch.device:
    requested = cfg["experiment"].get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check LTG-Net data/model/loss pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=2, help="Number of train batches to check.")
    parser.add_argument("--backward", action="store_true", help="Run backward pass for each checked batch.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(cfg)
    dls = build_dataloaders(cfg)
    sample = next(iter(dls["train"]))
    in_channels = int(sample["x_hist"].shape[2])

    model = LTGNet(cfg=cfg, in_channels=in_channels).to(device)
    loss_fn = CompositeLoss(cfg).to(device)
    model.train()
    loss_fn.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print(f"[sanity] device={device}, in_channels={in_channels}")
    print(f"[sanity] train_batches={len(dls['train'])}, val_batches={len(dls['val'])}, test_batches={len(dls['test'])}")

    checked = 0
    for batch in dls["train"]:
        if checked >= args.steps:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch)
        loss, details = loss_fn(outputs, batch)

        assert torch.isfinite(loss), f"loss not finite: {float(loss.detach().cpu().item())}"
        for name, value in details.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), f"detail {name} has non-finite values"

        print(
            f"[sanity] step={checked} "
            f"field_pred={tuple(outputs['field_pred'].shape)} "
            f"traj_pred={tuple(outputs['traj_pred'].shape)} "
            f"loss={float(loss.detach().cpu().item()):.6f}"
        )

        if args.backward:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            print(f"[sanity] step={checked} backward=ok")

        checked += 1

    print(f"[sanity] checked_steps={checked}, status=ok")


if __name__ == "__main__":
    main()
