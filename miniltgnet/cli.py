from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from miniltgnet.config import load_config
from miniltgnet.data import build_dataloaders, prepare_data
from miniltgnet.losses import MiniCompositeLoss
from miniltgnet.model import build_model
from miniltgnet.trainer import MiniTrainer, evaluate_model, setup_logger


def _resolve_device(cfg: dict) -> torch.device:
    requested = cfg["experiment"].get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _build_model_from_dataloader(cfg: dict, dataloaders: dict, device: torch.device):
    sample = next(iter(dataloaders["train"]))
    in_channels = int(sample["x_hist"].shape[2])
    model = build_model(cfg, in_channels=in_channels).to(device)
    return model, in_channels


def prepare(config: str) -> None:
    cfg = load_config(config)
    prepare_data(cfg)
    print("Prepared stats:", cfg["data"]["norm_stats_path"])
    print("Track cache used:", cfg["data"]["track_cache_path"])


def train(config: str) -> None:
    cfg = load_config(config)
    _seed_everything(
        int(cfg["experiment"]["seed"]),
        deterministic=bool(cfg["experiment"].get("deterministic", False)),
    )
    device = _resolve_device(cfg)
    output_dir = Path(cfg["experiment"]["output_dir"])
    logger = setup_logger(cfg["experiment"]["name"], output_dir)
    logger.info("Device: %s", device)

    dls = build_dataloaders(cfg)
    model, in_channels = _build_model_from_dataloader(cfg, dls, device)
    logger.info("Model in_channels: %d", in_channels)
    loss_fn = MiniCompositeLoss(cfg).to(device)

    trainer = MiniTrainer(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        dataloaders=dls,
        output_dir=output_dir,
        logger=logger,
        device=device,
    )
    resume = cfg["experiment"].get("resume_checkpoint")
    if resume:
        trainer.load_checkpoint(resume)
    trainer.fit()
    logger.info("Training completed.")


def evaluate(config: str, checkpoint: str, split: str = "test") -> dict[str, float]:
    cfg = load_config(config)
    _seed_everything(
        int(cfg["experiment"]["seed"]),
        deterministic=bool(cfg["experiment"].get("deterministic", False)),
    )
    device = _resolve_device(cfg)
    output_dir = Path(cfg["experiment"]["output_dir"])
    logger = setup_logger(cfg["experiment"]["name"] + "_eval", output_dir)
    dls = build_dataloaders(cfg)
    model, in_channels = _build_model_from_dataloader(cfg, dls, device)
    logger.info("Model in_channels: %d", in_channels)
    loss_fn = MiniCompositeLoss(cfg).to(device)

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists() and ckpt_path.name == "best.pt":
        epoch_candidates = sorted(ckpt_path.parent.glob("epoch_*.pt"))
        if epoch_candidates:
            ckpt_path = epoch_candidates[-1]
            logger.warning(
                "Checkpoint %s not found; fallback to latest epoch checkpoint: %s",
                checkpoint,
                ckpt_path,
            )
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        use_ema = bool(cfg.get("evaluation", {}).get("use_ema_for_eval", True))
        if use_ema and "model_ema" in ckpt and ckpt["model_ema"] is not None:
            state = ckpt["model_ema"]
        else:
            state = ckpt["model"] if "model" in ckpt else ckpt
    else:
        state = ckpt
    model.load_state_dict(state)

    metrics = evaluate_model(
        model=model,
        dataloader=dls[split],
        loss_fn=loss_fn,
        device=device,
        max_batches=int(cfg["evaluation"].get("test_max_batches", 0)),
        spectral_wavenumbers=int(cfg["evaluation"].get("spectral_wavenumbers", 32)),
        inference_config=cfg,
    )
    logger.info("Evaluation metrics (%s): %s", split, metrics)
    print(metrics)
    return metrics


def sanity(config: str, steps: int = 2, backward: bool = True) -> None:
    cfg = load_config(config)
    device = _resolve_device(cfg)
    dls = build_dataloaders(cfg)
    model, in_channels = _build_model_from_dataloader(cfg, dls, device)
    loss_fn = MiniCompositeLoss(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    loss_fn.train()

    print(f"[sanity] device={device}, in_channels={in_channels}")
    print(f"[sanity] train_batches={len(dls['train'])}, val_batches={len(dls['val'])}, test_batches={len(dls['test'])}")
    checked = 0
    for batch in dls["train"]:
        if checked >= steps:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch)
        loss, details = loss_fn(outputs, batch)
        assert torch.isfinite(loss), f"loss not finite: {float(loss.detach().cpu().item())}"
        for k, v in details.items():
            if isinstance(v, torch.Tensor):
                assert torch.isfinite(v).all(), f"detail {k} has non-finite values"
        print(
            f"[sanity] step={checked} field_pred={tuple(outputs['field_pred'].shape)} "
            f"traj_pred={tuple(outputs['traj_pred'].shape)} loss={float(loss.detach().cpu().item()):.6f}"
        )
        if backward:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            print(f"[sanity] step={checked} backward=ok")
        checked += 1
    print(f"[sanity] checked_steps={checked}, status=ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLTGNet CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="compute normalization stats for miniltgnet")
    p_prepare.add_argument("--config", required=True)

    p_train = sub.add_parser("train", help="train miniltgnet")
    p_train.add_argument("--config", required=True)

    p_eval = sub.add_parser("evaluate", help="evaluate checkpoint")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])

    p_sanity = sub.add_parser("sanity", help="quick model/data sanity check")
    p_sanity.add_argument("--config", required=True)
    p_sanity.add_argument("--steps", type=int, default=2)
    p_sanity.add_argument("--backward", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "train":
        train(args.config)
    elif args.command == "evaluate":
        evaluate(args.config, args.checkpoint, args.split)
    elif args.command == "sanity":
        sanity(args.config, args.steps, args.backward)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
