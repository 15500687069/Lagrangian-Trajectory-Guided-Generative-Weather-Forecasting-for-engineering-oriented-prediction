from __future__ import annotations

import argparse
from pathlib import Path

import torch
import xarray as xr

from ltg_net.config import load_config
from ltg_net.data.datamodule import build_dataloaders
from ltg_net.data.normalization import compute_variable_stats, save_stats
from ltg_net.data.skeleton import SkeletonConfig, build_skeleton_tracks, save_tracks
from ltg_net.eval.evaluator import LTGEvaluator
from ltg_net.losses import CompositeLoss
from ltg_net.models import LTGNet
from ltg_net.train.trainer import LTGTrainer
from ltg_net.utils import set_seed, setup_logger

TRAJ_PREFIXES = ("traj_predictor.",)
ENCODER_PREFIX = "encoder."


def _open_era5(path: str) -> xr.Dataset:
    path_obj = Path(path)
    if path_obj.suffix == ".zarr":
        try:
            import zarr  # noqa: F401
        except Exception as exc:
            raise ImportError("zarr package is required for .zarr datasets.") from exc
        return xr.open_zarr(path_obj, consolidated=False)
    return xr.open_dataset(path_obj)


def _select_region(ds: xr.Dataset, region_cfg: dict) -> xr.Dataset:
    if not region_cfg.get("enabled", False):
        return ds
    lat_min = float(region_cfg["lat_min"])
    lat_max = float(region_cfg["lat_max"])
    lon_min = float(region_cfg["lon_min"])
    lon_max = float(region_cfg["lon_max"])
    lat_values = ds["latitude"].values
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
    ds = ds.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    return ds


def prepare(config: str) -> None:
    cfg = load_config(config)
    data_cfg = cfg["data"]
    ds = _open_era5(data_cfg["era5_path"])
    ds = _select_region(ds, data_cfg["region"])

    split = data_cfg["split"]
    train_ds = ds.sel(time=slice(split["train_start"], split["train_end"]))
    stats = compute_variable_stats(train_ds, data_cfg["variables"])
    save_stats(stats, data_cfg["norm_stats_path"])

    sk_cfg = data_cfg["skeleton"]
    tracks = build_skeleton_tracks(
        ds=ds,
        cfg=SkeletonConfig(
            objects_per_step=int(data_cfg["objects_per_step"]),
            min_distance_deg=float(sk_cfg["min_distance_deg"]),
            vorticity_threshold=float(sk_cfg["vorticity_threshold"]),
            pressure_threshold=float(sk_cfg["pressure_threshold"]),
            max_link_distance_deg=float(sk_cfg["max_link_distance_deg"]),
        ),
    )
    save_tracks(tracks, data_cfg["track_cache_path"])
    print("Prepared normalization stats:", data_cfg["norm_stats_path"])
    print("Prepared skeleton tracks:", data_cfg["track_cache_path"])


def _resolve_device(cfg: dict) -> torch.device:
    requested = cfg["experiment"].get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _configure_runtime(cfg: dict, device: torch.device, logger=None) -> None:
    perf_cfg = cfg.get("experiment", {}).get("performance", {})
    allow_tf32 = bool(perf_cfg.get("allow_tf32", True))
    cudnn_benchmark = bool(perf_cfg.get("cudnn_benchmark", True))
    matmul_precision = str(perf_cfg.get("matmul_precision", "high"))

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception:
            if logger is not None:
                logger.warning("Failed to set matmul precision: %s", matmul_precision)

    if device.type == "cuda":
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.benchmark = cudnn_benchmark

    if logger is not None:
        logger.info(
            "Runtime performance: matmul_precision=%s, allow_tf32=%s, cudnn_benchmark=%s",
            matmul_precision,
            allow_tf32,
            cudnn_benchmark,
        )


def _build_model(
    cfg: dict,
    dataloaders: dict,
    device: torch.device,
    logger=None,
    compile_for_train: bool = False,
) -> LTGNet:
    sample = next(iter(dataloaders["train"]))
    in_channels = int(sample["x_hist"].shape[2])
    model = LTGNet(cfg=cfg, in_channels=in_channels).to(device)
    if compile_for_train:
        comp_cfg = cfg.get("experiment", {}).get("compile", {})
        if bool(comp_cfg.get("enabled", False)):
            strict = bool(comp_cfg.get("strict", False))
            if not hasattr(torch, "compile"):
                if strict:
                    raise RuntimeError("torch.compile is not available in current PyTorch version.")
                if logger is not None:
                    logger.warning("torch.compile unavailable, fallback to eager model.")
            else:
                kwargs = {}
                for key in ["backend", "mode", "fullgraph", "dynamic"]:
                    if key in comp_cfg:
                        kwargs[key] = comp_cfg[key]
                try:
                    model = torch.compile(model, **kwargs)
                    if logger is not None:
                        logger.info("torch.compile enabled: %s", kwargs)
                except Exception as exc:
                    if strict:
                        raise RuntimeError(f"torch.compile failed: {exc}") from exc
                    if logger is not None:
                        logger.warning("torch.compile failed, fallback to eager model: %s", exc)
    return model


def _apply_trainable_policy(model: LTGNet, cfg: dict, logger=None) -> None:
    trainable_cfg = cfg.get("experiment", {}).get("trainable", {})
    if not bool(trainable_cfg.get("enabled", False)):
        return

    include_patterns = list(trainable_cfg.get("include_patterns", ["traj_predictor"]))
    exclude_patterns = list(trainable_cfg.get("exclude_patterns", []))

    for _, p in model.named_parameters():
        p.requires_grad = False

    def _match(name: str, patterns: list[str]) -> bool:
        return any((pat in name) for pat in patterns)

    for name, p in model.named_parameters():
        if _match(name, include_patterns):
            p.requires_grad = True
        if exclude_patterns and _match(name, exclude_patterns):
            p.requires_grad = False

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params <= 0:
        raise RuntimeError(
            "Trainable policy enabled but no parameters left trainable. "
            f"include_patterns={include_patterns}, exclude_patterns={exclude_patterns}"
        )
    if logger is not None:
        logger.info(
            "Trainable policy enabled: trainable=%d / total=%d (%.2f%%), include=%s, exclude=%s",
            trainable_params,
            total_params,
            100.0 * trainable_params / max(1, total_params),
            include_patterns,
            exclude_patterns,
        )
        logger.info("Trainable parameter groups (first 20): %s", trainable_names[:20])
        if len(trainable_names) > 20:
            logger.info("... and %d more trainable tensors.", len(trainable_names) - 20)


def _load_checkpoint(path: str, device: torch.device):
    load_kwargs = {"map_location": device}
    try:
        return torch.load(path, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)
    except Exception:
        # Fallback for legacy checkpoints that require full pickle deserialization.
        return torch.load(path, weights_only=False, **load_kwargs)


def _to_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
        return ckpt_obj["model"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise RuntimeError("Unsupported checkpoint format.")


def _compose_fused_state_dict(
    model: LTGNet,
    field_state: dict[str, torch.Tensor],
    track_state: dict[str, torch.Tensor],
    fused_encoder_source: str = "field",
) -> dict[str, torch.Tensor]:
    fused = dict(model.state_dict())
    for k in list(fused.keys()):
        if k in field_state:
            fused[k] = field_state[k]
    if fused_encoder_source == "track":
        for k in list(fused.keys()):
            if k.startswith(ENCODER_PREFIX) and k in track_state:
                fused[k] = track_state[k]
    for k in list(fused.keys()):
        if k.startswith(TRAJ_PREFIXES) and k in track_state:
            fused[k] = track_state[k]
    return fused


def train(config: str) -> None:
    cfg = load_config(config)
    set_seed(int(cfg["experiment"]["seed"]))
    output_dir = Path(cfg["experiment"]["output_dir"])
    logger = setup_logger(cfg["experiment"]["name"], output_dir)
    device = _resolve_device(cfg)
    _configure_runtime(cfg, device, logger)

    dataloaders = build_dataloaders(cfg)
    model = _build_model(cfg, dataloaders, device=device, logger=logger, compile_for_train=True)
    _apply_trainable_policy(model, cfg, logger=logger)
    loss_fn = CompositeLoss(cfg).to(device)
    trainer = LTGTrainer(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        dataloaders=dataloaders,
        output_dir=output_dir,
        logger=logger,
        device=device,
    )
    resume_path = cfg["experiment"].get("resume_checkpoint")
    if resume_path:
        resume_reset_best = bool(cfg["experiment"].get("resume_reset_best", False))
        trainer.load_checkpoint(resume_path, reset_best=resume_reset_best)
    trainer.fit()
    logger.info("Training completed.")


def evaluate(
    config: str,
    checkpoint: str,
    split: str = "test",
    inference_mode: str = "single",
    field_checkpoint: str = "",
    track_checkpoint: str = "",
    fused_encoder_source: str = "field",
    save_fused_checkpoint: str = "",
) -> None:
    cfg = load_config(config)
    device = _resolve_device(cfg)
    output_dir = Path(cfg["experiment"]["output_dir"])
    logger = setup_logger(cfg["experiment"]["name"] + "_eval", output_dir)
    _configure_runtime(cfg, device, logger)

    dataloaders = build_dataloaders(cfg)
    model = _build_model(cfg, dataloaders, device=device, logger=logger, compile_for_train=False)
    mode = str(inference_mode)
    if mode == "single":
        if not checkpoint:
            raise ValueError("single mode requires --checkpoint.")
        state_dict = _to_state_dict(_load_checkpoint(checkpoint, device))
    elif mode == "field":
        field_path = field_checkpoint or checkpoint
        if not field_path:
            raise ValueError("field mode requires --field_checkpoint or --checkpoint.")
        state_dict = _to_state_dict(_load_checkpoint(field_path, device))
    elif mode == "track":
        track_path = track_checkpoint or checkpoint
        if not track_path:
            raise ValueError("track mode requires --track_checkpoint or --checkpoint.")
        state_dict = _to_state_dict(_load_checkpoint(track_path, device))
    elif mode == "fused":
        if not field_checkpoint or not track_checkpoint:
            raise ValueError("fused mode requires both --field_checkpoint and --track_checkpoint.")
        field_state = _to_state_dict(_load_checkpoint(field_checkpoint, device))
        track_state = _to_state_dict(_load_checkpoint(track_checkpoint, device))
        state_dict = _compose_fused_state_dict(
            model=model,
            field_state=field_state,
            track_state=track_state,
            fused_encoder_source=str(fused_encoder_source),
        )
        if save_fused_checkpoint:
            fused_path = Path(save_fused_checkpoint)
            fused_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": state_dict,
                    "meta": {
                        "inference_mode": "fused",
                        "field_checkpoint": field_checkpoint,
                        "track_checkpoint": track_checkpoint,
                        "fused_encoder_source": fused_encoder_source,
                    },
                },
                fused_path,
            )
            logger.info("Saved fused checkpoint: %s", fused_path)
    else:
        raise ValueError(f"Unsupported inference_mode: {mode}")

    model.load_state_dict(state_dict)

    evaluator = LTGEvaluator(
        model=model,
        dataloader=dataloaders[split],
        device=device,
        max_wavenumber=int(cfg["loss"]["spectral_wavenumbers"]),
    )
    metrics = evaluator.evaluate()
    logger.info("Evaluation metrics (%s): %s", split, metrics)
    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="LTG-Net command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prepare = subparsers.add_parser("prepare", help="prepare normalization and skeleton tracks")
    p_prepare.add_argument("--config", required=True)

    p_train = subparsers.add_parser("train", help="train LTG-Net")
    p_train.add_argument("--config", required=True)

    p_eval = subparsers.add_parser("evaluate", help="evaluate LTG-Net")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--checkpoint", default="")
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.add_argument(
        "--inference_mode",
        default="single",
        choices=["single", "field", "track", "fused"],
    )
    p_eval.add_argument("--field_checkpoint", default="")
    p_eval.add_argument("--track_checkpoint", default="")
    p_eval.add_argument("--fused_encoder_source", default="field", choices=["field", "track"])
    p_eval.add_argument("--save_fused_checkpoint", default="")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "train":
        train(args.config)
    elif args.command == "evaluate":
        evaluate(
            args.config,
            args.checkpoint,
            args.split,
            inference_mode=args.inference_mode,
            field_checkpoint=args.field_checkpoint,
            track_checkpoint=args.track_checkpoint,
            fused_encoder_source=args.fused_encoder_source,
            save_fused_checkpoint=args.save_fused_checkpoint,
        )
    else:
        raise ValueError("Unsupported command: %s" % args.command)


if __name__ == "__main__":
    main()
