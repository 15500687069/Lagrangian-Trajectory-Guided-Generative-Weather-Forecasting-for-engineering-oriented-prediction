from __future__ import annotations

import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from miniltgnet.inference import apply_inference_postprocess
from miniltgnet.metrics import acc, extreme_f1, rmse, spectral_distance, track_mae


def setup_logger(name: str, output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _average_meter(rows: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    meter: dict[str, list[float]] = defaultdict(list)
    for item in rows:
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                meter[k].append(float(v.detach().cpu().item()))
    return {k: sum(vs) / max(1, len(vs)) for k, vs in meter.items()}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module | None,
    device: torch.device,
    max_batches: int = 0,
    spectral_wavenumbers: int = 32,
    inference_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    if loss_fn is not None:
        loss_fn.eval()
    metrics: dict[str, list[float]] = defaultdict(list)
    for step_idx, batch in enumerate(tqdm(dataloader, desc="evaluate", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        pred_field_raw = outputs["field_pred"]
        pred_traj_raw = outputs["traj_pred"]
        pred_field, pred_traj = apply_inference_postprocess(
            pred_field=pred_field_raw,
            pred_traj=pred_traj_raw,
            batch=batch,
            cfg=inference_config,
        )
        tgt_field = batch["y_future"].float()
        tgt_traj = batch["traj_future"].float()

        if loss_fn is not None:
            _, details = loss_fn(outputs, batch)
            for k, v in details.items():
                metrics[k].append(float(v.detach().cpu().item()))

        metrics["rmse"].append(float(rmse(pred_field, tgt_field).item()))
        metrics["acc"].append(float(acc(pred_field, tgt_field).item()))
        metrics["track_mae"].append(float(track_mae(pred_traj, tgt_traj).item()))
        metrics["extreme_f1"].append(float(extreme_f1(pred_field, tgt_field).item()))

        # Keep spectral metric consistent with standardized alignment:
        # evaluate across the full horizon by flattening [B,T,C,H,W] -> [B*T,C,H,W].
        if pred_field.ndim == 5:
            b, t, c, h, w = pred_field.shape
            pred_field_spec = pred_field.reshape(b * t, c, h, w)
            tgt_field_spec = tgt_field.reshape(b * t, c, h, w)
        else:
            pred_field_spec = pred_field
            tgt_field_spec = tgt_field
        metrics["spectral_distance"].append(
            float(spectral_distance(pred_field_spec, tgt_field_spec, max_wavenumber=spectral_wavenumbers).item())
        )
    return {k: sum(v) / max(1, len(v)) for k, v in metrics.items()}


class MiniTrainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        dataloaders: dict[str, torch.utils.data.DataLoader],
        output_dir: Path,
        logger: logging.Logger,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.dataloaders = dataloaders
        self.output_dir = output_dir
        self.logger = logger
        self.device = device

        ocfg = cfg["optimization"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(ocfg["lr"]),
            weight_decay=float(ocfg.get("weight_decay", 0.0)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(ocfg["epochs"])),
            eta_min=float(ocfg.get("min_lr", 1e-6)),
        )
        self.grad_clip_norm = float(ocfg.get("grad_clip_norm", 1.0))
        self.grad_accum_steps = max(1, int(ocfg.get("grad_accum_steps", 1)))
        self.max_train_batches = int(ocfg.get("max_train_batches", 0))
        self.max_val_batches = int(ocfg.get("max_val_batches", 0))
        self.fail_on_nan = bool(ocfg.get("fail_on_nan", True))

        ecfg = cfg["evaluation"]
        self.validate_every = int(ecfg.get("validate_every", 1))
        self.save_every = int(ecfg.get("save_every", 1))
        self.test_every = int(ecfg.get("test_every", 0))
        self.test_max_batches = int(ecfg.get("test_max_batches", 0))
        self.spectral_wavenumbers = int(ecfg.get("spectral_wavenumbers", 32))

        self.checkpoint_metric = str(ecfg.get("checkpoint_metric", "loss_total"))
        self.checkpoint_mode = str(ecfg.get("checkpoint_mode", "min")).lower()
        if self.checkpoint_mode not in {"min", "max"}:
            self.checkpoint_mode = "min"
        self.checkpoint_min_delta = float(ecfg.get("checkpoint_min_delta", 0.0))

        early = ecfg.get("early_stopping", {})
        self.early_enabled = bool(early.get("enabled", True))
        self.early_monitor = str(early.get("monitor", self.checkpoint_metric))
        self.early_mode = str(early.get("mode", self.checkpoint_mode)).lower()
        if self.early_mode not in {"min", "max"}:
            self.early_mode = self.checkpoint_mode
        self.early_patience = int(early.get("patience", 3))
        self.early_min_delta = float(early.get("min_delta", 0.0))
        self.early_warmup_epochs = int(early.get("warmup_epochs", 0))
        self.no_improve_epochs = 0

        self.save_history = bool(ecfg.get("save_history", True))
        self.history_path = self.output_dir / str(ecfg.get("history_file", "metrics_history.jsonl"))
        self.test_metrics_path = self.output_dir / str(ecfg.get("test_metrics_file", "test_metrics.jsonl"))
        self.use_ema_for_eval = bool(ecfg.get("use_ema_for_eval", True))

        self.start_epoch = 0
        self.best_score = float("inf") if self.checkpoint_mode == "min" else float("-inf")
        self.best_early_score = float("inf") if self.early_mode == "min" else float("-inf")

        ema_cfg = cfg.get("ema", {})
        self.ema_enabled = bool(ema_cfg.get("enabled", True))
        self.ema_decay = float(ema_cfg.get("decay", 0.995))
        self.ema_start_step = int(ema_cfg.get("start_step", 0))
        self.global_step = 0
        self.ema_state: dict[str, torch.Tensor] = {}
        if self.ema_enabled:
            self._init_ema_state()

    @staticmethod
    def _is_better(current: float, best: float, mode: str, min_delta: float) -> bool:
        if mode == "max":
            return current > (best + min_delta)
        return current < (best - min_delta)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _prepare_history_files(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.start_epoch == 0 and not self.cfg["experiment"].get("resume_checkpoint"):
            for p in [self.history_path, self.test_metrics_path]:
                if p.exists():
                    p.unlink()

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        model_ema = self._build_ema_model_state_dict() if self.ema_enabled else None
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "model_ema": model_ema,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_score": self.best_score,
            "global_step": self.global_step,
            "ema_state": self.ema_state if self.ema_enabled else None,
            "config": self.cfg,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, self.output_dir / f"epoch_{epoch:03d}.pt")
        if is_best:
            torch.save(ckpt, self.output_dir / "best.pt")

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        strict_resume = bool(self.cfg.get("experiment", {}).get("resume_strict", False))
        state_dict = ckpt["model"]
        if strict_resume:
            self.model.load_state_dict(state_dict, strict=True)
        else:
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                self.logger.warning(
                    "Resume checkpoint has architecture mismatch; continuing with non-strict load. "
                    "missing=%d unexpected=%d err=%s",
                    len(missing),
                    len(unexpected),
                    exc,
                )
                if missing:
                    self.logger.warning("Missing keys (first 20): %s", missing[:20])
                if unexpected:
                    self.logger.warning("Unexpected keys (first 20): %s", unexpected[:20])

        resume_optimizer = bool(self.cfg.get("experiment", {}).get("resume_optimizer", False))
        resume_scheduler = bool(self.cfg.get("experiment", {}).get("resume_scheduler", False))
        if resume_optimizer:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Optimizer state skipped due to mismatch: %s", exc)
        else:
            self.logger.info("Resume optimizer disabled; using fresh optimizer state from current config.")
        if resume_scheduler:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Scheduler state skipped due to mismatch: %s", exc)
        else:
            self.logger.info("Resume scheduler disabled; using fresh scheduler state from current config.")
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.best_score = float(ckpt.get("best_score", self.best_score))
        if bool(self.cfg.get("experiment", {}).get("resume_reset_best", False)):
            self.best_score = float("inf") if self.checkpoint_mode == "min" else float("-inf")
            self.logger.info("Resume with reset best score enabled; best_score reset for current run.")
        self.global_step = int(ckpt.get("global_step", 0))
        if self.ema_enabled and ckpt.get("ema_state") is not None:
            self.ema_state = {k: v.to(self.device) for k, v in ckpt["ema_state"].items()}

        # If resuming into a fresh output directory, ensure best.pt exists for downstream evaluate commands.
        best_path = self.output_dir / "best.pt"
        if not best_path.exists():
            best_bootstrap = {
                "epoch": int(ckpt.get("epoch", self.start_epoch - 1)),
                "model": ckpt.get("model", self.model.state_dict()),
                "model_ema": ckpt.get("model_ema", self._build_ema_model_state_dict() if self.ema_enabled else None),
                "optimizer": ckpt.get("optimizer", self.optimizer.state_dict()),
                "scheduler": ckpt.get("scheduler", self.scheduler.state_dict()),
                "best_score": self.best_score,
                "global_step": self.global_step,
                "ema_state": ckpt.get("ema_state", self.ema_state if self.ema_enabled else None),
                "config": self.cfg,
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_bootstrap, best_path)
            self.logger.info("Bootstrapped best checkpoint for resumed run: %s", best_path)
        self.logger.info("Resumed from checkpoint: %s", path)

    def _init_ema_state(self) -> None:
        self.ema_state = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.ema_state[name] = p.detach().clone()

    def _update_ema(self) -> None:
        if not self.ema_enabled:
            return
        if self.global_step < self.ema_start_step:
            return
        decay = self.ema_decay
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in self.ema_state:
                    self.ema_state[name] = p.detach().clone()
                else:
                    self.ema_state[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    def _build_ema_model_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.model.state_dict()
        if not self.ema_state:
            return state
        out: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k in self.ema_state:
                out[k] = self.ema_state[k].detach().clone()
            else:
                out[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
        return out

    @contextmanager
    def _ema_scope(self):
        if not (self.ema_enabled and self.use_ema_for_eval and self.ema_state):
            yield
            return
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad and name in self.ema_state:
                    backup[name] = p.detach().clone()
                    p.copy_(self.ema_state[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if p.requires_grad and name in backup:
                        p.copy_(backup[name])

    def _run_train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.loss_fn.train()
        details: list[dict[str, torch.Tensor]] = []
        pbar = tqdm(self.dataloaders["train"], desc=f"train {epoch:03d}", leave=False)
        self.optimizer.zero_grad(set_to_none=True)
        accum_counter = 0
        for step_idx, batch in enumerate(pbar):
            if self.max_train_batches > 0 and step_idx >= self.max_train_batches:
                break
            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(batch)
            loss, d = self.loss_fn(outputs, batch)
            if self.fail_on_nan and not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step_idx}")
            (loss / self.grad_accum_steps).backward()

            accum_counter += 1
            should_step = (accum_counter % self.grad_accum_steps == 0) or (
                self.max_train_batches > 0 and (step_idx + 1) >= self.max_train_batches
            )
            if should_step:
                clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self._update_ema()
            d["lr"] = torch.tensor(self.optimizer.param_groups[0]["lr"], device=self.device)
            details.append(d)
            pbar.set_postfix(loss=float(loss.item()))
        if accum_counter > 0 and (accum_counter % self.grad_accum_steps != 0):
            clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            self._update_ema()
        return _average_meter(details)

    @torch.no_grad()
    def _run_val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        self.loss_fn.eval()
        details: list[dict[str, torch.Tensor]] = []
        with self._ema_scope():
            pbar = tqdm(self.dataloaders["val"], desc=f"val {epoch:03d}", leave=False)
            for step_idx, batch in enumerate(pbar):
                if self.max_val_batches > 0 and step_idx >= self.max_val_batches:
                    break
                batch = move_batch_to_device(batch, self.device)
                outputs = self.model(batch)
                _, d = self.loss_fn(outputs, batch)
                details.append(d)
        return _average_meter(details)

    @torch.no_grad()
    def _run_test_metrics(self) -> dict[str, float]:
        with self._ema_scope():
            return evaluate_model(
                model=self.model,
                dataloader=self.dataloaders["test"],
                loss_fn=None,
                device=self.device,
                max_batches=self.test_max_batches,
                spectral_wavenumbers=self.spectral_wavenumbers,
                inference_config=self.cfg,
            )

    def fit(self) -> None:
        ocfg = self.cfg["optimization"]
        total_epochs_cfg = int(ocfg["epochs"])
        epochs_mode = str(ocfg.get("epochs_mode", "absolute")).lower()
        if epochs_mode not in {"absolute", "additional"}:
            epochs_mode = "absolute"
        if self.start_epoch > 0 and epochs_mode == "additional":
            total_epochs = self.start_epoch + total_epochs_cfg
        else:
            total_epochs = total_epochs_cfg
        self.logger.info(
            "Training schedule: start_epoch=%d, epochs_cfg=%d, mode=%s, total_epochs=%d",
            self.start_epoch,
            total_epochs_cfg,
            epochs_mode,
            total_epochs,
        )
        self._prepare_history_files()
        for epoch in range(self.start_epoch, total_epochs):
            if hasattr(self.loss_fn, "set_progress"):
                self.loss_fn.set_progress(epoch=epoch, total_epochs=total_epochs)
            train_stats = self._run_train_epoch(epoch)
            self.scheduler.step()
            self.logger.info("Epoch %03d train: %s", epoch, train_stats)
            record: dict[str, Any] = {"epoch": epoch, "train": train_stats}

            if (epoch + 1) % self.validate_every == 0:
                val_stats = self._run_val_epoch(epoch)
                self.logger.info("Epoch %03d val: %s", epoch, val_stats)
                record["val"] = val_stats

                metric_name = self.checkpoint_metric if self.checkpoint_metric in val_stats else "loss_total"
                current = float(val_stats[metric_name])
                is_best = self._is_better(
                    current=current,
                    best=self.best_score,
                    mode=self.checkpoint_mode,
                    min_delta=self.checkpoint_min_delta,
                )
                if is_best:
                    self.best_score = current
                    self.logger.info(
                        "New best val metric (%s, mode=%s): %.6f",
                        metric_name,
                        self.checkpoint_mode,
                        self.best_score,
                    )
                if (epoch + 1) % self.save_every == 0 or is_best:
                    self._save_checkpoint(epoch, is_best=is_best)

                if self.test_every > 0 and (epoch + 1) % self.test_every == 0:
                    tmetrics = self._run_test_metrics()
                    self.logger.info("Epoch %03d test metrics: %s", epoch, tmetrics)
                    record["test"] = tmetrics
                    self._append_jsonl(
                        self.test_metrics_path,
                        {
                            "epoch": epoch,
                            "checkpoint_metric": metric_name,
                            "checkpoint_value": current,
                            "test": tmetrics,
                        },
                    )

                early_name = self.early_monitor if self.early_monitor in val_stats else metric_name
                early_val = float(val_stats[early_name])
                if self._is_better(early_val, self.best_early_score, self.early_mode, self.early_min_delta):
                    self.best_early_score = early_val
                    self.no_improve_epochs = 0
                elif self.early_enabled and (epoch + 1) >= self.early_warmup_epochs:
                    self.no_improve_epochs += 1
                    self.logger.info(
                        "EarlyStop monitor=%s mode=%s no_improve=%d/%d current=%.6f best=%.6f",
                        early_name,
                        self.early_mode,
                        self.no_improve_epochs,
                        self.early_patience,
                        early_val,
                        self.best_early_score,
                    )

                if self.save_history:
                    self._append_jsonl(self.history_path, record)

                if self.early_enabled and self.no_improve_epochs >= self.early_patience:
                    self.logger.info(
                        "Early stopping triggered at epoch %03d (monitor=%s, best=%.6f).",
                        epoch,
                        early_name,
                        self.best_early_score,
                    )
                    break
            elif self.save_history:
                self._append_jsonl(self.history_path, record)
