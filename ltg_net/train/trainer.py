from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from ltg_net.train.loops import average_meter, move_batch_to_device


def build_optimizer(cfg: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg["optimization"]["optimizer"]
    name = opt_cfg["name"].lower()
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg["weight_decay"])
    betas = tuple(float(x) for x in opt_cfg.get("betas", [0.9, 0.999]))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(cfg: dict[str, Any], optimizer: torch.optim.Optimizer):
    sch_cfg = cfg["optimization"]["scheduler"]
    total_epochs = int(cfg["optimization"]["epochs"])
    warmup_epochs = int(sch_cfg.get("warmup_epochs", 0))
    min_lr = float(sch_cfg.get("min_lr", 1e-6))
    base_lr = float(cfg["optimization"]["optimizer"]["lr"])

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max(1e-8, float(epoch + 1) / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        target = min_lr / base_lr
        return target + (1.0 - target) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


class LTGTrainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        dataloaders: dict[str, torch.utils.data.DataLoader],
        output_dir: Path,
        logger,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.dataloaders = dataloaders
        self.output_dir = output_dir
        self.logger = logger
        self.device = device
        self.optimizer = build_optimizer(cfg, model)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        precision = cfg["experiment"].get("precision", "fp32")
        self.use_amp = precision in {"fp16", "bf16"} and device.type == "cuda"
        if precision == "fp16":
            self.amp_dtype = torch.float16
        elif precision == "bf16":
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = None
        scaler_enabled = self.use_amp and precision == "fp16"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.grad_clip_norm = float(cfg["optimization"]["grad_clip_norm"])
        self.max_train_batches = int(cfg["optimization"].get("max_train_batches", 0))
        self.max_val_batches = int(cfg["optimization"].get("max_val_batches", 0))
        self.fail_on_nan = bool(cfg["optimization"].get("fail_on_nan", True))
        eval_cfg = cfg.get("evaluation", {})
        loss_mode = str(cfg.get("loss", {}).get("combine_mode", "fixed_weighted"))
        default_metric = "loss_total_raw" if loss_mode == "normalized_weighted" else "loss_total"
        self.checkpoint_metric = str(eval_cfg.get("checkpoint_metric", default_metric))
        default_mode = "min" if "loss" in self.checkpoint_metric.lower() else "max"
        checkpoint_mode = str(eval_cfg.get("checkpoint_mode", default_mode)).lower()
        self.checkpoint_mode = checkpoint_mode if checkpoint_mode in {"min", "max"} else default_mode
        self.checkpoint_min_delta = float(eval_cfg.get("checkpoint_min_delta", 0.0))

        self.validate_every = int(eval_cfg.get("validate_every", 1))
        self.save_every = int(eval_cfg.get("save_every", 1))
        self.save_history = bool(eval_cfg.get("save_history", True))
        self.history_path = self.output_dir / str(eval_cfg.get("history_file", "metrics_history.jsonl"))

        self.test_every = int(eval_cfg.get("test_every", 0))
        self.test_max_batches = int(eval_cfg.get("test_max_batches", 0))
        self.test_metrics_path = self.output_dir / str(eval_cfg.get("test_metrics_file", "test_metrics.jsonl"))

        early_cfg = eval_cfg.get("early_stopping", {})
        self.early_enabled = bool(early_cfg.get("enabled", False))
        default_early_mode = "min" if "loss" in self.checkpoint_metric.lower() else "max"
        early_mode = str(early_cfg.get("mode", default_early_mode)).lower()
        self.early_mode = early_mode if early_mode in {"min", "max"} else default_early_mode
        self.early_monitor = str(early_cfg.get("monitor", self.checkpoint_metric))
        self.early_patience = int(early_cfg.get("patience", 10))
        self.early_min_delta = float(early_cfg.get("min_delta", 0.0))
        self.early_warmup_epochs = int(early_cfg.get("warmup_epochs", 0))
        self.no_improve_epochs = 0

        dual_cfg = eval_cfg.get("dual_best", {})
        self.dual_best_enabled = bool(dual_cfg.get("enabled", False))
        self.dual_field_monitor = str(dual_cfg.get("field_monitor", "loss_field"))
        field_mode = str(dual_cfg.get("field_mode", "min")).lower()
        self.dual_field_mode = field_mode if field_mode in {"min", "max"} else "min"
        self.dual_field_min_delta = float(dual_cfg.get("field_min_delta", 0.0))
        self.dual_field_filename = str(dual_cfg.get("field_filename", "best_field.pt"))

        self.dual_track_monitor = str(dual_cfg.get("track_monitor", "loss_traj"))
        track_mode = str(dual_cfg.get("track_mode", "min")).lower()
        self.dual_track_mode = track_mode if track_mode in {"min", "max"} else "min"
        self.dual_track_min_delta = float(dual_cfg.get("track_min_delta", 0.0))
        self.dual_track_filename = str(dual_cfg.get("track_filename", "best_track.pt"))

        self.start_epoch = 0
        self.best_checkpoint_score = float("inf") if self.checkpoint_mode == "min" else float("-inf")
        self.best_early_score = float("inf") if self.early_mode == "min" else float("-inf")
        self.best_field_score = float("inf") if self.dual_field_mode == "min" else float("-inf")
        self.best_track_score = float("inf") if self.dual_track_mode == "min" else float("-inf")
        # Keep legacy field for backward compatibility with old checkpoints.
        self.best_val_loss = float("inf")

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
        resume_path = self.cfg.get("experiment", {}).get("resume_checkpoint")
        if self.start_epoch == 0 and not resume_path:
            for p in [self.history_path, self.test_metrics_path]:
                if p.exists():
                    p.unlink()

    def _autocast_context(self):
        if self.use_amp:
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast("cuda", enabled=True, dtype=self.amp_dtype)
            return torch.cuda.amp.autocast(enabled=True, dtype=self.amp_dtype)
        return nullcontext()

    def _save_checkpoint(self, epoch: int, is_best: bool = False, extra_best_files: list[str] | None = None) -> None:
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_checkpoint_score": self.best_checkpoint_score,
            "best_field_score": self.best_field_score,
            "best_track_score": self.best_track_score,
            "checkpoint_metric": self.checkpoint_metric,
            "checkpoint_mode": self.checkpoint_mode,
            "config": self.cfg,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, self.output_dir / f"epoch_{epoch:03d}.pt")
        best_files: list[str] = []
        if is_best:
            best_files.append("best.pt")
        if extra_best_files:
            best_files.extend(extra_best_files)
        for filename in sorted(set(best_files)):
            torch.save(ckpt, self.output_dir / filename)

    def load_checkpoint(self, path: str | Path, reset_best: bool = False) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = int(ckpt["epoch"]) + 1
        legacy_best = float(ckpt.get("best_val_loss", float("inf")))
        self.best_val_loss = legacy_best
        if "best_checkpoint_score" in ckpt:
            self.best_checkpoint_score = float(ckpt["best_checkpoint_score"])
        elif self.checkpoint_mode == "min":
            self.best_checkpoint_score = legacy_best
        if "best_field_score" in ckpt:
            self.best_field_score = float(ckpt["best_field_score"])
        if "best_track_score" in ckpt:
            self.best_track_score = float(ckpt["best_track_score"])

        ckpt_metric = str(ckpt.get("checkpoint_metric", ""))
        if (not reset_best) and ckpt_metric and ckpt_metric != self.checkpoint_metric:
            self.logger.info(
                "Checkpoint metric changed from %s -> %s, reset best-score tracking.",
                ckpt_metric,
                self.checkpoint_metric,
            )
            reset_best = True

        if reset_best:
            self.best_checkpoint_score = float("inf") if self.checkpoint_mode == "min" else float("-inf")
            self.best_early_score = float("inf") if self.early_mode == "min" else float("-inf")
            self.best_field_score = float("inf") if self.dual_field_mode == "min" else float("-inf")
            self.best_track_score = float("inf") if self.dual_track_mode == "min" else float("-inf")
            self.best_val_loss = float("inf")
            self.no_improve_epochs = 0
            self.logger.info("Best-score states reset after resume.")
        self.logger.info("Resumed from checkpoint: %s", path)

    def _run_train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.loss_fn.train()
        details = []
        pbar = tqdm(self.dataloaders["train"], desc=f"train {epoch:03d}", leave=False)
        for step_idx, batch in enumerate(pbar):
            if self.max_train_batches > 0 and step_idx >= self.max_train_batches:
                break
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                outputs = self.model(batch)
                loss, d = self.loss_fn(outputs, batch)
            if self.fail_on_nan and not torch.isfinite(loss):
                bad = {k: float(v.detach().cpu().item()) for k, v in d.items() if isinstance(v, torch.Tensor)}
                raise RuntimeError(f"Non-finite loss detected at epoch={epoch}, step={step_idx}: {bad}")

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            d["lr"] = torch.tensor(self.optimizer.param_groups[0]["lr"], device=self.device)
            details.append(d)
            pbar.set_postfix(loss=float(loss.item()))
        return average_meter(details)

    @torch.no_grad()
    def _run_eval_epoch(self, split: str, epoch: int) -> dict[str, float]:
        self.model.eval()
        self.loss_fn.eval()
        details = []
        pbar = tqdm(self.dataloaders[split], desc=f"{split} {epoch:03d}", leave=False)
        max_batches = self.max_val_batches if split != "train" else self.max_train_batches
        for step_idx, batch in enumerate(pbar):
            if max_batches > 0 and step_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, self.device)
            with self._autocast_context():
                outputs = self.model(batch)
                _, d = self.loss_fn(outputs, batch)
            if self.fail_on_nan:
                for key, value in d.items():
                    if isinstance(value, torch.Tensor) and (not torch.isfinite(value).all()):
                        raise RuntimeError(
                            f"Non-finite metric detected during {split} epoch={epoch}, step={step_idx}, key={key}"
                        )
            details.append(d)
        return average_meter(details)

    @torch.no_grad()
    def _run_test_metrics(self) -> dict[str, float]:
        if "test" not in self.dataloaders:
            return {}
        from ltg_net.eval.evaluator import LTGEvaluator

        evaluator = LTGEvaluator(
            model=self.model,
            dataloader=self.dataloaders["test"],
            device=self.device,
            max_wavenumber=int(self.cfg["loss"]["spectral_wavenumbers"]),
            max_batches=self.test_max_batches,
        )
        return evaluator.evaluate()

    def fit(self) -> None:
        total_epochs = int(self.cfg["optimization"]["epochs"])
        self._prepare_history_files()

        for epoch in range(self.start_epoch, total_epochs):
            if hasattr(self.loss_fn, "set_epoch"):
                self.loss_fn.set_epoch(epoch)
            train_stats = self._run_train_epoch(epoch)
            self.scheduler.step()
            self.logger.info("Epoch %03d train: %s", epoch, train_stats)

            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "train": train_stats,
            }

            if (epoch + 1) % self.validate_every == 0:
                val_stats = self._run_eval_epoch("val", epoch)
                self.logger.info("Epoch %03d val: %s", epoch, val_stats)
                epoch_record["val"] = val_stats

                metric_name = self.checkpoint_metric
                if metric_name not in val_stats:
                    metric_name = "loss_total"
                val_metric = float(val_stats[metric_name])
                is_best = self._is_better(
                    current=val_metric,
                    best=self.best_checkpoint_score,
                    mode=self.checkpoint_mode,
                    min_delta=self.checkpoint_min_delta,
                )
                if is_best:
                    self.best_checkpoint_score = val_metric
                    if self.checkpoint_mode == "min":
                        self.best_val_loss = val_metric
                    self.logger.info(
                        "New best val metric (%s, mode=%s): %.6f",
                        metric_name,
                        self.checkpoint_mode,
                        self.best_checkpoint_score,
                    )

                dual_best_files: list[str] = []
                if self.dual_best_enabled:
                    field_metric_name = self.dual_field_monitor if self.dual_field_monitor in val_stats else (
                        "loss_field" if "loss_field" in val_stats else metric_name
                    )
                    field_value = float(val_stats[field_metric_name])
                    if self._is_better(
                        current=field_value,
                        best=self.best_field_score,
                        mode=self.dual_field_mode,
                        min_delta=self.dual_field_min_delta,
                    ):
                        self.best_field_score = field_value
                        dual_best_files.append(self.dual_field_filename)
                        self.logger.info(
                            "New best field-priority metric (%s, mode=%s): %.6f -> %s",
                            field_metric_name,
                            self.dual_field_mode,
                            self.best_field_score,
                            self.dual_field_filename,
                        )

                    track_metric_name = self.dual_track_monitor if self.dual_track_monitor in val_stats else (
                        "loss_traj" if "loss_traj" in val_stats else metric_name
                    )
                    track_value = float(val_stats[track_metric_name])
                    if self._is_better(
                        current=track_value,
                        best=self.best_track_score,
                        mode=self.dual_track_mode,
                        min_delta=self.dual_track_min_delta,
                    ):
                        self.best_track_score = track_value
                        dual_best_files.append(self.dual_track_filename)
                        self.logger.info(
                            "New best track-priority metric (%s, mode=%s): %.6f -> %s",
                            track_metric_name,
                            self.dual_track_mode,
                            self.best_track_score,
                            self.dual_track_filename,
                        )

                if (epoch + 1) % self.save_every == 0 or is_best or len(dual_best_files) > 0:
                    self._save_checkpoint(epoch, is_best=is_best, extra_best_files=dual_best_files)

                test_metrics = {}
                if self.test_every > 0 and (epoch + 1) % self.test_every == 0:
                    test_metrics = self._run_test_metrics()
                    self.logger.info("Epoch %03d test metrics: %s", epoch, test_metrics)
                    epoch_record["test"] = test_metrics
                    self._append_jsonl(
                        self.test_metrics_path,
                        {
                            "epoch": epoch,
                            "checkpoint_metric": metric_name,
                            "checkpoint_value": val_metric,
                            "test": test_metrics,
                        },
                    )

                early_metric_name = self.early_monitor
                if early_metric_name not in val_stats:
                    early_metric_name = metric_name
                early_value = float(val_stats[early_metric_name])
                if self._is_better(
                    current=early_value,
                    best=self.best_early_score,
                    mode=self.early_mode,
                    min_delta=self.early_min_delta,
                ):
                    self.best_early_score = early_value
                    self.no_improve_epochs = 0
                elif self.early_enabled and (epoch + 1) >= self.early_warmup_epochs:
                    self.no_improve_epochs += 1
                    self.logger.info(
                        "EarlyStop monitor=%s mode=%s no_improve=%d/%d current=%.6f best=%.6f",
                        early_metric_name,
                        self.early_mode,
                        self.no_improve_epochs,
                        self.early_patience,
                        early_value,
                        self.best_early_score,
                    )

                if self.save_history:
                    self._append_jsonl(self.history_path, epoch_record)

                if self.early_enabled and self.no_improve_epochs >= self.early_patience:
                    self.logger.info(
                        "Early stopping triggered at epoch %03d (monitor=%s, best=%.6f).",
                        epoch,
                        early_metric_name,
                        self.best_early_score,
                    )
                    break
            elif self.save_history:
                self._append_jsonl(self.history_path, epoch_record)

    @torch.no_grad()
    def evaluate_test(self) -> dict[str, float]:
        return self._run_eval_epoch("test", epoch=-1)
