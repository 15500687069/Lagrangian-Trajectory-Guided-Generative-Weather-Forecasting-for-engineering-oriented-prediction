from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniltgnet.config import load_config
from miniltgnet.data import build_dataloaders
from miniltgnet.inference import apply_inference_postprocess
from miniltgnet.metrics import acc, extreme_f1, rmse, spectral_distance, track_mae
from miniltgnet.model import build_model
from miniltgnet.trainer import move_batch_to_device


def _resolve_device(cfg: dict[str, Any]) -> torch.device:
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
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if deterministic and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_model(
    cfg: dict[str, Any],
    dataloaders: dict[str, torch.utils.data.DataLoader],
    checkpoint: str,
    device: torch.device,
) -> torch.nn.Module:
    sample = next(iter(dataloaders["train"]))
    in_channels = int(sample["x_hist"].shape[2])
    model = build_model(cfg, in_channels=in_channels).to(device)
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        use_ema = bool(cfg.get("evaluation", {}).get("use_ema_for_eval", True))
        if use_ema and "model_ema" in ckpt and ckpt["model_ema"] is not None:
            state = ckpt["model_ema"]
        else:
            state = ckpt["model"] if "model" in ckpt else ckpt
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _clamp_geo(state: torch.Tensor) -> torch.Tensor:
    lat = state[..., 0].clamp(-90.0, 90.0)
    lon = state[..., 1] % 360.0
    return torch.stack([lat, lon], dim=-1)


def _predict_persistence(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    horizon = int(batch["y_future"].shape[1])
    last_field = batch["x_hist"][:, -1].float()
    last_traj = batch["traj_hist"][:, -1].float()
    field_pred = last_field.unsqueeze(1).repeat(1, horizon, 1, 1, 1)
    traj_pred = last_traj.unsqueeze(1).repeat(1, horizon, 1, 1)
    return field_pred, traj_pred


def _predict_linear(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    horizon = int(batch["y_future"].shape[1])
    x_hist = batch["x_hist"].float()
    traj_hist = batch["traj_hist"].float()
    if x_hist.shape[1] < 2:
        return _predict_persistence(batch)

    prev_field = x_hist[:, -1]
    delta_field = x_hist[:, -1] - x_hist[:, -2]
    fields: list[torch.Tensor] = []
    for _ in range(horizon):
        prev_field = prev_field + delta_field
        fields.append(prev_field)
    field_pred = torch.stack(fields, dim=1)

    prev_traj = traj_hist[:, -1]
    delta_lat = traj_hist[:, -1, :, 0] - traj_hist[:, -2, :, 0]
    delta_lon = (traj_hist[:, -1, :, 1] - traj_hist[:, -2, :, 1] + 180.0) % 360.0 - 180.0
    trajs: list[torch.Tensor] = []
    for _ in range(horizon):
        lat = prev_traj[..., 0] + delta_lat
        lon = prev_traj[..., 1] + delta_lon
        prev_traj = _clamp_geo(torch.stack([lat, lon], dim=-1))
        trajs.append(prev_traj)
    traj_pred = torch.stack(trajs, dim=1)
    return field_pred, traj_pred


def _build_climatology(
    train_loader: torch.utils.data.DataLoader,
    max_batches: int = 0,
) -> torch.Tensor:
    count = 0
    sum_y: torch.Tensor | None = None
    for step_idx, batch in enumerate(train_loader):
        if max_batches > 0 and step_idx >= max_batches:
            break
        y = batch["y_future"].float()  # [B,T,C,H,W]
        batch_sum = y.sum(dim=0)  # [T,C,H,W]
        if sum_y is None:
            sum_y = batch_sum
        else:
            sum_y = sum_y + batch_sum
        count += int(y.shape[0])
    if sum_y is None or count == 0:
        raise RuntimeError("Failed to build climatology: empty training loader.")
    return (sum_y / float(count)).contiguous()


def _average(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


@torch.no_grad()
def _evaluate_predictor(
    method: str,
    predict_fn: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
    spectral_wavenumbers: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    overall: dict[str, list[float]] = defaultdict(list)
    lead_rows: list[dict[str, float]] = []
    lead_values: dict[int, dict[str, list[float]]] = {}

    for step_idx, batch in enumerate(tqdm(dataloader, desc=f"benchmark:{method}", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        pred_field, pred_traj = predict_fn(batch)
        target_field = batch["y_future"].float()
        target_traj = batch["traj_future"].float()
        horizon = int(target_field.shape[1])

        overall["rmse"].append(float(rmse(pred_field, target_field).item()))
        overall["acc"].append(float(acc(pred_field, target_field).item()))
        overall["track_mae"].append(float(track_mae(pred_traj, target_traj).item()))
        overall["extreme_f1"].append(float(extreme_f1(pred_field, target_field).item()))
        overall["spectral_distance"].append(
            float(spectral_distance(pred_field[:, 0], target_field[:, 0], max_wavenumber=spectral_wavenumbers).item())
        )

        for lead in range(horizon):
            if lead not in lead_values:
                lead_values[lead] = defaultdict(list)
            pf = pred_field[:, lead]
            tf = target_field[:, lead]
            pt = pred_traj[:, lead]
            tt = target_traj[:, lead]
            lead_values[lead]["rmse"].append(float(rmse(pf, tf).item()))
            lead_values[lead]["acc"].append(float(acc(pf, tf).item()))
            lead_values[lead]["track_mae"].append(float(track_mae(pt, tt).item()))
            lead_values[lead]["extreme_f1"].append(float(extreme_f1(pf, tf).item()))
            lead_values[lead]["spectral_distance"].append(
                float(spectral_distance(pf, tf, max_wavenumber=spectral_wavenumbers).item())
            )

    summary = {k: _average(v) for k, v in overall.items()}
    for lead in sorted(lead_values.keys()):
        row = {"method": method, "lead": int(lead + 1)}
        row.update({k: _average(v) for k, v in lead_values[lead].items()})
        lead_rows.append(row)
    return summary, lead_rows


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_report(summary_rows: list[dict[str, Any]]) -> str:
    if not summary_rows:
        return "# MiniLTG Validation Report\n\nNo rows found.\n"

    def best_min(metric: str) -> dict[str, Any]:
        return min(summary_rows, key=lambda x: float(x[metric]))

    def best_max(metric: str) -> dict[str, Any]:
        return max(summary_rows, key=lambda x: float(x[metric]))

    rmse_best = best_min("rmse")
    acc_best = best_max("acc")
    f1_best = best_max("extreme_f1")
    track_best = best_min("track_mae")

    lines = [
        "# MiniLTG Validation Report",
        "",
        "## Best Metrics",
        f"- Lowest RMSE: `{rmse_best['method']}` = {rmse_best['rmse']:.6f}",
        f"- Highest ACC: `{acc_best['method']}` = {acc_best['acc']:.6f}",
        f"- Highest extreme_F1: `{f1_best['method']}` = {f1_best['extreme_f1']:.6f}",
        f"- Lowest track_MAE: `{track_best['method']}` = {track_best['track_mae']:.6f}",
        "",
    ]

    by_method = {row["method"]: row for row in summary_rows}
    if "model" in by_method and "persistence" in by_method:
        m = by_method["model"]
        p = by_method["persistence"]
        gate = (
            (float(m["rmse"]) < float(p["rmse"]))
            and (float(m["acc"]) > float(p["acc"]))
            and (float(m["extreme_f1"]) >= float(p["extreme_f1"]))
        )
        lines.append("## Go/No-Go Gate (vs Persistence)")
        lines.append(
            f"- Gate rule: `model rmse < persistence rmse` and "
            f"`model acc > persistence acc` and `model extreme_f1 >= persistence extreme_f1`"
        )
        lines.append(f"- Gate result: **{'PASS' if gate else 'FAIL'}**")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLTG validation benchmark against lightweight baselines.")
    parser.add_argument("--config", required=True, help="MiniLTG config path.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path for model method.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["model", "persistence", "linear", "climatology"],
        choices=["model", "persistence", "linear", "climatology"],
        help="Methods to benchmark.",
    )
    parser.add_argument("--max_batches", type=int, default=0, help="Limit eval batches (0 means all).")
    parser.add_argument("--climatology_max_batches", type=int, default=0, help="Limit batches for climatology build.")
    parser.add_argument("--output_dir", default="outputs/miniltg/validation", help="Output directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _seed_everything(
        int(cfg["experiment"]["seed"]),
        deterministic=bool(cfg["experiment"].get("deterministic", False)),
    )
    device = _resolve_device(cfg)
    dataloaders = build_dataloaders(cfg)
    spectral_wavenumbers = int(cfg.get("evaluation", {}).get("spectral_wavenumbers", 32))

    methods = list(dict.fromkeys(args.methods))
    predictors: dict[str, Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]] = {}

    if "model" in methods:
        if not args.checkpoint:
            raise ValueError("`--checkpoint` is required when `model` method is enabled.")
        model = _load_model(cfg, dataloaders, args.checkpoint, device)

        @torch.no_grad()
        def _predict_model(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            out = model(batch)
            return apply_inference_postprocess(
                pred_field=out["field_pred"],
                pred_traj=out["traj_pred"],
                batch=batch,
                cfg=cfg,
            )

        predictors["model"] = _predict_model

    if "persistence" in methods:
        predictors["persistence"] = _predict_persistence

    if "linear" in methods:
        predictors["linear"] = _predict_linear

    if "climatology" in methods:
        clim = _build_climatology(dataloaders["train"], max_batches=int(args.climatology_max_batches))
        clim = clim.to(device)

        def _predict_climatology(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            b = int(batch["y_future"].shape[0])
            field_pred = clim.unsqueeze(0).expand(b, -1, -1, -1, -1).contiguous()
            traj_pred = batch["traj_hist"][:, -1].unsqueeze(1).repeat(1, int(clim.shape[0]), 1, 1)
            return field_pred, traj_pred

        predictors["climatology"] = _predict_climatology

    summary_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    for method in methods:
        summary, lead = _evaluate_predictor(
            method=method,
            predict_fn=predictors[method],
            dataloader=dataloaders[args.split],
            device=device,
            max_batches=int(args.max_batches),
            spectral_wavenumbers=spectral_wavenumbers,
        )
        row = {"method": method, "split": args.split}
        row.update(summary)
        summary_rows.append(row)
        lead_rows.extend(lead)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "benchmark_summary.csv"
    lead_csv = out_dir / "benchmark_leadtime.csv"
    summary_json = out_dir / "benchmark_summary.json"
    report_md = out_dir / "benchmark_report.md"

    _save_csv(summary_csv, summary_rows)
    _save_csv(lead_csv, lead_rows)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary_rows, "leadtime": lead_rows}, f, ensure_ascii=False, indent=2)
    report_md.write_text(_build_report(summary_rows), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved leadtime CSV: {lead_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved report: {report_md}")
    print("Summary:")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
