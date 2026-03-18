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


def _avg(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _metric_dict(
    pred_field: torch.Tensor,
    target_field: torch.Tensor,
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    spectral_wavenumbers: int,
) -> dict[str, float]:
    if pred_field.ndim == 5:
        b, t, c, h, w = pred_field.shape
        pred_field_eval = pred_field.reshape(b * t, c, h, w)
        target_field_eval = target_field.reshape(b * t, c, h, w)
    else:
        pred_field_eval = pred_field
        target_field_eval = target_field
    return {
        "rmse": float(rmse(pred_field_eval, target_field_eval).item()),
        "acc": float(acc(pred_field_eval, target_field_eval).item()),
        "track_mae": float(track_mae(pred_traj, target_traj).item()),
        "extreme_f1": float(extreme_f1(pred_field_eval, target_field_eval).item()),
        "spectral_distance": float(
            spectral_distance(pred_field_eval, target_field_eval, max_wavenumber=spectral_wavenumbers).item()
        ),
    }


@torch.no_grad()
def run_alignment(
    method: str,
    predict_fn: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]],
    dataloader: torch.utils.data.DataLoader,
    variables: list[str],
    leads: list[int],
    device: torch.device,
    spectral_wavenumbers: int,
    max_batches: int,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    overall: dict[str, list[float]] = defaultdict(list)
    lead_meter: dict[int, dict[str, list[float]]] = {}
    var_lead_meter: dict[tuple[int, str], dict[str, list[float]]] = {}

    for step_idx, batch in enumerate(tqdm(dataloader, desc=f"align:{method}", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        pred_field, pred_traj = predict_fn(batch)
        target_field = batch["y_future"].float()
        target_traj = batch["traj_future"].float()

        # Overall metrics on full horizon
        m = _metric_dict(
            pred_field=pred_field,
            target_field=target_field,
            pred_traj=pred_traj,
            target_traj=target_traj,
            spectral_wavenumbers=spectral_wavenumbers,
        )
        for k, v in m.items():
            overall[k].append(v)

        for lead in leads:
            li = lead - 1
            if li < 0 or li >= int(target_field.shape[1]):
                continue
            pf = pred_field[:, li]
            tf = target_field[:, li]
            pt = pred_traj[:, li]
            tt = target_traj[:, li]

            if lead not in lead_meter:
                lead_meter[lead] = defaultdict(list)
            lm = _metric_dict(
                pred_field=pf,
                target_field=tf,
                pred_traj=pt,
                target_traj=tt,
                spectral_wavenumbers=spectral_wavenumbers,
            )
            for k, v in lm.items():
                lead_meter[lead][k].append(v)

            for ci, var_name in enumerate(variables):
                key = (lead, var_name)
                if key not in var_lead_meter:
                    var_lead_meter[key] = defaultdict(list)
                pfc = pf[:, ci : ci + 1]
                tfc = tf[:, ci : ci + 1]
                vm = {
                    "rmse": float(rmse(pfc, tfc).item()),
                    "acc": float(acc(pfc, tfc).item()),
                    "extreme_f1": float(extreme_f1(pfc, tfc).item()),
                    "spectral_distance": float(
                        spectral_distance(pfc, tfc, max_wavenumber=spectral_wavenumbers).item()
                    ),
                }
                for k, v in vm.items():
                    var_lead_meter[key][k].append(v)

    overall_row = {k: _avg(v) for k, v in overall.items()}
    lead_rows: list[dict[str, Any]] = []
    for lead in sorted(lead_meter.keys()):
        row = {"method": method, "lead": lead}
        row.update({k: _avg(v) for k, v in lead_meter[lead].items()})
        lead_rows.append(row)

    var_lead_rows: list[dict[str, Any]] = []
    for lead, var_name in sorted(var_lead_meter.keys(), key=lambda x: (x[0], x[1])):
        row = {"method": method, "lead": int(lead), "variable": var_name}
        row.update({k: _avg(v) for k, v in var_lead_meter[(lead, var_name)].items()})
        var_lead_rows.append(row)
    return overall_row, lead_rows, var_lead_rows


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _report_text(
    config_path: str,
    split: str,
    leads: list[int],
    variables: list[str],
    summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Standardized Alignment Report",
        "",
        "## Protocol",
        f"- config: `{config_path}`",
        f"- split: `{split}`",
        f"- lead_steps: `{leads}`",
        f"- variables: `{variables}`",
        "",
        "## Summary",
    ]
    for row in summary_rows:
        lines.append(
            f"- `{row['method']}`: rmse={row['rmse']:.6f}, acc={row['acc']:.6f}, "
            f"extreme_f1={row['extreme_f1']:.6f}, track_mae={row['track_mae']:.6f}, "
            f"spectral_distance={row['spectral_distance']:.6f}"
        )

    by_method = {r["method"]: r for r in summary_rows}
    if "model" in by_method and "persistence" in by_method:
        m = by_method["model"]
        p = by_method["persistence"]
        gate = (
            float(m["rmse"]) < float(p["rmse"])
            and float(m["acc"]) > float(p["acc"])
            and float(m["extreme_f1"]) >= float(p["extreme_f1"])
        )
        lines += [
            "",
            "## Gate",
            "- rule: `model rmse < persistence rmse` and `model acc > persistence acc` and "
            "`model extreme_f1 >= persistence extreme_f1`",
            f"- result: **{'PASS' if gate else 'FAIL'}**",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one standardized alignment round for MiniLTG.")
    parser.add_argument("--config", required=True, help="Config path.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint for model method.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["model", "persistence", "linear", "climatology"],
        choices=["model", "persistence", "linear", "climatology"],
    )
    parser.add_argument("--leads", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], help="Lead steps to evaluate.")
    parser.add_argument("--max_batches", type=int, default=0, help="Max eval batches (0=all).")
    parser.add_argument("--climatology_max_batches", type=int, default=0, help="Max train batches for climatology.")
    parser.add_argument("--output_dir", default="outputs/miniltg/standardized_round", help="Output directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _seed_everything(
        int(cfg["experiment"]["seed"]),
        deterministic=bool(cfg["experiment"].get("deterministic", False)),
    )
    device = _resolve_device(cfg)
    dataloaders = build_dataloaders(cfg)
    variables = list(cfg["data"]["variables"])
    spectral_wavenumbers = int(cfg.get("evaluation", {}).get("spectral_wavenumbers", 32))
    leads = sorted({int(x) for x in args.leads if int(x) > 0})

    methods = list(dict.fromkeys(args.methods))
    predictors: dict[str, Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]] = {}

    if "model" in methods:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when method includes `model`.")
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
        clim = _build_climatology(dataloaders["train"], max_batches=int(args.climatology_max_batches)).to(device)

        def _predict_climatology(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            b = int(batch["y_future"].shape[0])
            field_pred = clim.unsqueeze(0).expand(b, -1, -1, -1, -1).contiguous()
            traj_pred = batch["traj_hist"][:, -1].unsqueeze(1).repeat(1, int(clim.shape[0]), 1, 1)
            return field_pred, traj_pred

        predictors["climatology"] = _predict_climatology

    summary_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    var_lead_rows: list[dict[str, Any]] = []

    for method in methods:
        summary, leads_out, vars_out = run_alignment(
            method=method,
            predict_fn=predictors[method],
            dataloader=dataloaders[args.split],
            variables=variables,
            leads=leads,
            device=device,
            spectral_wavenumbers=spectral_wavenumbers,
            max_batches=int(args.max_batches),
        )
        row = {"method": method, "split": args.split}
        row.update(summary)
        summary_rows.append(row)
        lead_rows.extend(leads_out)
        var_lead_rows.extend(vars_out)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "alignment_summary.csv"
    lead_csv = out_dir / "alignment_lead.csv"
    var_lead_csv = out_dir / "alignment_var_lead.csv"
    result_json = out_dir / "alignment_results.json"
    report_md = out_dir / "alignment_report.md"

    _save_csv(summary_csv, summary_rows)
    _save_csv(lead_csv, lead_rows)
    _save_csv(var_lead_csv, var_lead_rows)

    with result_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "split": args.split,
                "leads": leads,
                "variables": variables,
                "summary": summary_rows,
                "lead": lead_rows,
                "var_lead": var_lead_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    report_md.write_text(
        _report_text(
            config_path=args.config,
            split=args.split,
            leads=leads,
            variables=variables,
            summary_rows=summary_rows,
        ),
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved lead CSV: {lead_csv}")
    print(f"Saved var-lead CSV: {var_lead_csv}")
    print(f"Saved JSON: {result_json}")
    print(f"Saved report: {report_md}")
    print("Summary:")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
