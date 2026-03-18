from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltg_net.config import load_config
from ltg_net.data.datamodule import build_dataloaders
from ltg_net.models import LTGNet
from ltg_net.train.loops import move_batch_to_device
from ltg_net.utils.metrics import acc, extreme_f1, rmse, spectral_metric, track_mae


METRICS = ["rmse", "acc", "track_mae", "extreme_f1", "spectral_distance"]
LOWER_BETTER = {"rmse", "track_mae", "spectral_distance"}


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


def _resolve_device(cfg: dict[str, Any], requested: str | None) -> torch.device:
    if requested:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    from_cfg = str(cfg["experiment"].get("device", "cuda"))
    if from_cfg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(cfg: dict[str, Any], dataloaders: dict[str, torch.utils.data.DataLoader], device: torch.device) -> LTGNet:
    sample = next(iter(dataloaders["train"]))
    in_channels = int(sample["x_hist"].shape[2])
    model = LTGNet(cfg=cfg, in_channels=in_channels).to(device)
    return model


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _select_lead_indices(forecast_steps: int, leads: list[int] | None) -> list[int]:
    if not leads:
        return list(range(forecast_steps))
    out: list[int] = []
    for lead in leads:
        idx = int(lead) - 1
        if idx < 0 or idx >= forecast_steps:
            raise ValueError(f"Lead step out of range: {lead}, forecast_steps={forecast_steps}")
        out.append(idx)
    return sorted(set(out))


def _subset_leads_field(field: torch.Tensor, lead_indices: list[int]) -> torch.Tensor:
    return field[:, lead_indices]


def _subset_leads_traj(traj: torch.Tensor, lead_indices: list[int]) -> torch.Tensor:
    return traj[:, lead_indices]


def _predict_persistence(batch: dict[str, torch.Tensor], horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    last_field = batch["x_hist"][:, -1].float()
    last_traj = batch["traj_hist"][:, -1].float()
    field_pred = last_field.unsqueeze(1).repeat(1, horizon, 1, 1, 1)
    traj_pred = last_traj.unsqueeze(1).repeat(1, horizon, 1, 1)
    return field_pred, traj_pred


def _clamp_geo(state: torch.Tensor) -> torch.Tensor:
    lat = state[..., 0].clamp(-90.0, 90.0)
    lon = state[..., 1] % 360.0
    return torch.stack([lat, lon], dim=-1)


def _predict_linear(batch: dict[str, torch.Tensor], horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_hist = batch["x_hist"].float()
    traj_hist = batch["traj_hist"].float()
    if x_hist.shape[1] < 2:
        return _predict_persistence(batch, horizon=horizon)

    prev_field = x_hist[:, -1]
    delta_field = x_hist[:, -1] - x_hist[:, -2]
    field_steps: list[torch.Tensor] = []
    for _ in range(horizon):
        prev_field = prev_field + delta_field
        field_steps.append(prev_field)
    field_pred = torch.stack(field_steps, dim=1)

    prev_traj = traj_hist[:, -1]
    delta_lat = traj_hist[:, -1, :, 0] - traj_hist[:, -2, :, 0]
    delta_lon = (traj_hist[:, -1, :, 1] - traj_hist[:, -2, :, 1] + 180.0) % 360.0 - 180.0
    traj_steps: list[torch.Tensor] = []
    for _ in range(horizon):
        lat = prev_traj[..., 0] + delta_lat
        lon = prev_traj[..., 1] + delta_lon
        prev_traj = _clamp_geo(torch.stack([lat, lon], dim=-1))
        traj_steps.append(prev_traj)
    traj_pred = torch.stack(traj_steps, dim=1)
    return field_pred, traj_pred


@torch.no_grad()
def _compute_climatology(
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_field = None
    sum_traj = None
    count = 0
    for step_idx, batch in enumerate(tqdm(train_loader, desc="climatology-fit", leave=False)):
        if max_batches > 0 and step_idx >= max_batches:
            break
        y_future = batch["y_future"].to(device).float()
        traj_future = batch["traj_future"].to(device).float()
        bsz = int(y_future.shape[0])
        if sum_field is None:
            sum_field = y_future.sum(dim=0)
            sum_traj = traj_future.sum(dim=0)
        else:
            sum_field = sum_field + y_future.sum(dim=0)
            sum_traj = sum_traj + traj_future.sum(dim=0)
        count += bsz
    if count <= 0 or sum_field is None or sum_traj is None:
        raise RuntimeError("No data available to compute climatology baseline.")
    return sum_field / float(count), sum_traj / float(count)


def _predict_climatology(
    batch: dict[str, torch.Tensor],
    clim_field: torch.Tensor,
    clim_traj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = int(batch["x_hist"].shape[0])
    field_pred = clim_field.unsqueeze(0).repeat(bsz, 1, 1, 1, 1)
    traj_pred = clim_traj.unsqueeze(0).repeat(bsz, 1, 1, 1)
    return field_pred, traj_pred


def _sample_metric_bundle(
    pred_field: torch.Tensor,
    target_field: torch.Tensor,
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    max_wavenumber: int,
    extreme_quantile: float,
    max_quantile_elements: int,
) -> dict[str, float]:
    return {
        "rmse": float(rmse(pred_field, target_field).item()),
        "acc": float(acc(pred_field, target_field).item()),
        "track_mae": float(track_mae(pred_traj, target_traj).item()),
        "extreme_f1": float(
            extreme_f1(
                pred_field,
                target_field,
                quantile=extreme_quantile,
                max_quantile_elements=max_quantile_elements,
            ).item()
        ),
        "spectral_distance": float(spectral_metric(pred_field[:, 0], target_field[:, 0], max_wavenumber).item()),
    }


def _bootstrap_ci_mean(values: np.ndarray, n_bootstrap: int, alpha: float, rng: np.random.Generator) -> tuple[float, float]:
    if values.size <= 1 or n_bootstrap <= 0:
        mean = float(values.mean()) if values.size > 0 else float("nan")
        return mean, mean
    n = int(values.size)
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha * 0.5))
    hi = float(np.quantile(means, 1.0 - alpha * 0.5))
    return lo, hi


def _bootstrap_p_better(
    model_values: np.ndarray,
    ref_values: np.ndarray,
    metric: str,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    if model_values.size != ref_values.size:
        raise ValueError(f"Length mismatch for metric={metric}: {model_values.size} vs {ref_values.size}")
    if metric in LOWER_BETTER:
        diff = ref_values - model_values
    else:
        diff = model_values - ref_values
    delta_mean = float(diff.mean())
    if diff.size <= 1 or n_bootstrap <= 0:
        p_better = float((diff > 0.0).mean())
        return delta_mean, p_better, delta_mean, delta_mean
    n = int(diff.size)
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    boot_means = diff[idx].mean(axis=1)
    p_better = float((boot_means > 0.0).mean())
    ci_lo = float(np.quantile(boot_means, alpha * 0.5))
    ci_hi = float(np.quantile(boot_means, 1.0 - alpha * 0.5))
    return delta_mean, p_better, ci_lo, ci_hi


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_report(
    cfg_path: str,
    ckpt_path: str | None,
    split: str,
    methods: list[str],
    leads: list[int],
    summary_rows: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# LTG Strict Benchmark Report")
    lines.append("")
    lines.append("## Protocol")
    lines.append(f"- config: `{cfg_path}`")
    if ckpt_path:
        lines.append(f"- checkpoint: `{ckpt_path}`")
    lines.append(f"- split: `{split}`")
    lines.append(f"- methods: `{methods}`")
    lines.append(f"- lead_steps: `{leads}`")
    lines.append("")
    lines.append("## Summary")
    for row in summary_rows:
        lines.append(
            f"- `{row['method']}`: rmse={row['rmse']:.6f}, acc={row['acc']:.6f}, "
            f"track_mae={row['track_mae']:.6f}, extreme_f1={row['extreme_f1']:.6f}, "
            f"spectral_distance={row['spectral_distance']:.6f}"
        )
    lines.append("")
    lines.append("## Skill vs Reference")
    ref = gate.get("reference")
    lines.append(f"- reference: `{ref}`")
    for row in skill_rows:
        lines.append(
            f"- `{row['method']}` / `{row['metric']}`: delta={row['delta_mean']:.6f}, "
            f"p_better={row['p_better']:.3f}, ci=[{row['delta_ci_lower']:.6f}, {row['delta_ci_upper']:.6f}]"
        )
    lines.append("")
    lines.append("## Gate")
    lines.append(f"- rule: `{gate['rule']}`")
    lines.append(f"- strict_p_threshold: `{gate['strict_p_threshold']}`")
    lines.append(f"- result: **{'PASS' if gate['pass'] else 'FAIL'}**")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict benchmark for LTG-Net with baseline methods.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="", help="Model checkpoint path (required when method includes model).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["model", "persistence", "linear", "climatology"],
        choices=["model", "persistence", "linear", "climatology"],
    )
    parser.add_argument("--leads", nargs="*", type=int, default=None, help="1-based lead steps to evaluate.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--ci_alpha", type=float, default=0.05)
    parser.add_argument("--reference", default="persistence")
    parser.add_argument("--strict_p_threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="", choices=["", "cpu", "cuda"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--climatology_train_max_batches", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/ltg/strict_benchmark")
    args = parser.parse_args()

    methods = [str(m) for m in args.methods]
    if "model" in methods and not args.checkpoint:
        raise ValueError("--checkpoint is required when methods include 'model'.")
    if args.reference not in methods:
        raise ValueError(f"--reference must be included in methods. got reference={args.reference}, methods={methods}")

    cfg = load_config(args.config)
    _seed_everything(int(args.seed), deterministic=bool(args.deterministic))
    device = _resolve_device(cfg, requested=(args.device or None))
    print(f"Device: {device.type}")

    dataloaders = build_dataloaders(cfg)
    split_loader = dataloaders[args.split]
    sample = next(iter(split_loader))
    horizon = int(sample["y_future"].shape[1])
    lead_indices = _select_lead_indices(horizon, args.leads)
    lead_steps = [idx + 1 for idx in lead_indices]

    model = None
    if "model" in methods:
        model = _build_model(cfg, dataloaders=dataloaders, device=device)
        state = _load_checkpoint(args.checkpoint, device=device)
        model.load_state_dict(state)
        model.eval()

    clim_field = None
    clim_traj = None
    if "climatology" in methods:
        clim_field, clim_traj = _compute_climatology(
            train_loader=dataloaders["train"],
            device=device,
            max_batches=int(args.climatology_train_max_batches),
        )

    metric_values: dict[str, dict[str, list[float]]] = {
        m: {k: [] for k in METRICS} for m in methods
    }
    lead_metric_values: dict[str, dict[int, dict[str, list[float]]]] = {
        m: {lead: {k: [] for k in METRICS} for lead in lead_steps} for m in methods
    }

    max_wavenumber = int(cfg["loss"]["spectral_wavenumbers"])
    extreme_quantile = float(cfg["loss"].get("field_extreme_quantile", 0.98))
    max_quantile_elements = int(cfg["loss"].get("field_extreme_max_elements", 5_000_000))

    with torch.no_grad():
        pbar = tqdm(split_loader, desc=f"strict-{args.split}", leave=False)
        for step_idx, batch in enumerate(pbar):
            if args.max_batches > 0 and step_idx >= int(args.max_batches):
                break
            batch = move_batch_to_device(batch, device)
            target_field = _subset_leads_field(batch["y_future"].float(), lead_indices)
            target_traj = _subset_leads_traj(batch["traj_future"].float(), lead_indices)

            pred_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            if "model" in methods:
                assert model is not None
                out = model(batch)
                pred_cache["model"] = (
                    _subset_leads_field(out["field_pred"], lead_indices),
                    _subset_leads_traj(out["traj_pred"], lead_indices),
                )
            if "persistence" in methods:
                pf, pt = _predict_persistence(batch, horizon=horizon)
                pred_cache["persistence"] = (_subset_leads_field(pf, lead_indices), _subset_leads_traj(pt, lead_indices))
            if "linear" in methods:
                pf, pt = _predict_linear(batch, horizon=horizon)
                pred_cache["linear"] = (_subset_leads_field(pf, lead_indices), _subset_leads_traj(pt, lead_indices))
            if "climatology" in methods:
                assert clim_field is not None and clim_traj is not None
                pf, pt = _predict_climatology(batch, clim_field=clim_field, clim_traj=clim_traj)
                pred_cache["climatology"] = (
                    _subset_leads_field(pf, lead_indices),
                    _subset_leads_traj(pt, lead_indices),
                )

            bsz = int(target_field.shape[0])
            for method in methods:
                pred_field, pred_traj = pred_cache[method]
                for i in range(bsz):
                    pf_i = pred_field[i : i + 1]
                    tf_i = target_field[i : i + 1]
                    pt_i = pred_traj[i : i + 1]
                    tt_i = target_traj[i : i + 1]
                    bundle = _sample_metric_bundle(
                        pred_field=pf_i,
                        target_field=tf_i,
                        pred_traj=pt_i,
                        target_traj=tt_i,
                        max_wavenumber=max_wavenumber,
                        extreme_quantile=extreme_quantile,
                        max_quantile_elements=max_quantile_elements,
                    )
                    for metric, value in bundle.items():
                        metric_values[method][metric].append(float(value))

                    for li, lead in enumerate(lead_steps):
                        pf_l = pf_i[:, li : li + 1]
                        tf_l = tf_i[:, li : li + 1]
                        pt_l = pt_i[:, li : li + 1]
                        tt_l = tt_i[:, li : li + 1]
                        lead_bundle = _sample_metric_bundle(
                            pred_field=pf_l,
                            target_field=tf_l,
                            pred_traj=pt_l,
                            target_traj=tt_l,
                            max_wavenumber=max_wavenumber,
                            extreme_quantile=extreme_quantile,
                            max_quantile_elements=max_quantile_elements,
                        )
                        for metric, value in lead_bundle.items():
                            lead_metric_values[method][lead][metric].append(float(value))

    rng = np.random.default_rng(int(args.seed))
    summary_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    for method in methods:
        row = {"method": method, "split": args.split}
        for metric in METRICS:
            arr = np.asarray(metric_values[method][metric], dtype=np.float64)
            mean = float(arr.mean()) if arr.size > 0 else float("nan")
            row[metric] = mean
            lo, hi = _bootstrap_ci_mean(
                arr,
                n_bootstrap=int(args.bootstrap),
                alpha=float(args.ci_alpha),
                rng=rng,
            )
            ci_rows.append(
                {
                    "method": method,
                    "split": args.split,
                    "metric": metric,
                    "mean": mean,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_samples": int(arr.size),
                }
            )
        summary_rows.append(row)

        for lead in lead_steps:
            lead_row = {"method": method, "split": args.split, "lead": int(lead)}
            for metric in METRICS:
                arr = np.asarray(lead_metric_values[method][lead][metric], dtype=np.float64)
                lead_row[metric] = float(arr.mean()) if arr.size > 0 else float("nan")
            lead_rows.append(lead_row)

    skill_rows: list[dict[str, Any]] = []
    lead_skill_rows: list[dict[str, Any]] = []
    ref = str(args.reference)
    ref_metrics = {k: np.asarray(metric_values[ref][k], dtype=np.float64) for k in METRICS}
    for method in methods:
        if method == ref:
            continue
        for metric in METRICS:
            model_arr = np.asarray(metric_values[method][metric], dtype=np.float64)
            ref_arr = ref_metrics[metric]
            delta_mean, p_better, lo, hi = _bootstrap_p_better(
                model_values=model_arr,
                ref_values=ref_arr,
                metric=metric,
                n_bootstrap=int(args.bootstrap),
                alpha=float(args.ci_alpha),
                rng=rng,
            )
            skill_rows.append(
                {
                    "method": method,
                    "reference": ref,
                    "metric": metric,
                    "delta_mean": delta_mean,
                    "p_better": p_better,
                    "delta_ci_lower": lo,
                    "delta_ci_upper": hi,
                }
            )

        for lead in lead_steps:
            for metric in METRICS:
                model_arr = np.asarray(lead_metric_values[method][lead][metric], dtype=np.float64)
                ref_arr = np.asarray(lead_metric_values[ref][lead][metric], dtype=np.float64)
                delta_mean, p_better, lo, hi = _bootstrap_p_better(
                    model_values=model_arr,
                    ref_values=ref_arr,
                    metric=metric,
                    n_bootstrap=int(args.bootstrap),
                    alpha=float(args.ci_alpha),
                    rng=rng,
                )
                lead_skill_rows.append(
                    {
                        "method": method,
                        "reference": ref,
                        "lead": int(lead),
                        "metric": metric,
                        "delta_mean": delta_mean,
                        "p_better": p_better,
                        "delta_ci_lower": lo,
                        "delta_ci_upper": hi,
                    }
                )

    summary_by_method = {r["method"]: r for r in summary_rows}
    model_row = summary_by_method.get("model")
    ref_row = summary_by_method.get(ref)
    p_map = {(r["method"], r["metric"]): float(r["p_better"]) for r in skill_rows}
    base_gate = False
    strict_gate = False
    if model_row is not None and ref_row is not None:
        base_gate = (
            float(model_row["rmse"]) < float(ref_row["rmse"])
            and float(model_row["acc"]) > float(ref_row["acc"])
            and float(model_row["extreme_f1"]) >= float(ref_row["extreme_f1"])
        )
        strict_gate = (
            base_gate
            and p_map.get(("model", "rmse"), 0.0) >= float(args.strict_p_threshold)
            and p_map.get(("model", "acc"), 0.0) >= float(args.strict_p_threshold)
            and p_map.get(("model", "extreme_f1"), 0.0) >= float(args.strict_p_threshold)
        )

    gate = {
        "reference": ref,
        "rule": (
            "model rmse < reference rmse and "
            "model acc > reference acc and "
            "model extreme_f1 >= reference extreme_f1 and "
            "p_better(rmse/acc/extreme_f1) >= strict_p_threshold"
        ),
        "strict_p_threshold": float(args.strict_p_threshold),
        "base_pass": bool(base_gate),
        "pass": bool(strict_gate),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "strict_summary.csv"
    ci_csv = out_dir / "strict_ci.csv"
    lead_csv = out_dir / "strict_lead_summary.csv"
    skill_csv = out_dir / f"strict_skill_vs_{ref}.csv"
    lead_skill_csv = out_dir / f"strict_lead_skill_vs_{ref}.csv"
    result_json = out_dir / "strict_results.json"
    report_md = out_dir / "strict_report.md"

    _write_csv(
        summary_csv,
        summary_rows,
        fieldnames=["method", "split", *METRICS],
    )
    _write_csv(
        ci_csv,
        ci_rows,
        fieldnames=["method", "split", "metric", "mean", "ci_lower", "ci_upper", "n_samples"],
    )
    _write_csv(
        lead_csv,
        lead_rows,
        fieldnames=["method", "split", "lead", *METRICS],
    )
    _write_csv(
        skill_csv,
        skill_rows,
        fieldnames=["method", "reference", "metric", "delta_mean", "p_better", "delta_ci_lower", "delta_ci_upper"],
    )
    _write_csv(
        lead_skill_csv,
        lead_skill_rows,
        fieldnames=[
            "method",
            "reference",
            "lead",
            "metric",
            "delta_mean",
            "p_better",
            "delta_ci_lower",
            "delta_ci_upper",
        ],
    )

    result_payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "methods": methods,
        "leads": lead_steps,
        "bootstrap": int(args.bootstrap),
        "ci_alpha": float(args.ci_alpha),
        "summary": summary_rows,
        "ci": ci_rows,
        "lead_summary": lead_rows,
        "skill_vs_reference": skill_rows,
        "lead_skill_vs_reference": lead_skill_rows,
        "gate": gate,
    }
    result_json.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md.write_text(
        _build_report(
            cfg_path=args.config,
            ckpt_path=args.checkpoint if args.checkpoint else None,
            split=args.split,
            methods=methods,
            leads=lead_steps,
            summary_rows=summary_rows,
            skill_rows=skill_rows,
            gate=gate,
        ),
        encoding="utf-8",
    )

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved CI CSV: {ci_csv}")
    print(f"Saved lead CSV: {lead_csv}")
    print(f"Saved skill CSV: {skill_csv}")
    print(f"Saved lead-skill CSV: {lead_skill_csv}")
    print(f"Saved JSON: {result_json}")
    print(f"Saved report: {report_md}")
    print("Summary:")
    for row in summary_rows:
        print(row)
    print(f"Gate: {'PASS' if gate['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
