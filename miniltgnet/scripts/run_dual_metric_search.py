from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniltgnet.config import load_config
from miniltgnet.data import build_dataloaders
from miniltgnet.model import build_model
from miniltgnet.trainer import evaluate_model


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


def _resolve_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["experiment"].get("device", "cuda"))
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_ckpt(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not dict: {path}")
    return ckpt


def _pick_state(ckpt: dict[str, Any], use_ema: bool) -> dict[str, torch.Tensor]:
    if use_ema and ckpt.get("model_ema") is not None:
        return ckpt["model_ema"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    raise ValueError("Checkpoint missing model state.")


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        out[k] = v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
    return out


def _apply_op(
    state: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor] | None,
    prefixes: list[str],
    mode: str,
    alpha: float,
) -> int:
    if donor is None or mode == "none":
        return 0
    updated = 0
    for k in list(state.keys()):
        if not any(k.startswith(p) for p in prefixes):
            continue
        if k not in donor:
            continue
        if mode == "swap":
            state[k] = donor[k].detach().clone()
            updated += 1
        elif mode == "blend":
            base = state[k]
            src = donor[k]
            if (
                isinstance(base, torch.Tensor)
                and isinstance(src, torch.Tensor)
                and base.shape == src.shape
                and torch.is_floating_point(base)
                and torch.is_floating_point(src)
            ):
                b = base.to(torch.float32)
                s = src.to(torch.float32)
                state[k] = ((1.0 - alpha) * b + alpha * s).to(base.dtype)
            else:
                state[k] = src.detach().clone() if isinstance(src, torch.Tensor) else copy.deepcopy(src)
            updated += 1
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    return updated


def _build_candidate_states(
    base_state: dict[str, torch.Tensor],
    track_state: dict[str, torch.Tensor],
    spectral_state: dict[str, torch.Tensor],
    acc_state: dict[str, torch.Tensor] | None,
) -> list[dict[str, Any]]:
    traj_candidates = [("none", 0.0), ("swap", 1.0), ("blend", 0.60), ("blend", 0.75)]
    field_candidates = [("none", 0.0), ("blend", 0.15), ("blend", 0.25)]
    hf_candidates = [("none", 0.0), ("swap", 1.0), ("blend", 0.50)]
    encoder_candidates = [("none", 0.0)]
    if acc_state is not None:
        encoder_candidates.append(("blend", 0.10))

    candidates: list[dict[str, Any]] = []
    for traj_mode, traj_alpha in traj_candidates:
        for field_mode, field_alpha in field_candidates:
            for hf_mode, hf_alpha in hf_candidates:
                for enc_mode, enc_alpha in encoder_candidates:
                    state = _clone_state(base_state)
                    n_traj = _apply_op(
                        state=state,
                        donor=track_state,
                        prefixes=["traj.", "traj_refiner."],
                        mode=traj_mode,
                        alpha=traj_alpha,
                    )
                    n_field = _apply_op(
                        state=state,
                        donor=spectral_state,
                        prefixes=["field_step."],
                        mode=field_mode,
                        alpha=field_alpha,
                    )
                    n_hf = _apply_op(
                        state=state,
                        donor=spectral_state,
                        prefixes=["hf_refiner."],
                        mode=hf_mode,
                        alpha=hf_alpha,
                    )
                    n_enc = _apply_op(
                        state=state,
                        donor=acc_state,
                        prefixes=["encoder."],
                        mode=enc_mode,
                        alpha=enc_alpha,
                    )
                    name = (
                        f"traj-{traj_mode}{traj_alpha:.2f}_field-{field_mode}{field_alpha:.2f}"
                        f"_hf-{hf_mode}{hf_alpha:.2f}_enc-{enc_mode}{enc_alpha:.2f}"
                    )
                    candidates.append(
                        {
                            "name": name,
                            "state": state,
                            "traj_mode": traj_mode,
                            "traj_alpha": traj_alpha,
                            "field_mode": field_mode,
                            "field_alpha": field_alpha,
                            "hf_mode": hf_mode,
                            "hf_alpha": hf_alpha,
                            "encoder_mode": enc_mode,
                            "encoder_alpha": enc_alpha,
                            "updated_keys": int(n_traj + n_field + n_hf + n_enc),
                        }
                    )
    return candidates


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_best_checkpoint(
    template_ckpt: dict[str, Any],
    state: dict[str, torch.Tensor],
    out_path: Path,
    summary: dict[str, Any],
) -> None:
    ckpt = copy.deepcopy(template_ckpt)
    ckpt["model"] = _clone_state(state)
    if ckpt.get("model_ema") is not None:
        ckpt["model_ema"] = _clone_state(state)
    ckpt["dual_metric_search"] = summary
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatic multi-try search for better track_mae + spectral_distance under no-regression constraints."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--base_checkpoint", required=True, help="Overall strong checkpoint.")
    parser.add_argument("--track_checkpoint", required=True, help="Track strong checkpoint.")
    parser.add_argument("--spectral_checkpoint", required=True, help="Spectral strong checkpoint.")
    parser.add_argument("--acc_checkpoint", default="", help="Optional ACC strong checkpoint.")
    parser.add_argument("--output_dir", default="outputs/miniltg/dual_metric_search")
    parser.add_argument("--use_ema_if_available", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--rmse_tol", type=float, default=0.0015)
    parser.add_argument("--acc_tol", type=float, default=0.0015)
    parser.add_argument("--f1_tol", type=float, default=0.0020)
    parser.add_argument("--w_track", type=float, default=0.55)
    parser.add_argument("--w_spec", type=float, default=0.45)
    args = parser.parse_args()

    cfg = load_config(args.config)
    _seed_everything(int(cfg["experiment"]["seed"]), deterministic=bool(cfg["experiment"].get("deterministic", False)))
    device = _resolve_device(cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_checkpoint)
    track_path = Path(args.track_checkpoint)
    spectral_path = Path(args.spectral_checkpoint)
    acc_path = Path(args.acc_checkpoint) if args.acc_checkpoint else None
    for p in [base_path, track_path, spectral_path, acc_path]:
        if p is not None and not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    dls = build_dataloaders(cfg)
    sample = next(iter(dls["train"]))
    in_channels = int(sample["x_hist"].shape[2])
    model = build_model(cfg, in_channels=in_channels).to(device)
    spectral_wavenumbers = int(cfg.get("evaluation", {}).get("spectral_wavenumbers", 32))

    base_ckpt = _load_ckpt(base_path, device)
    track_ckpt = _load_ckpt(track_path, device)
    spectral_ckpt = _load_ckpt(spectral_path, device)
    acc_ckpt = _load_ckpt(acc_path, device) if acc_path is not None else None

    use_ema = bool(args.use_ema_if_available)
    base_state = _pick_state(base_ckpt, use_ema=use_ema)
    track_state = _pick_state(track_ckpt, use_ema=use_ema)
    spectral_state = _pick_state(spectral_ckpt, use_ema=use_ema)
    acc_state = _pick_state(acc_ckpt, use_ema=use_ema) if acc_ckpt is not None else None

    candidates = _build_candidate_states(
        base_state=base_state,
        track_state=track_state,
        spectral_state=spectral_state,
        acc_state=acc_state,
    )
    print(f"Total candidates: {len(candidates)}")

    rows: list[dict[str, Any]] = []
    baseline: dict[str, float] | None = None
    for i, cand in enumerate(candidates, start=1):
        model.load_state_dict(cand["state"], strict=True)
        metrics = evaluate_model(
            model=model,
            dataloader=dls[args.split],
            loss_fn=None,
            device=device,
            max_batches=int(args.max_batches),
            spectral_wavenumbers=spectral_wavenumbers,
        )
        row = {
            "candidate": cand["name"],
            "traj_mode": cand["traj_mode"],
            "traj_alpha": float(cand["traj_alpha"]),
            "field_mode": cand["field_mode"],
            "field_alpha": float(cand["field_alpha"]),
            "hf_mode": cand["hf_mode"],
            "hf_alpha": float(cand["hf_alpha"]),
            "encoder_mode": cand["encoder_mode"],
            "encoder_alpha": float(cand["encoder_alpha"]),
            "updated_keys": int(cand["updated_keys"]),
            "rmse": float(metrics["rmse"]),
            "acc": float(metrics["acc"]),
            "extreme_f1": float(metrics["extreme_f1"]),
            "track_mae": float(metrics["track_mae"]),
            "spectral_distance": float(metrics["spectral_distance"]),
        }
        rows.append(row)
        if cand["name"] == "traj-none0.00_field-none0.00_hf-none0.00_enc-none0.00":
            baseline = {k: float(row[k]) for k in ["rmse", "acc", "extreme_f1", "track_mae", "spectral_distance"]}
        print(f"[{i:03d}/{len(candidates):03d}] {cand['name']} -> {metrics}")

    if baseline is None:
        raise RuntimeError("Baseline candidate missing.")

    best_feasible: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None
    for row in rows:
        gain_track = (baseline["track_mae"] - row["track_mae"]) / (abs(baseline["track_mae"]) + 1e-8)
        gain_spec = (baseline["spectral_distance"] - row["spectral_distance"]) / (abs(baseline["spectral_distance"]) + 1e-8)
        objective = float(args.w_track * gain_track + args.w_spec * gain_spec)
        row["gain_track"] = float(gain_track)
        row["gain_spectral"] = float(gain_spec)
        row["objective"] = objective
        row["delta_rmse"] = float(row["rmse"] - baseline["rmse"])
        row["delta_acc"] = float(row["acc"] - baseline["acc"])
        row["delta_extreme_f1"] = float(row["extreme_f1"] - baseline["extreme_f1"])
        row["feasible"] = bool(
            row["delta_rmse"] <= float(args.rmse_tol)
            and row["delta_acc"] >= -float(args.acc_tol)
            and row["delta_extreme_f1"] >= -float(args.f1_tol)
        )
        if best_any is None or objective > float(best_any["objective"]):
            best_any = row
        if row["feasible"] and (best_feasible is None or objective > float(best_feasible["objective"])):
            best_feasible = row

    assert best_any is not None
    chosen = best_feasible if best_feasible is not None else best_any
    chosen_name = str(chosen["candidate"])
    chosen_state = next(c["state"] for c in candidates if c["name"] == chosen_name)

    rows_sorted = sorted(rows, key=lambda x: float(x["objective"]), reverse=True)
    _save_csv(out_dir / "dual_metric_search_results.csv", rows_sorted)

    best_ckpt_path = out_dir / "best_dual_metric.pt"
    summary = {
        "baseline": baseline,
        "best_feasible": best_feasible,
        "best_any": best_any,
        "chosen": chosen,
        "constraints": {"rmse_tol": args.rmse_tol, "acc_tol": args.acc_tol, "f1_tol": args.f1_tol},
        "weights": {"w_track": args.w_track, "w_spec": args.w_spec},
    }
    _save_best_checkpoint(base_ckpt, chosen_state, best_ckpt_path, summary)
    with (out_dir / "dual_metric_search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Dual Metric Search Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved CSV: {out_dir / 'dual_metric_search_results.csv'}")
    print(f"Saved best checkpoint: {best_ckpt_path}")
    print(f"Saved JSON: {out_dir / 'dual_metric_search_summary.json'}")


if __name__ == "__main__":
    main()
