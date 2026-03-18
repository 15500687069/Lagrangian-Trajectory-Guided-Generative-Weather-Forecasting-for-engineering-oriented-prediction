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


def _load_ckpt(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint not dict: {path}")
    return ckpt


def _extract_state(ckpt: dict[str, Any], use_ema: bool) -> dict[str, torch.Tensor]:
    if use_ema and "model_ema" in ckpt and ckpt["model_ema"] is not None:
        return ckpt["model_ema"]
    if "model" in ckpt:
        return ckpt["model"]
    raise KeyError("Checkpoint missing `model`.")


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        out[k] = v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
    return out


def _is_prefix_key(key: str, prefixes: list[str]) -> bool:
    return any(key.startswith(p) for p in prefixes)


def _blend_tensor(base: torch.Tensor, donor: torch.Tensor, alpha: float) -> torch.Tensor:
    if not (isinstance(base, torch.Tensor) and isinstance(donor, torch.Tensor)):
        return donor
    if base.shape != donor.shape:
        return donor
    if torch.is_floating_point(base) and torch.is_floating_point(donor):
        b = base.to(torch.float32)
        d = donor.to(torch.float32)
        out = (1.0 - alpha) * b + alpha * d
        return out.to(dtype=base.dtype)
    return donor


def _apply_recipe(
    base_state: dict[str, torch.Tensor],
    donors: dict[str, dict[str, torch.Tensor]],
    recipe: dict[str, Any],
) -> dict[str, torch.Tensor]:
    state = _clone_state(base_state)
    for op in recipe.get("ops", []):
        donor_name = str(op["donor"])
        donor_state = donors[donor_name]
        prefixes = list(op["prefixes"])
        mode = str(op.get("mode", "swap"))
        alpha = float(op.get("alpha", 1.0))
        for k in list(state.keys()):
            if not _is_prefix_key(k, prefixes):
                continue
            if k not in donor_state:
                continue
            if mode == "swap":
                state[k] = donor_state[k].detach().clone()
            elif mode == "blend":
                state[k] = _blend_tensor(state[k], donor_state[k], alpha=alpha)
            else:
                raise ValueError(f"Unsupported op mode: {mode}")
    return state


def _candidate_recipes() -> list[dict[str, Any]]:
    recipes: list[dict[str, Any]] = []
    recipes.append({"name": "baseline_overall", "ops": []})

    # Global soups (overall <- track).
    for a in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        recipes.append(
            {
                "name": f"global_blend_track_a{int(a*100):02d}",
                "ops": [{"donor": "track", "prefixes": [""], "mode": "blend", "alpha": a}],
            }
        )

    # Direct module swap.
    recipes.append({"name": "swap_traj", "ops": [{"donor": "track", "prefixes": ["traj."], "mode": "swap"}]})
    recipes.append(
        {"name": "swap_field_step", "ops": [{"donor": "track", "prefixes": ["field_step."], "mode": "swap"}]}
    )
    recipes.append(
        {
            "name": "swap_traj_field_step",
            "ops": [
                {"donor": "track", "prefixes": ["traj."], "mode": "swap"},
                {"donor": "track", "prefixes": ["field_step."], "mode": "swap"},
            ],
        }
    )

    # Module-level blend for trajectory and field.
    for a in [0.25, 0.40, 0.55, 0.70]:
        recipes.append(
            {
                "name": f"blend_traj_a{int(a*100):02d}",
                "ops": [{"donor": "track", "prefixes": ["traj."], "mode": "blend", "alpha": a}],
            }
        )
    for a in [0.20, 0.35, 0.50]:
        recipes.append(
            {
                "name": f"blend_field_a{int(a*100):02d}",
                "ops": [{"donor": "track", "prefixes": ["field_step."], "mode": "blend", "alpha": a}],
            }
        )

    # Hybrid recipes with acc donor on encoder to protect field skill.
    recipes.append(
        {
            "name": "blend_traj50_encoder10acc",
            "ops": [
                {"donor": "track", "prefixes": ["traj."], "mode": "blend", "alpha": 0.50},
                {"donor": "acc", "prefixes": ["encoder."], "mode": "blend", "alpha": 0.10},
            ],
        }
    )
    recipes.append(
        {
            "name": "blend_traj60_encoder15acc",
            "ops": [
                {"donor": "track", "prefixes": ["traj."], "mode": "blend", "alpha": 0.60},
                {"donor": "acc", "prefixes": ["encoder."], "mode": "blend", "alpha": 0.15},
            ],
        }
    )
    recipes.append(
        {
            "name": "blend_traj40_field20_encoder10acc",
            "ops": [
                {"donor": "track", "prefixes": ["traj."], "mode": "blend", "alpha": 0.40},
                {"donor": "track", "prefixes": ["field_step."], "mode": "blend", "alpha": 0.20},
                {"donor": "acc", "prefixes": ["encoder."], "mode": "blend", "alpha": 0.10},
            ],
        }
    )
    recipes.append(
        {
            "name": "swap_traj_blend_field30",
            "ops": [
                {"donor": "track", "prefixes": ["traj."], "mode": "swap"},
                {"donor": "track", "prefixes": ["field_step."], "mode": "blend", "alpha": 0.30},
            ],
        }
    )
    recipes.append(
        {
            "name": "swap_traj_blend_field20_encoder10acc",
            "ops": [
                {"donor": "track", "prefixes": ["traj."], "mode": "swap"},
                {"donor": "track", "prefixes": ["field_step."], "mode": "blend", "alpha": 0.20},
                {"donor": "acc", "prefixes": ["encoder."], "mode": "blend", "alpha": 0.10},
            ],
        }
    )
    return recipes


def _score_row(
    row: dict[str, Any],
    baseline: dict[str, float],
    w_track: float,
    w_spec: float,
) -> tuple[float, float, float]:
    gain_track = (baseline["track_mae"] - float(row["track_mae"])) / (abs(baseline["track_mae"]) + 1e-8)
    gain_spec = (baseline["spectral_distance"] - float(row["spectral_distance"])) / (
        abs(baseline["spectral_distance"]) + 1e-8
    )
    objective = float(w_track * gain_track + w_spec * gain_spec)
    return objective, gain_track, gain_spec


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint_from_state(
    template_ckpt: dict[str, Any],
    state: dict[str, torch.Tensor],
    out_path: Path,
) -> None:
    ckpt = copy.deepcopy(template_ckpt)
    ckpt["model"] = _clone_state(state)
    if "model_ema" in ckpt and ckpt["model_ema"] is not None:
        ckpt["model_ema"] = _clone_state(state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search stitched checkpoints for better track_mae + spectral_distance.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--overall_ckpt", required=True)
    parser.add_argument("--track_ckpt", required=True)
    parser.add_argument("--acc_ckpt", required=True)
    parser.add_argument("--output_dir", default="outputs/miniltg/stitch_search")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--w_track", type=float, default=0.6)
    parser.add_argument("--w_spec", type=float, default=0.4)
    parser.add_argument("--rmse_tol", type=float, default=0.0015)
    parser.add_argument("--acc_tol", type=float, default=0.0015)
    parser.add_argument("--f1_tol", type=float, default=0.0020)
    args = parser.parse_args()

    cfg = load_config(args.config)
    _seed_everything(int(cfg["experiment"]["seed"]), deterministic=bool(cfg["experiment"].get("deterministic", False)))
    device = _resolve_device(cfg)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dls = build_dataloaders(cfg)
    in_channels = int(next(iter(dls["train"]))["x_hist"].shape[2])
    spectral_wavenumbers = int(cfg.get("evaluation", {}).get("spectral_wavenumbers", 32))

    overall_ckpt = _load_ckpt(Path(args.overall_ckpt), device)
    track_ckpt = _load_ckpt(Path(args.track_ckpt), device)
    acc_ckpt = _load_ckpt(Path(args.acc_ckpt), device)
    use_ema = bool(cfg.get("evaluation", {}).get("use_ema_for_eval", True))

    base_state = _extract_state(overall_ckpt, use_ema=use_ema)
    donors = {
        "overall": _extract_state(overall_ckpt, use_ema=use_ema),
        "track": _extract_state(track_ckpt, use_ema=use_ema),
        "acc": _extract_state(acc_ckpt, use_ema=use_ema),
    }

    recipes = _candidate_recipes()
    rows: list[dict[str, Any]] = []
    best_feasible: dict[str, Any] | None = None
    best_objective_any: dict[str, Any] | None = None

    for idx, recipe in enumerate(recipes, start=1):
        name = recipe["name"]
        state = _apply_recipe(base_state=base_state, donors=donors, recipe=recipe)

        model = build_model(cfg, in_channels=in_channels).to(device)
        model.load_state_dict(state)
        metrics = evaluate_model(
            model=model,
            dataloader=dls[args.split],
            loss_fn=None,
            device=device,
            max_batches=int(args.max_batches),
            spectral_wavenumbers=spectral_wavenumbers,
        )

        row: dict[str, Any] = {
            "candidate": name,
            "rmse": float(metrics["rmse"]),
            "acc": float(metrics["acc"]),
            "extreme_f1": float(metrics["extreme_f1"]),
            "track_mae": float(metrics["track_mae"]),
            "spectral_distance": float(metrics["spectral_distance"]),
        }
        rows.append(row)
        print(f"[{idx:02d}/{len(recipes):02d}] {name}: {row}")

    baseline = next((r for r in rows if r["candidate"] == "baseline_overall"), None)
    if baseline is None:
        raise RuntimeError("Missing baseline_overall in rows.")

    for row in rows:
        objective, gain_track, gain_spec = _score_row(
            row=row,
            baseline=baseline,
            w_track=float(args.w_track),
            w_spec=float(args.w_spec),
        )
        row["gain_track"] = gain_track
        row["gain_spectral"] = gain_spec
        row["objective"] = objective
        row["delta_rmse"] = float(row["rmse"] - baseline["rmse"])
        row["delta_acc"] = float(row["acc"] - baseline["acc"])
        row["delta_extreme_f1"] = float(row["extreme_f1"] - baseline["extreme_f1"])

        feasible = (
            row["delta_rmse"] <= float(args.rmse_tol)
            and row["delta_acc"] >= -float(args.acc_tol)
            and row["delta_extreme_f1"] >= -float(args.f1_tol)
        )
        row["feasible"] = bool(feasible)

        if best_objective_any is None or float(row["objective"]) > float(best_objective_any["objective"]):
            best_objective_any = row
        if feasible and (best_feasible is None or float(row["objective"]) > float(best_feasible["objective"])):
            best_feasible = row

    rows_sorted = sorted(rows, key=lambda x: float(x["objective"]), reverse=True)
    _save_csv(out_dir / "search_results.csv", rows_sorted)

    chosen = best_feasible if best_feasible is not None else best_objective_any
    assert chosen is not None
    chosen_name = str(chosen["candidate"])
    chosen_recipe = next(r for r in recipes if r["name"] == chosen_name)
    chosen_state = _apply_recipe(base_state=base_state, donors=donors, recipe=chosen_recipe)

    best_ckpt_path = out_dir / "best_stitched.pt"
    _save_checkpoint_from_state(template_ckpt=overall_ckpt, state=chosen_state, out_path=best_ckpt_path)

    summary = {
        "baseline": baseline,
        "best_feasible": best_feasible,
        "best_any": best_objective_any,
        "chosen": chosen,
        "chosen_checkpoint": str(best_ckpt_path),
        "constraints": {
            "rmse_tol": float(args.rmse_tol),
            "acc_tol": float(args.acc_tol),
            "f1_tol": float(args.f1_tol),
        },
        "weights": {"w_track": float(args.w_track), "w_spec": float(args.w_spec)},
    }
    with (out_dir / "search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Stitch Search Summary ===")
    print(f"baseline: {baseline}")
    print(f"best_feasible: {best_feasible}")
    print(f"best_any: {best_objective_any}")
    print(f"chosen: {chosen}")
    print(f"saved checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
