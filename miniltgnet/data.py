from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset


def _open_era5(path: str | Path) -> xr.Dataset:
    path = Path(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path, consolidated=False)
    return xr.open_dataset(path)


def _select_region(ds: xr.Dataset, region_cfg: dict[str, Any]) -> xr.Dataset:
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
    return ds.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))


def _flatten_variable(da: xr.DataArray) -> np.ndarray:
    arr = da.values.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, None]  # [T,1,H,W]
    elif arr.ndim != 4:
        raise ValueError(f"Unsupported variable ndim for {da.name}: {arr.shape}")
    return arr


def _np_shape1(x: np.ndarray) -> np.ndarray:
    if x.ndim == 0:
        return x.reshape(1)
    return x


def compute_variable_stats(ds: xr.Dataset, variables: list[str]) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for var in variables:
        arr = _flatten_variable(ds[var])
        mean = arr.mean(axis=(0, 2, 3)).astype(np.float32)
        std = arr.std(axis=(0, 2, 3)).astype(np.float32)
        std = np.maximum(std, 1e-6)
        stats[var] = {"mean": _np_shape1(mean), "std": _np_shape1(std)}
    return stats


def save_stats(path: str | Path, stats: dict[str, dict[str, np.ndarray]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for var, item in stats.items():
        payload[f"{var}__mean"] = item["mean"].astype(np.float32)
        payload[f"{var}__std"] = item["std"].astype(np.float32)
    np.savez(path, **payload)


def load_stats(path: str | Path, variables: list[str]) -> dict[str, dict[str, np.ndarray]]:
    npz = np.load(Path(path))
    out: dict[str, dict[str, np.ndarray]] = {}
    for var in variables:
        out[var] = {
            "mean": _np_shape1(np.asarray(npz[f"{var}__mean"], dtype=np.float32)),
            "std": _np_shape1(np.asarray(npz[f"{var}__std"], dtype=np.float32)),
        }
    return out


def prepare_data(cfg: dict[str, Any]) -> None:
    data_cfg = cfg["data"]
    track_path = Path(data_cfg["track_cache_path"])
    if not track_path.exists():
        raise FileNotFoundError(
            f"Track cache not found: {track_path}. "
            "Please prepare tracks once with the original ltg_net pipeline first."
        )
    tracks = xr.open_dataset(track_path)
    if "traj_lat" not in tracks or "traj_lon" not in tracks:
        raise ValueError(
            f"Track cache missing traj_lat/traj_lon variables: {track_path}. "
            "Please regenerate via `python -m ltg_net.cli prepare --config <config>`."
        )
    lat_nan_ratio = float(np.isnan(tracks["traj_lat"].values).mean())
    lon_nan_ratio = float(np.isnan(tracks["traj_lon"].values).mean())
    if lat_nan_ratio >= 0.999 and lon_nan_ratio >= 0.999:
        raise ValueError(
            f"Track cache is effectively all-NaN (lat_nan_ratio={lat_nan_ratio:.3f}, "
            f"lon_nan_ratio={lon_nan_ratio:.3f}): {track_path}. "
            "Please regenerate track cache with the latest skeleton fix."
        )
    ds = _open_era5(data_cfg["era5_path"])
    ds = _select_region(ds, data_cfg["region"])
    split = data_cfg["split"]
    train_ds = ds.sel(time=slice(split["train_start"], split["train_end"]))
    stats = compute_variable_stats(train_ds, data_cfg["variables"])
    save_stats(data_cfg["norm_stats_path"], stats)


def _sanitize_traj(traj: np.ndarray, default_lat: float, default_lon: float) -> np.ndarray:
    # traj [T,O,2]
    traj = traj.copy()
    t, o, _ = traj.shape
    for obj in range(o):
        for d in range(2):
            arr = traj[:, obj, d]
            valid = np.isfinite(arr)
            if valid.any():
                first = int(np.argmax(valid))
                arr[:first] = float(arr[first])
                for i in range(first + 1, t):
                    if not np.isfinite(arr[i]):
                        arr[i] = arr[i - 1]
            else:
                arr[:] = default_lat if d == 0 else default_lon
            traj[:, obj, d] = arr
    traj[..., 1] = np.mod(traj[..., 1], 360.0)
    return traj


class MiniERA5Dataset(Dataset):
    def __init__(
        self,
        era5_path: str | Path,
        track_path: str | Path,
        variables: list[str],
        split_start: str,
        split_end: str,
        history_steps: int,
        forecast_steps: int,
        stride: int,
        region_cfg: dict[str, Any],
        norm_stats_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.variables = variables
        self.history_steps = int(history_steps)
        self.forecast_steps = int(forecast_steps)
        self.stride = int(stride)

        ds = _open_era5(era5_path)
        ds = _select_region(ds, region_cfg)
        ds = ds.sel(time=slice(split_start, split_end))
        self.ds = ds
        self.lat = ds["latitude"].values.astype(np.float32)
        self.lon = ds["longitude"].values.astype(np.float32)
        self.time = ds["time"].values

        tracks = xr.open_dataset(track_path)
        tracks = _select_region(tracks.sel(time=slice(split_start, split_end)), region_cfg)
        self.tracks = tracks

        self.stats = None
        if norm_stats_path is not None and Path(norm_stats_path).exists():
            self.stats = load_stats(norm_stats_path, variables)

        self.indices = self._build_indices()

    def _build_indices(self) -> np.ndarray:
        n = len(self.time)
        start = self.history_steps - 1
        end = n - self.forecast_steps - 1
        if end < start:
            return np.array([], dtype=np.int64)
        return np.arange(start, end + 1, self.stride, dtype=np.int64)

    def _stack_fields(self, start: int, end: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for var in self.variables:
            arr = _flatten_variable(self.ds[var].isel(time=slice(start, end)))
            if self.stats is not None:
                mean = self.stats[var]["mean"].reshape(1, -1, 1, 1)
                std = self.stats[var]["std"].reshape(1, -1, 1, 1)
                arr = (arr - mean) / (std + 1e-6)
            chunks.append(arr)
        return np.concatenate(chunks, axis=1).astype(np.float32)

    def _traj_window(self, start: int, end: int) -> np.ndarray:
        lat = self.tracks["traj_lat"].isel(time=slice(start, end)).values.astype(np.float32)
        lon = self.tracks["traj_lon"].isel(time=slice(start, end)).values.astype(np.float32)
        traj = np.stack([lat, lon], axis=-1)
        traj = _sanitize_traj(traj, float(np.nanmean(self.lat)), float(np.nanmean(self.lon)))
        return traj

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t = int(self.indices[idx])
        hs = t - self.history_steps + 1
        he = t + 1
        fs = t + 1
        fe = t + self.forecast_steps + 1
        x_hist = self._stack_fields(hs, he)
        y_future = self._stack_fields(fs, fe)
        traj_hist = self._traj_window(hs, he)
        traj_future = self._traj_window(fs, fe)
        return {
            "x_hist": torch.from_numpy(x_hist),
            "y_future": torch.from_numpy(y_future),
            "traj_hist": torch.from_numpy(traj_hist),
            "traj_future": torch.from_numpy(traj_future),
            "lat": torch.from_numpy(self.lat.copy()),
            "lon": torch.from_numpy(self.lon.copy()),
            "time_index": torch.tensor(t, dtype=torch.long),
        }


def build_dataloaders(cfg: dict[str, Any]) -> dict[str, DataLoader]:
    data_cfg = cfg["data"]
    split = data_cfg["split"]
    common = dict(
        era5_path=data_cfg["era5_path"],
        track_path=data_cfg["track_cache_path"],
        variables=data_cfg["variables"],
        history_steps=data_cfg["history_steps"],
        forecast_steps=data_cfg["forecast_steps"],
        stride=data_cfg["stride"],
        region_cfg=data_cfg["region"],
        norm_stats_path=data_cfg["norm_stats_path"],
    )
    ds_train = MiniERA5Dataset(split_start=split["train_start"], split_end=split["train_end"], **common)
    ds_val = MiniERA5Dataset(split_start=split["val_start"], split_end=split["val_end"], **common)
    ds_test = MiniERA5Dataset(split_start=split["test_start"], split_end=split["test_end"], **common)

    dl_cfg = data_cfg["dataloader"]
    batch_size = int(dl_cfg["batch_size"])
    num_workers = int(dl_cfg["num_workers"])
    pin_memory = bool(dl_cfg.get("pin_memory", True)) and torch.cuda.is_available()
    eval_workers = 0 if num_workers == 0 else max(1, num_workers // 2)

    return {
        "train": DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=eval_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=eval_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }
