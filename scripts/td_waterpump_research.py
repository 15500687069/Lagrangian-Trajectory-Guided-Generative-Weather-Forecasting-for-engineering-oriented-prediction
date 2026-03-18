from __future__ import annotations

import argparse
import calendar
import copy
import glob
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
import yaml
from matplotlib import rcParams
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import binary_dilation, label, minimum_filter
from scipy import stats as sstats

try:
    from mpl_toolkits.basemap import Basemap
except Exception:
    Basemap = None


CP = 1005.0
G = 9.81
LV = 2.5e6


DEFAULT_CONFIG: dict[str, Any] = {
    "study": {
        "region": [110.0, 17.0, 113.0, 20.0],  # [lon_min, lat_min, lon_max, lat_max]
        "start_time": "2024-10-28T00:00:00Z",
        "end_time": "2024-10-30T00:00:00Z",
        "gpm_cross_track_lat": 18.8,
        "output_dir": "outputs/td_waterpump_research",
    },
    "paths": {
        "era5_raw_dir": "data/td_case/era5_raw",
        "era5_pressure_file": "data/td_case/era5_pressure.nc",
        "era5_single_file": "data/td_case/era5_single.nc",
        "era5_merged_file": "data/td_case/era5_merged.nc",
        "gpm_dir": "data/td_case/gpm",
        "gpm_glob": "data/td_case/gpm/*.HDF5",
        "selected_gpm_file": "",
    },
    "download": {
        "era5": {
            "enabled": True,
            "overwrite": False,
            "step_hours": 1,
            "pressure_levels": [1000, 925, 850, 700, 500],
            "cds_url": "",
            "cds_key_env": "CDSAPI_KEY",
        },
        "gpm": {
            "enabled": True,
            "provider": "earthaccess",  # earthaccess | cmr
            "short_name": "GPM_2ADPR",
            "version": "07",
            "max_granules": 60,
            "page_size": 200,
            "overwrite": False,
            "earthaccess_strategy": "netrc",
            "earthdata_token_env": "EARTHDATA_TOKEN",
        },
    },
    "analysis": {
        "rain_threshold": 0.1,
        "reflectivity_threshold": 35.0,
        "min_layers": 5,
        "min_depth_km": 1.5,
        "dz_km": 0.125,
        "min_voxels": 20,
        "merge_distance_km": 5.0,
        "mse_window_radius": 1,
        "mse_pressure_level_hpa": 850.0,
        "height_range_km": [1.5, 12.0],
        "point_spacing_km": 5.0,
        "layer_indices": [135, 136, 137, 138, 139, 140],
        "td_detection": {
            "min_wind": 8.5,
            "min_pressure_diff": 3.0,
            "min_rh": 85.0,
            "search_radius_deg": 10.0,
            "psi_radius_deg": 3.0,
        },
    },
    "statistics": {
        "enabled": True,
        "oidra_split_quantile": 0.5,
        "filter": {
            "min_pump_count": 1,
            "min_precip_mean": 0.0,
            "max_mean_mse_var_quantile": 0.99,
            "min_precip_type_coverage": 0.5,
            "require_td_center": True,
        },
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    deep_update(cfg, user_cfg)
    return cfg


def to_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_time_utc(value: str) -> datetime:
    ts = pd.to_datetime(value, utc=True)
    if not isinstance(ts, pd.Timestamp):
        raise ValueError(f"Invalid time string: {value}")
    return ts.to_pydatetime()


def to_isotime_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def month_iter(start: datetime, end: datetime) -> list[tuple[int, int]]:
    y, m = start.year, start.month
    months: list[tuple[int, int]] = []
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return months


def day_list_for_month(y: int, m: int, start: datetime, end: datetime) -> list[str]:
    nday = calendar.monthrange(y, m)[1]
    month_start = datetime(y, m, 1, tzinfo=timezone.utc)
    month_end = datetime(y, m, nday, 23, 59, 59, tzinfo=timezone.utc)
    win_start = max(start, month_start)
    win_end = min(end, month_end)
    if win_start > win_end:
        return []
    return [f"{d:02d}" for d in range(win_start.day, win_end.day + 1)]


def time_list(step_hours: int) -> list[str]:
    if step_hours < 1 or step_hours > 24 or 24 % step_hours != 0:
        raise ValueError("step_hours must divide 24, e.g. 1/2/3/4/6/8/12/24")
    return [f"{h:02d}:00" for h in range(0, 24, step_hours)]


def is_valid_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with xr.open_dataset(path):
            return True
    except Exception:
        return False


def drop_era_aux_vars(ds: xr.Dataset) -> xr.Dataset:
    drop = [v for v in ("number", "expver") if v in ds.data_vars]
    if drop:
        ds = ds.drop_vars(drop)
    return ds


def normalize_era_dataset(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "level" in ds.dims and "pressure_level" not in ds.dims:
        ds = ds.rename({"level": "pressure_level"})
    if "longitude" in ds.coords:
        lon = ds["longitude"]
        if float(np.nanmax(lon.values)) > 180.0:
            lon = ((lon + 180.0) % 360.0) - 180.0
            ds = ds.assign_coords(longitude=lon).sortby("longitude")
    return ds


def get_time_coord_name(ds: xr.Dataset) -> str:
    for name in ("time", "valid_time"):
        if name in ds.coords:
            return name
    raise KeyError("Cannot find time coordinate (time/valid_time)")


def get_lat_coord_name(ds: xr.Dataset) -> str:
    for name in ("latitude", "lat"):
        if name in ds.coords:
            return name
    raise KeyError("Cannot find latitude coordinate")


def get_lon_coord_name(ds: xr.Dataset) -> str:
    for name in ("longitude", "lon"):
        if name in ds.coords:
            return name
    raise KeyError("Cannot find longitude coordinate")


def select_era_level(da: xr.DataArray, level_hpa: float) -> xr.DataArray:
    for level_dim in ("pressure_level", "level", "isobaricInhPa"):
        if level_dim in da.dims:
            return da.sel({level_dim: level_hpa}, method="nearest")
    return da


def pick_first_var(ds: xr.Dataset, names: list[str]) -> xr.DataArray:
    for name in names:
        if name in ds.data_vars:
            return ds[name]
    raise KeyError(f"None of variables found: {names}")


def maybe_get_var(ds: xr.Dataset, names: list[str]) -> xr.DataArray | None:
    for name in names:
        if name in ds.data_vars:
            return ds[name]
    return None


def apply_common_plot_style() -> None:
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
    )


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(max(a, 0.0)))


def nearest_index(value: float, arr: np.ndarray) -> int:
    return int(np.argmin(np.abs(arr - value)))


def region_mask(lat: np.ndarray, lon: np.ndarray, region: list[float]) -> np.ndarray:
    lon_min, lat_min, lon_max, lat_max = region
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)


def basemap_or_none(ax: plt.Axes, region: list[float], resolution: str = "i"):
    if Basemap is None:
        return None
    lon_min, lat_min, lon_max, lat_max = region
    return Basemap(
        projection="cyl",
        llcrnrlon=lon_min,
        urcrnrlon=lon_max,
        llcrnrlat=lat_min,
        urcrnrlat=lat_max,
        resolution=resolution,
        ax=ax,
    )


def project_points(m, lon: np.ndarray, lat: np.ndarray):
    if m is None:
        return lon, lat
    return m(lon, lat)


def draw_map_borders(m) -> None:
    if m is None:
        return
    m.drawcoastlines(linewidth=1.0)
    try:
        m.drawcountries(linewidth=0.8)
    except Exception:
        pass


def plot_swath_boundary(ax: plt.Axes, m, lat: np.ndarray, lon: np.ndarray, region: list[float]) -> None:
    lon_min, lat_min, lon_max, lat_max = region
    max_jump_km = 50.0

    def _plot_edge(edge_lon: np.ndarray, edge_lat: np.ndarray) -> None:
        seg_lon: list[float] = []
        seg_lat: list[float] = []
        for idx in range(len(edge_lon)):
            if idx > 0:
                d = haversine(
                    float(edge_lat[idx - 1]),
                    float(edge_lon[idx - 1]),
                    float(edge_lat[idx]),
                    float(edge_lon[idx]),
                )
                if d > max_jump_km:
                    if len(seg_lon) > 1:
                        x, y = project_points(m, np.asarray(seg_lon), np.asarray(seg_lat))
                        ax.plot(x, y, "k-", lw=1.5, zorder=10)
                    seg_lon, seg_lat = [], []
            if lat_min <= edge_lat[idx] <= lat_max and lon_min <= edge_lon[idx] <= lon_max:
                seg_lon.append(float(edge_lon[idx]))
                seg_lat.append(float(edge_lat[idx]))
        if len(seg_lon) > 1:
            x, y = project_points(m, np.asarray(seg_lon), np.asarray(seg_lat))
            ax.plot(x, y, "k-", lw=1.5, zorder=10)

    _plot_edge(lon[:, 0], lat[:, 0])
    _plot_edge(lon[:, -1], lat[:, -1])


def decode_precip_type(type_raw: np.ndarray | None) -> np.ndarray | None:
    if type_raw is None:
        return None
    ptype = np.zeros_like(type_raw, dtype=np.int32)
    valid = np.isfinite(type_raw) & (type_raw > 0) & (type_raw != -9999) & (type_raw != -1111)
    ptype[valid] = (type_raw[valid] // 10000000).astype(np.int32)
    ptype = np.clip(ptype, 0, 3)
    return ptype


def read_gpm_granule(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        lat = f["FS/Latitude"][:].astype(np.float64)
        lon = f["FS/Longitude"][:].astype(np.float64)
        precip = f["FS/SLV/precipRateNearSurface"][:].astype(np.float64)
        zdbz = f["FS/SLV/zFactorFinal"][:, :, :, 0].astype(np.float64)

        height = None
        if "FS/PRE/height" in f:
            height = f["FS/PRE/height"][:].astype(np.float64) / 1000.0

        type_raw = None
        if "FS/CSF/typePrecip" in f:
            type_raw = f["FS/CSF/typePrecip"][:].astype(np.float64)

        y = f["FS/ScanTime/Year"][:].astype(np.int32)
        m = f["FS/ScanTime/Month"][:].astype(np.int32)
        d = f["FS/ScanTime/DayOfMonth"][:].astype(np.int32)
        hh = f["FS/ScanTime/Hour"][:].astype(np.int32)
        mm = f["FS/ScanTime/Minute"][:].astype(np.int32)
        ss = f["FS/ScanTime/Second"][:].astype(np.int32)

    scan_times = pd.to_datetime(
        {
            "year": y,
            "month": m,
            "day": d,
            "hour": hh,
            "minute": mm,
            "second": ss,
        },
        utc=True,
        errors="coerce",
    )
    return {
        "path": path,
        "lat": lat,
        "lon": lon,
        "precip": precip,
        "zdbz": zdbz,
        "height_km": height,
        "ptype_raw": type_raw,
        "ptype": decode_precip_type(type_raw),
        "scan_times": scan_times.to_numpy(dtype="datetime64[ns]"),
    }


def granule_overpass_time(scan_times: np.ndarray) -> np.datetime64 | None:
    if scan_times.size == 0:
        return None
    ts = pd.to_datetime(scan_times, utc=True, errors="coerce")
    ts = ts[~ts.isna()]
    if len(ts) == 0:
        return None
    mid = ts[len(ts) // 2]
    if isinstance(mid, pd.Timestamp):
        return mid.to_datetime64()
    return np.datetime64(mid)


def nearest_era_time_index(era_ds: xr.Dataset, target_time: np.datetime64 | None) -> int:
    time_name = get_time_coord_name(era_ds)
    era_times = era_ds[time_name].values.astype("datetime64[ns]")
    if target_time is None:
        return int(len(era_times) // 2)
    idx = int(np.argmin(np.abs(era_times - target_time.astype("datetime64[ns]"))))
    return idx


def compute_local_mse_variance(
    era_ds: xr.Dataset,
    time_idx: int,
    lat0: float,
    lon0: float,
    level_hpa: float,
    radius: int,
) -> tuple[float, float]:
    t_var = pick_first_var(era_ds, ["t", "temperature", "t850"])
    z_var = pick_first_var(era_ds, ["z", "geopotential", "z850"])
    q_var = pick_first_var(era_ds, ["q", "specific_humidity", "q850"])

    t_var = select_era_level(t_var, level_hpa)
    z_var = select_era_level(z_var, level_hpa)
    q_var = select_era_level(q_var, level_hpa)

    time_name = get_time_coord_name(era_ds)
    t2 = t_var.isel({time_name: time_idx})
    z2 = z_var.isel({time_name: time_idx})
    q2 = q_var.isel({time_name: time_idx})

    lat_name = get_lat_coord_name(era_ds)
    lon_name = get_lon_coord_name(era_ds)
    lat_vals = era_ds[lat_name].values.astype(np.float64)
    lon_vals = era_ds[lon_name].values.astype(np.float64)

    if np.nanmax(lon_vals) > 180.0:
        lon_vals = ((lon_vals + 180.0) % 360.0) - 180.0
    lon0_use = ((lon0 + 180.0) % 360.0) - 180.0

    i = nearest_index(lat0, lat_vals)
    j = nearest_index(lon0_use, lon_vals)
    i0, i1 = max(0, i - radius), min(len(lat_vals), i + radius + 1)
    j0, j1 = max(0, j - radius), min(len(lon_vals), j + radius + 1)

    t_box = t2.values[i0:i1, j0:j1]
    z_box = z2.values[i0:i1, j0:j1]
    q_box = q2.values[i0:i1, j0:j1]

    z_units = str(z2.attrs.get("units", "")).lower()
    if "m2 s-2" in z_units or "m**2 s**-2" in z_units or "m^2 s^-2" in z_units:
        phi = z_box
    else:
        phi = G * z_box

    mse = CP * t_box + phi + LV * q_box
    return float(np.nanmean(mse)), float(np.nanvar(mse))


def compute_group_oidra(pumps: list[dict[str, Any]]) -> float:
    if len(pumps) < 2:
        return float("nan")
    areas = np.asarray([p["area_voxels"] for p in pumps], dtype=np.float64)
    if not np.isfinite(areas).all() or np.nansum(areas) <= 0:
        areas = np.ones(len(pumps), dtype=np.float64)
    weights = areas / np.nansum(areas)

    score = 0.0
    wsum = 0.0
    for i in range(len(pumps)):
        for j in range(i + 1, len(pumps)):
            d = haversine(pumps[i]["lat"], pumps[i]["lon"], pumps[j]["lat"], pumps[j]["lon"])
            w = float(weights[i] * weights[j])
            score += w / (d + 1e-6)
            wsum += w
    if wsum <= 0:
        return float("nan")
    return score / wsum


def merge_close_pumps(pumps: list[dict[str, Any]], merge_distance_km: float) -> list[dict[str, Any]]:
    if not pumps:
        return []
    merged: list[dict[str, Any]] = []
    for p in sorted(pumps, key=lambda x: x["area_voxels"], reverse=True):
        duplicate = False
        for q in merged:
            if p["rain_id"] == q["rain_id"]:
                d = haversine(p["lat"], p["lon"], q["lat"], q["lon"])
                if d < merge_distance_km:
                    duplicate = True
                    break
        if not duplicate:
            merged.append(p)
    return merged


def compute_precip_structure_metrics(
    precip: np.ndarray,
    ptype: np.ndarray | None,
    mask2d: np.ndarray,
) -> dict[str, float]:
    valid = mask2d & np.isfinite(precip)
    out = {
        "precip_mean": float("nan"),
        "precip_max": float("nan"),
        "precip_sum": float("nan"),
        "rain_area_fraction": float("nan"),
        "heavy_rain_fraction": float("nan"),
        "extreme_rain_fraction": float("nan"),
        "stratiform_fraction": float("nan"),
        "convective_fraction": float("nan"),
        "other_fraction": float("nan"),
    }
    if not np.any(valid):
        return out

    vals = precip[valid]
    out["precip_mean"] = float(np.nanmean(vals))
    out["precip_max"] = float(np.nanmax(vals))
    out["precip_sum"] = float(np.nansum(vals))
    out["rain_area_fraction"] = float(np.nanmean(vals > 0.1))
    out["heavy_rain_fraction"] = float(np.nanmean(vals >= 10.0))
    out["extreme_rain_fraction"] = float(np.nanmean(vals >= 20.0))

    if ptype is not None:
        pt = ptype[valid]
        rain_valid = pt > 0
        if np.any(rain_valid):
            denom = float(np.sum(rain_valid))
            out["stratiform_fraction"] = float(np.sum(pt[rain_valid] == 1) / denom)
            out["convective_fraction"] = float(np.sum(pt[rain_valid] == 2) / denom)
            out["other_fraction"] = float(np.sum(pt[rain_valid] == 3) / denom)
    return out


def analyze_gpm_granule(
    granule: dict[str, Any],
    era_ds: xr.Dataset,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    region = cfg["study"]["region"]
    acfg = cfg["analysis"]

    lat = granule["lat"]
    lon = granule["lon"]
    precip_raw = granule["precip"]
    zdbz_raw = granule["zdbz"]
    ptype = granule["ptype"]

    mask2d = region_mask(lat, lon, region)
    if not np.any(mask2d):
        raise ValueError("Granule does not intersect study region")

    precip = np.where(mask2d, precip_raw, np.nan)
    zdbz = np.where(mask2d[:, :, None], zdbz_raw, np.nan)

    rain_mask = np.nan_to_num(precip, nan=0.0) > float(acfg["rain_threshold"])
    rain_mask = binary_dilation(rain_mask, np.ones((3, 3), dtype=np.uint8))
    rain_labels, _ = label(rain_mask)

    core_mask = np.isfinite(zdbz) & (zdbz >= float(acfg["reflectivity_threshold"]))
    core_labels, n_core = label(core_mask, np.ones((3, 3, 3), dtype=np.uint8))

    overpass = granule_overpass_time(granule["scan_times"])
    era_time_idx = nearest_era_time_index(era_ds, overpass)

    pumps: list[dict[str, Any]] = []
    for lid in range(1, n_core + 1):
        idx = np.where(core_labels == lid)
        area_voxels = int(idx[0].size)
        if area_voxels < int(acfg["min_voxels"]):
            continue

        ii, jj, kk = idx
        if len(np.unique(kk)) < int(acfg["min_layers"]):
            continue

        depth_km = (int(np.max(kk)) - int(np.min(kk))) * float(acfg["dz_km"])
        if depth_km < float(acfg["min_depth_km"]):
            continue

        ci, cj = int(np.median(ii)), int(np.median(jj))
        lat0 = float(lat[ci, cj])
        lon0 = float(lon[ci, cj])

        mse_mean, mse_var = compute_local_mse_variance(
            era_ds=era_ds,
            time_idx=era_time_idx,
            lat0=lat0,
            lon0=lon0,
            level_hpa=float(acfg["mse_pressure_level_hpa"]),
            radius=int(acfg["mse_window_radius"]),
        )

        dists = []
        for i_idx, j_idx in zip(ii, jj):
            d = haversine(float(lat[i_idx, j_idx]), float(lon[i_idx, j_idx]), lat0, lon0)
            if d > 0:
                dists.append(d)
        local_oidra = 0.0 if len(dists) == 0 else float(1.0 / (np.mean(dists) + 1e-6))

        rain_id = int(rain_labels[ci, cj])
        max_dbz = float(np.nanmax(zdbz[ii, jj, kk]))
        pumps.append(
            {
                "id": lid,
                "lat": lat0,
                "lon": lon0,
                "local_oidra": local_oidra,
                "mse_mean": mse_mean,
                "mse_var": mse_var,
                "rain_id": rain_id,
                "area_voxels": area_voxels,
                "depth_km": float(depth_km),
                "max_dbz": max_dbz,
            }
        )

    pumps = merge_close_pumps(pumps, float(acfg["merge_distance_km"]))
    group_oidra = compute_group_oidra(pumps)
    precip_metrics = compute_precip_structure_metrics(precip=precip_raw, ptype=ptype, mask2d=mask2d)

    local_oidra_vals = np.asarray([p["local_oidra"] for p in pumps], dtype=np.float64)
    mse_var_vals = np.asarray([p["mse_var"] for p in pumps], dtype=np.float64)

    result = {
        "path": str(granule["path"]),
        "overpass_time": overpass,
        "era_time_idx": era_time_idx,
        "pump_count": int(len(pumps)),
        "group_oidra": float(group_oidra),
        "mean_local_oidra": float(np.nanmean(local_oidra_vals)) if local_oidra_vals.size else float("nan"),
        "mean_mse_var": float(np.nanmean(mse_var_vals)) if mse_var_vals.size else float("nan"),
        "median_mse_var": float(np.nanmedian(mse_var_vals)) if mse_var_vals.size else float("nan"),
        "pumps": pumps,
        "mask2d": mask2d,
        "precip_region": precip,
        "zdbz_region": zdbz,
        "precip_metrics": precip_metrics,
        "granule": granule,
    }
    return result


def find_local_minima(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, min_dist_deg: float = 3.0) -> list[tuple[float, float]]:
    finite_mask = np.isfinite(field)
    if not np.any(finite_mask):
        return []

    field_safe = np.where(finite_mask, field, np.nanmax(field[finite_mask]) + 1.0)
    local_min = field_safe == minimum_filter(field_safe, size=3, mode="nearest")
    cand_i, cand_j = np.where(local_min & finite_mask)
    points = [(float(lat[i]), float(lon[j]), float(field_safe[i, j])) for i, j in zip(cand_i, cand_j)]
    points.sort(key=lambda x: x[2])

    selected: list[tuple[float, float]] = []
    for la, lo, _ in points:
        keep = True
        for sla, slo in selected:
            if np.hypot(la - sla, lo - slo) < min_dist_deg:
                keep = False
                break
        if keep:
            selected.append((la, lo))
    return selected


def calculate_streamfunction(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    r_earth = 6371000.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    if len(lat_rad) < 2 or len(lon_rad) < 2:
        return np.zeros_like(u, dtype=np.float64)

    dlat = np.gradient(lat_rad)
    dlon = np.gradient(lon_rad)
    psi = np.zeros_like(u, dtype=np.float64)

    for i in range(1, len(lat)):
        psi[i, :] = psi[i - 1, :] - u[i - 1, :] * r_earth * dlat[i - 1]
    for j in range(1, len(lon)):
        psi[:, j] = psi[:, j - 1] + v[:, j - 1] * r_earth * np.cos(lat_rad) * dlon[j - 1]

    psi -= np.nanmean(psi)
    return psi


def _distance_deg_grid(lat: np.ndarray, lon: np.ndarray, center_lat: float, center_lon: float) -> np.ndarray:
    return np.sqrt((lat[:, None] - center_lat) ** 2 + (lon[None, :] - center_lon) ** 2)


def detect_tropical_depression_at_time(
    era_ds: xr.Dataset,
    time_idx: int,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    td_cfg = cfg["analysis"]["td_detection"]
    search_radius = float(td_cfg["search_radius_deg"])
    psi_radius = float(td_cfg["psi_radius_deg"])

    lat_name = get_lat_coord_name(era_ds)
    lon_name = get_lon_coord_name(era_ds)
    lat = era_ds[lat_name].values.astype(np.float64)
    lon = era_ds[lon_name].values.astype(np.float64)
    time_name = get_time_coord_name(era_ds)
    time_val = era_ds[time_name].values[time_idx]

    msl = pick_first_var(era_ds, ["msl", "mean_sea_level_pressure"]).isel({time_name: time_idx}).values.astype(np.float64)
    if np.nanmean(msl) > 2000.0:
        msl_hpa = msl / 100.0
    else:
        msl_hpa = msl

    u_var = pick_first_var(era_ds, ["u", "u_component_of_wind", "u850"])
    v_var = pick_first_var(era_ds, ["v", "v_component_of_wind", "v850"])
    rh_var = maybe_get_var(era_ds, ["r", "relative_humidity", "r850"])

    u1000 = select_era_level(u_var, 1000).isel({time_name: time_idx}).values.astype(np.float64)
    v1000 = select_era_level(v_var, 1000).isel({time_name: time_idx}).values.astype(np.float64)
    u850 = select_era_level(u_var, 850).isel({time_name: time_idx}).values.astype(np.float64)
    v850 = select_era_level(v_var, 850).isel({time_name: time_idx}).values.astype(np.float64)
    rh850 = None
    if rh_var is not None:
        rh850 = select_era_level(rh_var, 850).isel({time_name: time_idx}).values.astype(np.float64)

    psi = calculate_streamfunction(u850, v850, lat, lon)
    minima = find_local_minima(msl_hpa, lat, lon, min_dist_deg=3.0)
    if not minima:
        return None

    best: dict[str, Any] | None = None
    for center_lat, center_lon in minima:
        dist = _distance_deg_grid(lat, lon, center_lat, center_lon)
        inner_mask = dist <= 3.0
        outer_mask = (dist >= 8.0) & (dist <= 10.0)

        if not np.any(inner_mask) or not np.any(outer_mask):
            continue

        wind = np.sqrt(u1000**2 + v1000**2)
        max_wind = float(np.nanmax(np.where(inner_mask, wind, np.nan)))
        pressure_diff = float(np.nanmean(msl_hpa[outer_mask]) - np.nanmin(msl_hpa[inner_mask]))
        if max_wind < float(td_cfg["min_wind"]):
            continue
        if pressure_diff < float(td_cfg["min_pressure_diff"]):
            continue

        lat_mask = (lat >= center_lat - search_radius) & (lat <= center_lat + search_radius)
        lon_mask = (lon >= center_lon - search_radius) & (lon <= center_lon + search_radius)
        if not np.any(lat_mask) or not np.any(lon_mask):
            continue

        psi_region = psi[np.ix_(lat_mask, lon_mask)]
        pressure_region = msl_hpa[np.ix_(lat_mask, lon_mask)]
        rh_region = rh850[np.ix_(lat_mask, lon_mask)] if rh850 is not None else None
        lat_reg = lat[lat_mask]
        lon_reg = lon[lon_mask]

        psi_min_idx = np.unravel_index(np.nanargmin(psi_region), psi_region.shape)
        psi_min_lat = float(lat_reg[psi_min_idx[0]])
        psi_min_lon = float(lon_reg[psi_min_idx[1]])
        if np.hypot(center_lat - psi_min_lat, center_lon - psi_min_lon) > psi_radius:
            continue

        center_rh = float("nan")
        if rh_region is not None:
            i0 = nearest_index(center_lat, lat_reg)
            j0 = nearest_index(center_lon, lon_reg)
            center_rh = float(rh_region[i0, j0])
            if not np.any(rh_region >= float(td_cfg["min_rh"])):
                continue

        psi_min = float(np.nanmin(psi_region))
        contour_thresh = psi_min * 1.1 if psi_min < 0 else psi_min * 0.9
        contour_mask = psi_region <= contour_thresh if psi_min < 0 else psi_region >= contour_thresh

        candidate = {
            "time": time_val,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "max_wind": max_wind,
            "pressure_diff": pressure_diff,
            "center_rh": center_rh,
            "psi_min_lat": psi_min_lat,
            "psi_min_lon": psi_min_lon,
            "psi_min_value": psi_min,
            "region_lat": lat_reg,
            "region_lon": lon_reg,
            "pressure_field": pressure_region,
            "psi_field": psi_region,
            "rh_field": rh_region,
            "contour_mask": contour_mask,
        }

        if best is None:
            best = candidate
            continue

        best_metric = (best["pressure_diff"], best["max_wind"])
        cur_metric = (candidate["pressure_diff"], candidate["max_wind"])
        if cur_metric > best_metric:
            best = candidate

    return best


def create_complete_ticks(min_val: float, max_val: float) -> np.ndarray:
    ticks: list[float] = [float(min_val)]
    span = max_val - min_val
    if span <= 10:
        step = 2
    elif span <= 20:
        step = 5
    else:
        step = 10

    current = np.ceil(min_val / step) * step
    while current < max_val:
        if current > min_val:
            ticks.append(float(current))
        current += step
    if max_val not in ticks:
        ticks.append(float(max_val))
    return np.asarray(sorted(set(ticks)), dtype=np.float64)


def setup_plot_common(region_lat: np.ndarray, region_lon: np.ndarray, title: str):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
        }
    )

    min_lon, max_lon = float(np.min(region_lon)), float(np.max(region_lon))
    min_lat, max_lat = float(np.min(region_lat)), float(np.max(region_lat))
    fig, ax = plt.subplots(figsize=(12, 9))
    m = basemap_or_none(ax, [min_lon, min_lat, max_lon, max_lat], resolution="i")
    draw_map_borders(m)

    lon_grid, lat_grid = np.meshgrid(region_lon, region_lat)
    x, y = project_points(m, lon_grid, lat_grid)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_xticks(create_complete_ticks(min_lon, max_lon))
    ax.set_yticks(create_complete_ticks(min_lat, max_lat))
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.grid(True, linestyle="--", alpha=0.25)
    return fig, ax, m, x, y


def plot_pressure_field(dep: dict[str, Any], save_path: Path) -> None:
    time_str = pd.to_datetime(dep["time"]).strftime("%Y-%m-%d %H:%M UTC")
    fig, ax, m, x, y = setup_plot_common(dep["region_lat"], dep["region_lon"], f"Sea Level Pressure at {time_str}")
    pressure = dep["pressure_field"]
    levels = np.linspace(np.nanmin(pressure), np.nanmax(pressure), 18)
    cf = ax.contourf(x, y, pressure, levels=levels, cmap="coolwarm", extend="both")
    cx, cy = project_points(m, np.asarray([dep["center_lon"]]), np.asarray([dep["center_lat"]]))
    ax.plot(cx, cy, "mo", markersize=10, markeredgewidth=1.2, markeredgecolor="k", label="Center")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Pressure (hPa)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_streamfunction_field(dep: dict[str, Any], save_path: Path) -> None:
    time_str = pd.to_datetime(dep["time"]).strftime("%Y-%m-%d %H:%M UTC")
    fig, ax, m, x, y = setup_plot_common(dep["region_lat"], dep["region_lon"], f"850hPa Streamfunction at {time_str}")
    psi_plot = dep["psi_field"] * 1e-6
    max_val = float(np.nanmax(np.abs(psi_plot)))
    levels = np.linspace(-max_val, max_val, 21)
    cf = ax.contourf(x, y, psi_plot, levels=levels, cmap="coolwarm", extend="both")
    ax.contour(x, y, dep["contour_mask"].astype(float), levels=[0.5], colors="blue", linewidths=1.8)

    cx, cy = project_points(m, np.asarray([dep["center_lon"]]), np.asarray([dep["center_lat"]]))
    px, py = project_points(m, np.asarray([dep["psi_min_lon"]]), np.asarray([dep["psi_min_lat"]]))
    ax.plot(cx, cy, "mo", markersize=10, markeredgewidth=1.2, markeredgecolor="k", label="Pressure Center")
    ax.plot(px, py, "x", color="#FFFF00", markersize=11, markeredgewidth=2, label="Psi Min")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Streamfunction ($10^6 m^2 s^{-1}$)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_relative_humidity_field(dep: dict[str, Any], save_path: Path) -> None:
    if dep["rh_field"] is None:
        return
    time_str = pd.to_datetime(dep["time"]).strftime("%Y-%m-%d %H:%M UTC")
    fig, ax, m, x, y = setup_plot_common(dep["region_lat"], dep["region_lon"], f"850hPa Relative Humidity at {time_str}")
    rh = dep["rh_field"]
    min_rh = max(60.0, float(np.floor(np.nanmin(rh) / 10.0) * 10.0))
    levels = np.linspace(min_rh, 100.0, 17)
    cf = ax.contourf(x, y, rh, levels=levels, cmap="coolwarm", extend="both")
    ax.contour(x, y, rh, levels=[85.0], colors="green", linewidths=1.8, linestyles="--")
    cx, cy = project_points(m, np.asarray([dep["center_lon"]]), np.asarray([dep["center_lat"]]))
    ax.plot(cx, cy, "mo", markersize=10, markeredgewidth=1.2, markeredgecolor="k", label="Center")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Relative Humidity (%)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_surface_precip_and_type(result: dict[str, Any], cfg: dict[str, Any], save_path: Path) -> None:
    region = cfg["study"]["region"]
    granule = result["granule"]
    lat = granule["lat"]
    lon = granule["lon"]
    precip = granule["precip"]
    ptype = granule["ptype"]
    mask2d = result["mask2d"]

    pr_bounds = [0, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50]
    pr_colors = ["w", "#B3D9FF", "#66CCFF", "#3399FF", "#33CCCC", "#FFD700", "#FFB347", "#FF4500", "#954F97"]
    pr_cmap = ListedColormap(pr_colors)
    pr_norm = BoundaryNorm(pr_bounds, pr_cmap.N)
    type_cmap = ListedColormap(["#FFFFFF", "#3366FF", "#FF4500", "orange"])
    type_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], type_cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    titles = ["(A) Near-surface Precipitation Rate", "(B) Precipitation Type"]
    datasets = [precip, ptype if ptype is not None else np.zeros_like(precip)]
    cmaps = [pr_cmap, type_cmap]
    norms = [pr_norm, type_norm]

    for ax, data, cmap, norm, title in zip(axes, datasets, cmaps, norms, titles):
        m = basemap_or_none(ax, region, resolution="i")
        draw_map_borders(m)
        valid = mask2d & np.isfinite(data)
        x, y = project_points(m, lon[valid], lat[valid])
        sc = ax.scatter(x, y, c=data[valid], cmap=cmap, norm=norm, s=30, marker="s", edgecolors="none")
        plot_swath_boundary(ax, m, lat, lon, region)
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
        ax.set_xticks(np.arange(region[0], region[2] + 0.1, 1))
        ax.set_yticks(np.arange(region[1], region[3] + 0.1, 1))
        ax.tick_params(axis="both", direction="inout", length=6, width=1.2)
        ax.set_xlabel("Longitude (degE)")
        ax.set_ylabel("Latitude (degN)")
        ax.set_title(title, fontweight="bold")
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        if "Type" in title:
            cb.set_ticks([0, 1, 2, 3])
            cb.set_ticklabels(["No rain", "Stratiform", "Convective", "Other"])
        else:
            cb.set_label("mm h$^{-1}$")

    plt.subplots_adjust(left=0.06, right=0.94, bottom=0.12, top=0.88, wspace=0.1)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_cross_track_profile(result: dict[str, Any], cfg: dict[str, Any], save_path: Path) -> None:
    region = cfg["study"]["region"]
    acfg = cfg["analysis"]
    lat_target = float(cfg["study"]["gpm_cross_track_lat"])
    granule = result["granule"]
    lat = granule["lat"]
    lon = granule["lon"]
    precip = granule["precip"]
    zdbz = granule["zdbz"]
    height = granule["height_km"]
    point_spacing_km = float(acfg["point_spacing_km"])
    hmin, hmax = map(float, acfg["height_range_km"])

    scan_idx = int(np.argmin(np.abs(np.nanmean(lat, axis=1) - lat_target)))
    z_sec = zdbz[scan_idx][::-1]
    if height is not None:
        h_sec = height[scan_idx][::-1]
        yvals = np.nanmean(h_sec, axis=0)
    else:
        yvals = np.arange(z_sec.shape[1], dtype=np.float64) * float(acfg["dz_km"])

    keep = (yvals >= hmin) & (yvals <= hmax)
    yvals = yvals[keep]
    z_sec = z_sec[:, keep]
    if z_sec.size == 0 or yvals.size == 0:
        return
    distance_km = np.arange(z_sec.shape[0], dtype=np.float64) * point_spacing_km

    z_bounds = [15, 20, 25, 30, 35, 40, 45, 50, 55]
    z_colors = ["w", "#3b528b", "#21918c", "#5dc962", "#fde725", "#ffcc33", "#ff9900", "#CC0000"]
    z_cmap = ListedColormap(z_colors)
    z_norm = BoundaryNorm(z_bounds, z_cmap.N)

    pr_bounds = [0, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50]
    pr_colors = ["w", "#B3D9FF", "#66CCFF", "#3399FF", "#33CCCC", "#FFD700", "#FFB347", "#FF4500", "#954F97"]
    pr_cmap = ListedColormap(pr_colors)
    pr_norm = BoundaryNorm(pr_bounds, pr_cmap.N)

    fig = plt.figure(figsize=(16, 7))
    ax1 = plt.subplot(1, 2, 1)
    m = basemap_or_none(ax1, region, resolution="i")
    draw_map_borders(m)
    mask2d = result["mask2d"] & np.isfinite(precip)
    x, y = project_points(m, lon[mask2d], lat[mask2d])
    sc = ax1.scatter(x, y, c=precip[mask2d], cmap=pr_cmap, norm=pr_norm, s=30, marker="s", edgecolors="none")
    xline, yline = project_points(m, lon[scan_idx], lat[scan_idx])
    ax1.plot(xline, yline, "k-", lw=2)
    ax1.text(xline[-1] - 0.2, yline[-1], "A", fontsize=20, fontweight="bold")
    ax1.text(xline[0] + 0.15, yline[0], "B", fontsize=20, fontweight="bold")
    cb1 = plt.colorbar(sc, ax=ax1, pad=0.02)
    cb1.set_label("mm h$^{-1}$", fontsize=20)
    ax1.set_xlim(region[0], region[2])
    ax1.set_ylim(region[1], region[3])
    ax1.set_xticks(np.arange(region[0], region[2] + 0.1, 1))
    ax1.set_yticks(np.arange(region[1], region[3] + 0.1, 1))
    ax1.tick_params(axis="both", which="major", direction="out", length=6, width=1.2, labelsize=20, pad=6)
    ax1.set_title("(a) Near-surface precipitation", fontsize=22, fontweight="bold")
    plot_swath_boundary(ax1, m, lat, lon, region)

    ax2 = plt.subplot(1, 2, 2)
    pcm = ax2.pcolormesh(distance_km, yvals, z_sec.T, cmap=z_cmap, norm=z_norm, shading="nearest")
    ax2.set_ylim(hmin, hmax)
    ax2.set_xlabel("Along-track distance (km)")
    ax2.set_ylabel("Height (km)")
    ax2.set_title("(b) Cross-track reflectivity profile", fontsize=22, fontweight="bold")
    xticks = ax2.get_xticks()
    xticks = xticks[(xticks > 0) & (xticks < distance_km[-1])]
    xticks = np.concatenate(([0], xticks, [distance_km[-1]]))
    labels = [f"{int(x)}" for x in xticks]
    labels[0] = "A"
    labels[-1] = "B"
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(labels)
    cb2 = plt.colorbar(pcm, ax=ax2, pad=0.02)
    cb2.set_label("dBZ", fontsize=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_gpm_multilayer_raw(result: dict[str, Any], cfg: dict[str, Any], save_path_3d: Path, save_path_2d: Path) -> None:
    region = cfg["study"]["region"]
    acfg = cfg["analysis"]
    layer_indices = [int(v) for v in acfg["layer_indices"]]
    granule = result["granule"]
    lat = granule["lat"]
    lon = granule["lon"]
    zdbz = granule["zdbz"]
    region_mask2d = result["mask2d"]

    bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    colors = [
        "#E6F3FF",
        "#ADD8E6",
        "#87CEEB",
        "#3366FF",
        "#0000FF",
        "#000099",
        "#00CC00",
        "#FF9900",
        "#FF0000",
        "#CC0000",
        "#954F97",
    ]
    cmap = ListedColormap(colors[: len(bounds)])
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection="3d")
    valid_layers: list[int] = []
    sc = None
    for i, idx in enumerate(layer_indices):
        if idx < 0 or idx >= zdbz.shape[2]:
            continue
        z_layer = zdbz[:, :, idx].copy()
        z_layer[z_layer < 0] = np.nan
        mask = region_mask2d & np.isfinite(z_layer)
        if not np.any(mask):
            continue
        sc = ax.scatter(
            lon[mask],
            lat[mask],
            np.full(np.sum(mask), i + 1, dtype=np.float64),
            c=z_layer[mask],
            cmap=cmap,
            norm=norm,
            s=14,
            alpha=0.85,
            marker="s",
            edgecolors="none",
        )
        valid_layers.append(idx)

    ax.set_xlim(region[0], region[2])
    ax.set_ylim(region[1], region[3])
    ax.set_zlim(0.5, max(1.5, len(valid_layers) + 0.5))
    ax.set_xticks(np.arange(region[0], region[2] + 0.1, 1))
    ax.set_yticks(np.arange(region[1], region[3] + 0.1, 1))
    ax.set_zticks(np.arange(1, len(valid_layers) + 1))
    ax.set_zticklabels([f"Bin {v}" for v in valid_layers])
    ax.set_xlabel("Longitude (degE)", labelpad=10)
    ax.set_ylabel("Latitude (degN)", labelpad=10)
    ax.set_title("3D Structure of GPM DPR Reflectivity", fontweight="bold", pad=22)
    ax.view_init(elev=20, azim=38)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.1, fraction=0.02)
        cbar.set_label("dBZ")
    fig.savefig(save_path_3d, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    panel_labels = ["A", "B", "C", "D", "E", "F"]
    sc2 = None
    for i, ax in enumerate(axes):
        if i >= len(layer_indices):
            ax.axis("off")
            continue
        idx = layer_indices[i]
        if idx < 0 or idx >= zdbz.shape[2]:
            ax.axis("off")
            continue
        m = basemap_or_none(ax, region, resolution="i")
        draw_map_borders(m)
        z_layer = zdbz[:, :, idx].copy()
        z_layer[z_layer < 0] = np.nan
        mask = region_mask2d & np.isfinite(z_layer)
        if np.any(mask):
            x, y = project_points(m, lon[mask], lat[mask])
            sc2 = ax.scatter(x, y, c=z_layer[mask], cmap=cmap, norm=norm, s=20, marker="s", edgecolors="none")
        plot_swath_boundary(ax, m, lat, lon, region)
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
        ax.set_xticks(np.arange(region[0], region[2] + 0.1, 1))
        ax.set_yticks(np.arange(region[1], region[3] + 0.1, 1))
        ax.tick_params(direction="inout", length=6, width=1.2)
        ax.set_xlabel("Longitude (degE)")
        ax.set_ylabel("Latitude (degN)")
        ax.set_title(f"({panel_labels[i]}) Layer {idx}", fontweight="bold", fontsize=22, loc="left")

    if sc2 is not None:
        cax = fig.add_axes([0.91, 0.18, 0.018, 0.64])
        cb = fig.colorbar(sc2, cax=cax)
        cb.set_label("dBZ")
        cb.set_ticks(bounds)
    fig.subplots_adjust(left=0.06, right=0.9, hspace=0.3)
    fig.savefig(save_path_2d, dpi=300)
    plt.close(fig)


def plot_oidra_mse_map(result: dict[str, Any], cfg: dict[str, Any], save_path: Path) -> None:
    region = cfg["study"]["region"]
    lat = result["granule"]["lat"]
    lon = result["granule"]["lon"]
    precip = result["granule"]["precip"]
    pumps = result["pumps"]
    mask2d = result["mask2d"]

    pr_bounds = [0, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50]
    pr_colors = ["w", "#B3D9FF", "#66CCFF", "#3399FF", "#33CCCC", "#FFD700", "#FFB347", "#FF4500", "#954F97"]
    pr_cmap = ListedColormap(pr_colors)
    pr_norm = BoundaryNorm(pr_bounds, pr_cmap.N)

    fig, ax = plt.subplots(figsize=(12, 8))
    m = basemap_or_none(ax, region, resolution="i")
    draw_map_borders(m)

    valid = mask2d & np.isfinite(precip)
    x, y = project_points(m, lon[valid], lat[valid])
    sc_precip = ax.scatter(x, y, c=precip[valid], cmap=pr_cmap, norm=pr_norm, s=30, marker="s", alpha=0.6, edgecolors="none")

    if len(pumps) > 0:
        oid_vals = np.asarray([p["local_oidra"] for p in pumps], dtype=np.float64)
        mse_vals = np.asarray([p["mse_var"] for p in pumps], dtype=np.float64)
        size = 60.0 + 220.0 * (oid_vals - np.nanmin(oid_vals)) / (np.nanptp(oid_vals) + 1e-6)
        px = np.asarray([p["lon"] for p in pumps], dtype=np.float64)
        py = np.asarray([p["lat"] for p in pumps], dtype=np.float64)
        px, py = project_points(m, px, py)
        sc = ax.scatter(px, py, s=size, c=mse_vals, cmap="bwr", edgecolors="k", alpha=0.85, zorder=5)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Var(MSE) (J$^2$ kg$^{-2}$)", fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        cbar.formatter = FuncFormatter(lambda x, pos: f"{x / 1e6:.2f}e6")
        cbar.update_ticks()

    cb2 = fig.colorbar(sc_precip, ax=ax, pad=0.12, label="mm h$^{-1}$")
    cb2.ax.tick_params(labelsize=20)
    ax.set_xticks(np.arange(region[0], region[2] + 0.1, 1))
    ax.set_yticks(np.arange(region[1], region[3] + 0.1, 1))
    ax.set_xlabel("Longitude (degE)")
    ax.set_ylabel("Latitude (degN)")
    ax.tick_params(direction="out", length=6, width=1)
    ax.set_title("Water Pumps: OIDRA MSE and Variance", fontsize=22, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    plot_swath_boundary(ax, m, lat, lon, region)
    fig.subplots_adjust(right=0.92)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_oidra_vs_precip(df: pd.DataFrame, save_path: Path) -> None:
    if df.empty:
        return
    keep = df["group_oidra"].notna() & df["precip_mean"].notna()
    if keep.sum() < 2:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        df.loc[keep, "group_oidra"],
        df.loc[keep, "precip_mean"],
        c=df.loc[keep, "mean_mse_var"],
        cmap="bwr",
        s=95,
        edgecolors="k",
        alpha=0.9,
    )
    ax.set_xlabel("Group OIDRA")
    ax.set_ylabel("Mean Precipitation (mm h$^{-1}$)")
    ax.set_title("Environment-Organization-Precipitation Relation")
    ax.grid(True, linestyle="--", alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Mean Var(MSE)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _normalize_cds_key(key: str | None) -> str | None:
    if not key:
        return None
    key = key.strip()
    if ":" in key:
        left, right = key.split(":", 1)
        if left.isdigit() and right:
            return right.strip()
    return key


def _cds_retrieve_with_fallback(client, dataset: str, request: dict[str, Any], target: Path) -> None:
    ensure_parent(target)
    variants: list[dict[str, Any]] = []
    req1 = dict(request)
    req1["data_format"] = "netcdf"
    req1["download_format"] = "unarchived"
    variants.append(req1)
    req2 = dict(request)
    req2["format"] = "netcdf"
    variants.append(req2)

    last_exc: Exception | None = None
    for req in variants:
        try:
            if target.exists():
                target.unlink()
            client.retrieve(dataset, req).download(str(target))
            if is_valid_netcdf(target):
                return
        except Exception as exc:  # pragma: no cover - network/auth runtime
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to retrieve {dataset} -> {target}")


def ensure_era5_data(cfg: dict[str, Any]) -> xr.Dataset:
    paths = cfg["paths"]
    dcfg = cfg["download"]["era5"]
    start = parse_time_utc(cfg["study"]["start_time"])
    end = parse_time_utc(cfg["study"]["end_time"])

    merged_path = to_path(paths["era5_merged_file"])
    pressure_path = to_path(paths["era5_pressure_file"])
    single_path = to_path(paths["era5_single_file"])
    raw_dir = to_path(paths["era5_raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    if is_valid_netcdf(merged_path):
        ds = xr.open_dataset(merged_path)
        return normalize_era_dataset(drop_era_aux_vars(ds))

    if not bool(dcfg["enabled"]):
        if is_valid_netcdf(pressure_path) and is_valid_netcdf(single_path):
            dp = normalize_era_dataset(drop_era_aux_vars(xr.open_dataset(pressure_path)))
            ds = normalize_era_dataset(drop_era_aux_vars(xr.open_dataset(single_path)))
            merged = xr.merge([dp, ds], compat="override")
            ensure_parent(merged_path)
            merged.to_netcdf(merged_path)
            return normalize_era_dataset(drop_era_aux_vars(merged))
        raise FileNotFoundError("ERA5 download disabled and merged/pressure/single files are missing.")

    import cdsapi

    cds_url = str(dcfg.get("cds_url", "")).strip() or None
    cds_key = _normalize_cds_key(os.getenv(str(dcfg.get("cds_key_env", "CDSAPI_KEY")), "").strip())
    client_kwargs: dict[str, Any] = {}
    if cds_url:
        client_kwargs["url"] = cds_url
    if cds_key:
        client_kwargs["key"] = cds_key
    client = cdsapi.Client(**client_kwargs)

    area = [cfg["study"]["region"][3], cfg["study"]["region"][0], cfg["study"]["region"][1], cfg["study"]["region"][2]]
    times = time_list(int(dcfg["step_hours"]))
    levels = [str(int(v)) for v in dcfg["pressure_levels"]]

    pressure_parts: list[Path] = []
    single_parts: list[Path] = []
    for y, m in month_iter(start, end):
        days = day_list_for_month(y, m, start, end)
        if not days:
            continue
        tag = f"{y:04d}{m:02d}"
        pressure_file = raw_dir / f"pressure_{tag}.nc"
        single_file = raw_dir / f"single_{tag}.nc"
        pressure_parts.append(pressure_file)
        single_parts.append(single_file)

        if bool(dcfg["overwrite"]) or not is_valid_netcdf(pressure_file):
            req = {
                "product_type": "reanalysis",
                "variable": [
                    "geopotential",
                    "temperature",
                    "specific_humidity",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "relative_humidity",
                ],
                "pressure_level": levels,
                "year": f"{y:04d}",
                "month": f"{m:02d}",
                "day": days,
                "time": times,
                "area": area,
            }
            _cds_retrieve_with_fallback(client, "reanalysis-era5-pressure-levels", req, pressure_file)

        if bool(dcfg["overwrite"]) or not is_valid_netcdf(single_file):
            req = {
                "product_type": "reanalysis",
                "variable": ["mean_sea_level_pressure"],
                "year": f"{y:04d}",
                "month": f"{m:02d}",
                "day": days,
                "time": times,
                "area": area,
            }
            _cds_retrieve_with_fallback(client, "reanalysis-era5-single-levels", req, single_file)

    if not pressure_parts or not single_parts:
        raise RuntimeError("No ERA5 files downloaded. Check study time range and request settings.")

    dp = normalize_era_dataset(drop_era_aux_vars(xr.open_mfdataset([str(p) for p in pressure_parts], combine="by_coords")))
    ds = normalize_era_dataset(drop_era_aux_vars(xr.open_mfdataset([str(p) for p in single_parts], combine="by_coords")))
    ensure_parent(pressure_path)
    ensure_parent(single_path)
    dp.to_netcdf(pressure_path)
    ds.to_netcdf(single_path)

    merged = xr.merge([dp, ds], compat="override")
    for v in merged.data_vars:
        merged[v] = merged[v].astype(np.float32)
    ensure_parent(merged_path)
    merged.to_netcdf(merged_path)
    return normalize_era_dataset(drop_era_aux_vars(merged))


def _download_http_file(url: str, target: Path, bearer_token: str | None = None) -> bool:
    headers = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    with requests.get(url, stream=True, headers=headers, timeout=120) as resp:
        if resp.status_code >= 400:
            return False
        ensure_parent(target)
        with target.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return target.exists() and target.stat().st_size > 0


def _cmr_search_granule_links(cfg: dict[str, Any]) -> list[str]:
    gcfg = cfg["download"]["gpm"]
    region = cfg["study"]["region"]
    start = parse_time_utc(cfg["study"]["start_time"])
    end = parse_time_utc(cfg["study"]["end_time"])

    params = {
        "short_name": gcfg["short_name"],
        "version": gcfg["version"],
        "temporal": f"{to_isotime_z(start)},{to_isotime_z(end)}",
        "bounding_box": f"{region[0]},{region[1]},{region[2]},{region[3]}",
        "page_size": int(gcfg["page_size"]),
        "page_num": 1,
    }
    links: list[str] = []
    max_granules = int(gcfg["max_granules"])
    while len(links) < max_granules:
        resp = requests.get(
            "https://cmr.earthdata.nasa.gov/search/granules.json",
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        feed = resp.json().get("feed", {})
        entries = feed.get("entry", [])
        if not entries:
            break
        for entry in entries:
            for link in entry.get("links", []):
                href = link.get("href", "")
                if not href:
                    continue
                if "opendap" in href.lower():
                    continue
                if href.lower().endswith((".h5", ".hdf5")):
                    links.append(href)
                    break
            if len(links) >= max_granules:
                break
        params["page_num"] += 1
    return links[:max_granules]


def _download_gpm_via_cmr(cfg: dict[str, Any], gpm_dir: Path) -> list[Path]:
    gcfg = cfg["download"]["gpm"]
    links = _cmr_search_granule_links(cfg)
    token = os.getenv(str(gcfg.get("earthdata_token_env", "EARTHDATA_TOKEN")), "").strip() or None
    out: list[Path] = []
    for url in links:
        name = Path(urlparse(url).path).name
        target = gpm_dir / name
        if target.exists() and target.stat().st_size > 0 and not bool(gcfg["overwrite"]):
            out.append(target)
            continue
        ok = _download_http_file(url, target, bearer_token=token)
        if ok:
            out.append(target)
    return sorted(out)


def _download_gpm_via_earthaccess(cfg: dict[str, Any], gpm_dir: Path) -> list[Path]:
    import earthaccess

    gcfg = cfg["download"]["gpm"]
    strategy = str(gcfg.get("earthaccess_strategy", "netrc"))
    token = os.getenv(str(gcfg.get("earthdata_token_env", "EARTHDATA_TOKEN")), "").strip()
    logged_in = False

    if token:
        attempts = [
            {"strategy": "token", "token": token, "persist": True},
            {"token": token, "persist": True},
        ]
        for kwargs in attempts:
            try:
                earthaccess.login(**kwargs)
                logged_in = True
                break
            except Exception:
                pass
    if not logged_in:
        earthaccess.login(strategy=strategy, persist=True)

    region = cfg["study"]["region"]
    start = parse_time_utc(cfg["study"]["start_time"])
    end = parse_time_utc(cfg["study"]["end_time"])

    short_names = [str(gcfg["short_name"]), "GPM_2ADPR", "2A-DPR"]
    short_names = list(dict.fromkeys(short_names))
    results = []
    for short_name in short_names:
        kwargs: dict[str, Any] = {
            "short_name": short_name,
            "temporal": (to_isotime_z(start), to_isotime_z(end)),
            "bounding_box": (region[0], region[1], region[2], region[3]),
            "count": int(gcfg["max_granules"]),
        }
        version = str(gcfg.get("version", "")).strip()
        if version:
            kwargs["version"] = version
        results = earthaccess.search_data(**kwargs)
        if results:
            break

    if not results:
        return []

    results = results[: int(gcfg["max_granules"])]
    downloaded = earthaccess.download(results, str(gpm_dir))
    if downloaded is None:
        downloaded = []
    paths = [Path(p) for p in downloaded if p]
    return sorted(paths)


def discover_local_gpm_files(cfg: dict[str, Any]) -> list[Path]:
    paths = cfg["paths"]
    gpm_dir = to_path(paths["gpm_dir"])
    gpm_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        str(to_path(paths["gpm_glob"])),
        str(gpm_dir / "*.h5"),
        str(gpm_dir / "*.H5"),
        str(gpm_dir / "*.hdf5"),
        str(gpm_dir / "*.HDF5"),
    ]
    found: list[Path] = []
    for pat in patterns:
        for p in glob.glob(pat):
            path = Path(p)
            if path.exists():
                found.append(path)
    return sorted(set(found))


def ensure_gpm_files(cfg: dict[str, Any]) -> list[Path]:
    gpm_dir = to_path(cfg["paths"]["gpm_dir"])
    gpm_dir.mkdir(parents=True, exist_ok=True)

    files = discover_local_gpm_files(cfg)
    if files and not bool(cfg["download"]["gpm"]["enabled"]):
        return files

    if bool(cfg["download"]["gpm"]["enabled"]):
        provider = str(cfg["download"]["gpm"]["provider"]).lower()
        downloaded: list[Path] = []
        try:
            if provider == "earthaccess":
                downloaded = _download_gpm_via_earthaccess(cfg, gpm_dir)
            elif provider == "cmr":
                downloaded = _download_gpm_via_cmr(cfg, gpm_dir)
        except Exception as exc:  # pragma: no cover - network/auth runtime
            print(f"[warn] GPM download failed via {provider}: {exc}")

        if downloaded:
            files = sorted(set(files + downloaded))

    selected = str(cfg["paths"].get("selected_gpm_file", "")).strip()
    if selected:
        sel_path = to_path(selected)
        if not sel_path.exists():
            raise FileNotFoundError(f"selected_gpm_file not found: {sel_path}")
        files = [sel_path]

    return sorted(set(files))


def choose_case_index(
    results: list[dict[str, Any]],
    selected_file: str | None = None,
    allowed_files: set[str] | None = None,
) -> int:
    if not results:
        raise ValueError("Empty results")
    if selected_file:
        selected_file = str(Path(selected_file).resolve())
        for idx, res in enumerate(results):
            if str(Path(res["path"]).resolve()) == selected_file:
                return idx

    best_idx = 0
    best_key = (-1, -1.0)
    for idx, res in enumerate(results):
        if allowed_files is not None and str(res["path"]) not in allowed_files:
            continue
        pumps = int(res["pump_count"])
        oidra = float(res["group_oidra"]) if np.isfinite(res["group_oidra"]) else -1.0
        key = (pumps, oidra)
        if key > best_key:
            best_key = key
            best_idx = idx
    if allowed_files is not None and best_key == (-1, -1.0):
        for idx, res in enumerate(results):
            if str(res["path"]) in allowed_files:
                return idx
    return best_idx


def save_summary_tables(results: list[dict[str, Any]], output_dir: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pump_rows: list[dict[str, Any]] = []
    for res in results:
        overpass = pd.to_datetime(res["overpass_time"], utc=True) if res["overpass_time"] is not None else pd.NaT
        pump_lats = [float(p["lat"]) for p in res["pumps"]]
        pump_lons = [float(p["lon"]) for p in res["pumps"]]
        pump_pos = [{"lat": la, "lon": lo} for la, lo in zip(pump_lats, pump_lons)]
        td_info = res.get("td_info")
        pump_c_lat = float(np.nanmean(pump_lats)) if pump_lats else float("nan")
        pump_c_lon = float(np.nanmean(pump_lons)) if pump_lons else float("nan")
        if td_info is not None and np.isfinite(pump_c_lat) and np.isfinite(pump_c_lon):
            pump_td_dist = float(np.hypot(pump_c_lat - float(td_info["center_lat"]), pump_c_lon - float(td_info["center_lon"])))
        else:
            pump_td_dist = float("nan")
        row = {
            "gpm_file": res["path"],
            "overpass_time_utc": overpass.isoformat() if pd.notna(overpass) else "",
            "td_detected": bool(res.get("td_detected", False)),
            "td_center_lat": float(td_info["center_lat"]) if td_info is not None else float("nan"),
            "td_center_lon": float(td_info["center_lon"]) if td_info is not None else float("nan"),
            "td_max_wind": float(td_info["max_wind"]) if td_info is not None else float("nan"),
            "td_pressure_diff": float(td_info["pressure_diff"]) if td_info is not None else float("nan"),
            "pump_count": res["pump_count"],
            "pump_centroid_lat": pump_c_lat,
            "pump_centroid_lon": pump_c_lon,
            "pump_td_distance_deg": pump_td_dist,
            "pump_positions_json": json.dumps(pump_pos, ensure_ascii=False),
            "group_oidra": res["group_oidra"],
            "mean_local_oidra": res["mean_local_oidra"],
            "mean_mse_var": res["mean_mse_var"],
            "median_mse_var": res["median_mse_var"],
            "precip_mean": res["precip_metrics"]["precip_mean"],
            "precip_max": res["precip_metrics"]["precip_max"],
            "precip_sum": res["precip_metrics"]["precip_sum"],
            "rain_area_fraction": res["precip_metrics"]["rain_area_fraction"],
            "heavy_rain_fraction": res["precip_metrics"]["heavy_rain_fraction"],
            "extreme_rain_fraction": res["precip_metrics"]["extreme_rain_fraction"],
            "stratiform_fraction": res["precip_metrics"]["stratiform_fraction"],
            "convective_fraction": res["precip_metrics"]["convective_fraction"],
            "other_fraction": res["precip_metrics"]["other_fraction"],
        }
        rows.append(row)

        for p in res["pumps"]:
            pump_rows.append(
                {
                    "gpm_file": res["path"],
                    "overpass_time_utc": row["overpass_time_utc"],
                    "pump_id": p["id"],
                    "lat": p["lat"],
                    "lon": p["lon"],
                    "area_voxels": p["area_voxels"],
                    "depth_km": p["depth_km"],
                    "max_dbz": p["max_dbz"],
                    "local_oidra": p["local_oidra"],
                    "mse_mean": p["mse_mean"],
                    "mse_var": p["mse_var"],
                    "rain_id": p["rain_id"],
                }
            )

    df = pd.DataFrame(rows)
    quantile = float(cfg.get("statistics", {}).get("oidra_split_quantile", 0.5))
    quantile = min(max(quantile, 0.05), 0.95)
    if not df.empty and df["group_oidra"].notna().sum() >= 2:
        qv = float(df["group_oidra"].quantile(quantile))
        df["oidra_split_value"] = qv
        df["oidra_class"] = np.where(df["group_oidra"] >= qv, "high_oidra", "low_oidra")
    else:
        df["oidra_split_value"] = float("nan")
        df["oidra_class"] = "unknown"

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "sample_library.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pump_rows).to_csv(output_dir / "water_pump_catalog.csv", index=False, encoding="utf-8-sig")

    rel = (
        df.groupby("oidra_class", dropna=False)[
            [
                "precip_mean",
                "precip_max",
                "precip_sum",
                "heavy_rain_fraction",
                "extreme_rain_fraction",
                "convective_fraction",
                "stratiform_fraction",
                "mean_mse_var",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    rel.to_csv(output_dir / "oidra_precip_summary.csv", index=False, encoding="utf-8-sig")
    return df


def apply_sample_filters(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame(columns=["gpm_file", "exclude_reason"])
        empty.to_csv(output_dir / "sample_exclusion_log.csv", index=False, encoding="utf-8-sig")
        return df.copy(), empty

    fcfg = cfg.get("statistics", {}).get("filter", {})
    min_pump_count = int(fcfg.get("min_pump_count", 1))
    min_precip_mean = float(fcfg.get("min_precip_mean", 0.0))
    max_mse_quant = float(fcfg.get("max_mean_mse_var_quantile", 0.99))
    min_type_cov = float(fcfg.get("min_precip_type_coverage", 0.5))
    require_td_center = bool(fcfg.get("require_td_center", True))
    max_mse_quant = min(max(max_mse_quant, 0.5), 1.0)

    q_mse = float(df["mean_mse_var"].quantile(max_mse_quant)) if df["mean_mse_var"].notna().any() else float("inf")

    keep_mask: list[bool] = []
    reason_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        reasons: list[str] = []
        if not np.isfinite(row.get("group_oidra", np.nan)):
            reasons.append("group_oidra_nan")
        if not np.isfinite(row.get("mean_mse_var", np.nan)):
            reasons.append("mean_mse_var_nan")
        if int(row.get("pump_count", 0)) < min_pump_count:
            reasons.append("pump_count_below_threshold")
        if np.isfinite(row.get("precip_mean", np.nan)) and float(row["precip_mean"]) < min_precip_mean:
            reasons.append("precip_mean_below_threshold")

        sf = row.get("stratiform_fraction", np.nan)
        cf = row.get("convective_fraction", np.nan)
        if np.isfinite(sf) and np.isfinite(cf) and float(sf + cf) < min_type_cov:
            reasons.append("precip_type_coverage_low")

        if np.isfinite(row.get("mean_mse_var", np.nan)) and float(row["mean_mse_var"]) > q_mse:
            reasons.append("mean_mse_var_outlier")
        if require_td_center and not bool(row.get("td_detected", False)):
            reasons.append("td_center_not_detected")

        keep = len(reasons) == 0
        keep_mask.append(keep)
        if not keep:
            reason_rows.append(
                {
                    "gpm_file": row.get("gpm_file", ""),
                    "overpass_time_utc": row.get("overpass_time_utc", ""),
                    "exclude_reason": ";".join(reasons),
                }
            )

    filtered = df.loc[np.asarray(keep_mask, dtype=bool)].copy()
    excluded = pd.DataFrame(reason_rows)
    if not excluded.empty:
        excluded.to_csv(output_dir / "sample_exclusion_log.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["gpm_file", "overpass_time_utc", "exclude_reason"]).to_csv(
            output_dir / "sample_exclusion_log.csv", index=False, encoding="utf-8-sig"
        )

    filtered.to_csv(output_dir / "sample_library_filtered.csv", index=False, encoding="utf-8-sig")
    return filtered, excluded


def _series_to_float_array(df: pd.DataFrame, col: str, mask: np.ndarray) -> np.ndarray:
    if col not in df.columns:
        return np.asarray([], dtype=np.float64)
    vals = pd.to_numeric(df.loc[mask, col], errors="coerce").to_numpy(dtype=np.float64)
    return vals[np.isfinite(vals)]


def _group_test(high: np.ndarray, low: np.ndarray) -> dict[str, float]:
    out = {
        "n_high": float(high.size),
        "n_low": float(low.size),
        "mean_high": float(np.nanmean(high)) if high.size else float("nan"),
        "mean_low": float(np.nanmean(low)) if low.size else float("nan"),
        "delta_high_minus_low": float(np.nanmean(high) - np.nanmean(low)) if high.size and low.size else float("nan"),
        "pvalue_mannwhitney": float("nan"),
    }
    if high.size >= 2 and low.size >= 2:
        try:
            stat = sstats.mannwhitneyu(high, low, alternative="two-sided")
            out["pvalue_mannwhitney"] = float(stat.pvalue)
        except Exception:
            pass
    return out


def summarize_organization_precip_relation(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_samples": int(len(df))}
    if df.empty or df["group_oidra"].notna().sum() < 2:
        out["message"] = "insufficient_samples"
        with (output_dir / "organization_precip_stats.json").open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out

    quantile = float(cfg.get("statistics", {}).get("oidra_split_quantile", 0.5))
    quantile = min(max(quantile, 0.05), 0.95)
    split_val = float(df["group_oidra"].quantile(quantile))
    split_mask = df["group_oidra"] >= split_val
    out["oidra_split_quantile"] = quantile
    out["oidra_split_value"] = split_val
    out["n_high_oidra"] = int(np.sum(split_mask))
    out["n_low_oidra"] = int(np.sum(~split_mask))

    metrics = [
        "precip_mean",
        "precip_max",
        "precip_sum",
        "heavy_rain_fraction",
        "extreme_rain_fraction",
        "convective_fraction",
        "stratiform_fraction",
        "mean_mse_var",
    ]
    tests: dict[str, dict[str, float]] = {}
    for metric in metrics:
        high = _series_to_float_array(df, metric, split_mask.to_numpy())
        low = _series_to_float_array(df, metric, (~split_mask).to_numpy())
        tests[metric] = _group_test(high, low)

    out["group_tests"] = tests

    keep = df["group_oidra"].notna() & df["precip_mean"].notna()
    if int(np.sum(keep)) >= 3:
        rho, pval = sstats.spearmanr(df.loc[keep, "group_oidra"], df.loc[keep, "precip_mean"])
        out["spearman_oidra_precip_mean"] = {"rho": float(rho), "pvalue": float(pval)}
    keep2 = df["group_oidra"].notna() & df["mean_mse_var"].notna()
    if int(np.sum(keep2)) >= 3:
        rho, pval = sstats.spearmanr(df.loc[keep2, "group_oidra"], df.loc[keep2, "mean_mse_var"])
        out["spearman_oidra_msevar"] = {"rho": float(rho), "pvalue": float(pval)}

    with (output_dir / "organization_precip_stats.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def build_mechanism_chain_text(stats_summary: dict[str, Any]) -> str:
    tests = stats_summary.get("group_tests", {})
    mse = tests.get("mean_mse_var", {})
    pm = tests.get("precip_mean", {})
    hf = tests.get("heavy_rain_fraction", {})

    mse_delta = float(mse.get("delta_high_minus_low", float("nan")))
    pm_delta = float(pm.get("delta_high_minus_low", float("nan")))
    hf_delta = float(hf.get("delta_high_minus_low", float("nan")))

    part1 = "环境热力背景更均一时，更有利于多个水泵形成协同组织。"
    part2 = "更高的水泵组织状态更容易对应更集中、更高效的降水结构。"

    if np.isfinite(mse_delta):
        if mse_delta < 0:
            part1 = "高 OIDRA 组对应更低的 MSE 方差，支持“环境更均一有利于协同组织”。"
        else:
            part1 = "高 OIDRA 组未表现出更低的 MSE 方差，环境均一性对组织的约束仍需更多样本检验。"

    if np.isfinite(pm_delta) and np.isfinite(hf_delta):
        if pm_delta > 0 or hf_delta > 0:
            part2 = "高 OIDRA 组对应更高的平均降水和/或强降水占比，支持“组织增强降水效率”。"
        else:
            part2 = "高 OIDRA 组在降水强度指标上未明显高于低 OIDRA 组，组织-降水链条仍需扩样本验证。"

    lines = [
        "# 机制链条总结",
        "",
        "本阶段基于样本库统计，机制链可表达为：",
        "",
        f"1. {part1}",
        f"2. {part2}",
        "",
        "归纳表述：环境热力背景更均一 -> 水泵更易协同组织 -> 降水结构更集中/更高效。",
    ]
    return "\n".join(lines)


def run_pipeline(cfg: dict[str, Any]) -> None:
    apply_common_plot_style()
    output_dir = to_path(cfg["study"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Preparing ERA5 data...")
    era_ds = ensure_era5_data(cfg)
    print(f"ERA5 ready: dims={dict(era_ds.sizes)}")

    print("[2/5] Preparing GPM files...")
    gpm_files = ensure_gpm_files(cfg)
    if not gpm_files:
        raise RuntimeError("No GPM files found/downloaded. Please check credentials, time range, and region settings.")
    print(f"GPM files: {len(gpm_files)}")

    print("[3/5] Running water-pump analysis for all granules...")
    results: list[dict[str, Any]] = []
    for fp in gpm_files:
        try:
            granule = read_gpm_granule(fp)
            if not np.any(region_mask(granule["lat"], granule["lon"], cfg["study"]["region"])):
                continue
            res = analyze_gpm_granule(granule, era_ds=era_ds, cfg=cfg)
            td_info = detect_tropical_depression_at_time(era_ds, time_idx=int(res["era_time_idx"]), cfg=cfg)
            res["td_info"] = td_info
            res["td_detected"] = td_info is not None
            results.append(res)
            print(
                f"  - {fp.name}: pumps={res['pump_count']}, group_oidra={res['group_oidra']:.4f}, "
                f"mean_mse_var={res['mean_mse_var']:.3e}, td={'Y' if res['td_detected'] else 'N'}"
            )
        except Exception as exc:
            print(f"  - skip {fp.name}: {exc}")

    if not results:
        raise RuntimeError("No valid granule intersects study region after analysis.")

    print("[4/5] Saving sample library and relationship summaries...")
    df = save_summary_tables(results, output_dir=output_dir, cfg=cfg)
    stats_df = df
    if bool(cfg.get("statistics", {}).get("enabled", True)):
        filtered_df, excluded_df = apply_sample_filters(df, cfg=cfg, output_dir=output_dir)
        stats_df = filtered_df if not filtered_df.empty else df
        plot_oidra_vs_precip(stats_df, output_dir / "fig_oidra_vs_precip.png")
        stats_summary = summarize_organization_precip_relation(stats_df, cfg=cfg, output_dir=output_dir)
        mechanism_text = build_mechanism_chain_text(stats_summary)
        with (output_dir / "mechanism_chain_summary.md").open("w", encoding="utf-8") as f:
            f.write(mechanism_text + "\n")
        print(
            f"  - samples(raw/filtered/excluded): "
            f"{len(df)}/{len(filtered_df)}/{len(excluded_df)}"
        )
    else:
        plot_oidra_vs_precip(df, output_dir / "fig_oidra_vs_precip.png")

    selected_file = str(cfg["paths"].get("selected_gpm_file", "")).strip() or None
    allowed_files = set(stats_df["gpm_file"].astype(str).tolist()) if not stats_df.empty else None
    case_idx = choose_case_index(results, selected_file=selected_file, allowed_files=allowed_files)
    case = results[case_idx]
    case_name = Path(case["path"]).stem
    case_dir = output_dir / f"case_{case_name}"
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"[5/5] Plotting case figures for: {Path(case['path']).name}")
    plot_cross_track_profile(case, cfg, case_dir / "fig_cross_track_profile.png")
    plot_gpm_multilayer_raw(case, cfg, case_dir / "fig_gpm_3d.png", case_dir / "fig_gpm_2d_layers.png")
    plot_surface_precip_and_type(case, cfg, case_dir / "fig_surface_precip_type.png")
    plot_oidra_mse_map(case, cfg, case_dir / "fig_oidra_mse_map.png")

    dep = case.get("td_info")
    if dep is None:
        dep = detect_tropical_depression_at_time(era_ds, time_idx=int(case["era_time_idx"]), cfg=cfg)
    if dep is not None:
        plot_pressure_field(dep, case_dir / "fig_pressure_field.png")
        plot_streamfunction_field(dep, case_dir / "fig_streamfunction_field.png")
        plot_relative_humidity_field(dep, case_dir / "fig_rh850_field.png")

        dep_summary = {
            "time": str(pd.to_datetime(dep["time"])),
            "center_lat": dep["center_lat"],
            "center_lon": dep["center_lon"],
            "max_wind": dep["max_wind"],
            "pressure_diff": dep["pressure_diff"],
            "center_rh": dep["center_rh"],
            "psi_min_lat": dep["psi_min_lat"],
            "psi_min_lon": dep["psi_min_lon"],
            "psi_min_value": dep["psi_min_value"],
        }
        with (case_dir / "tropical_depression_summary.json").open("w", encoding="utf-8") as f:
            json.dump(dep_summary, f, ensure_ascii=False, indent=2)

    with (case_dir / "selected_case_pumps.json").open("w", encoding="utf-8") as f:
        json.dump(case["pumps"], f, ensure_ascii=False, indent=2)

    print("Pipeline done.")
    print(f"Outputs saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "South China Sea non-monsoon tropical depression research pipeline: "
            "ERA5/GPM download + water-pump identification + OIDRA/MSE + plotting."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/td_waterpump_research.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--dump-default-config",
        action="store_true",
        help="Print default config YAML to stdout and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dump_default_config:
        print(yaml.safe_dump(DEFAULT_CONFIG, allow_unicode=False))
        return

    cfg = load_config(args.config)
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    run_pipeline(cfg)
    end_time = datetime.now()
    print(f"End time: {end_time}")
    print(f"Elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()
