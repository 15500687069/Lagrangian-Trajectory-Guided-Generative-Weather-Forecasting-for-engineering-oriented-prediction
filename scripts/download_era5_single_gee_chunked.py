from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_era5_from_manual import build_dataset, save_dataset


GEE_ERA5_HOURLY = "ECMWF/ERA5/HOURLY"


@dataclass(frozen=True)
class YearMonth:
    year: int
    month: int

    @property
    def tag(self) -> str:
        return f"{self.year:04d}{self.month:02d}"


def parse_year_month(value: str) -> YearMonth:
    parts = value.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected YYYY-MM, got: {value}")
    year = int(parts[0])
    month = int(parts[1])
    if year < 1940 or month < 1 or month > 12:
        raise ValueError(f"Invalid year/month: {value}")
    return YearMonth(year, month)


def month_range(start: YearMonth, end: YearMonth) -> list[YearMonth]:
    if (start.year, start.month) > (end.year, end.month):
        raise ValueError("start must be <= end")
    months: list[YearMonth] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append(YearMonth(y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return months


def chunked(items: list, chunk_size: int) -> list[list]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _is_valid_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with xr.open_dataset(path):
            return True
    except Exception:
        return False


def _normalize_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    rename_dims = {}
    for name in list(ds.dims):
        low = name.lower()
        if low in {"lat", "latitude", "y"}:
            rename_dims[name] = "latitude"
        if low in {"lon", "longitude", "x"}:
            rename_dims[name] = "longitude"
    if rename_dims:
        ds = ds.rename(rename_dims)

    rename_coords = {}
    for name in list(ds.coords):
        low = name.lower()
        if low in {"lat", "latitude", "y"} and name != "latitude":
            rename_coords[name] = "latitude"
        if low in {"lon", "longitude", "x"} and name != "longitude":
            rename_coords[name] = "longitude"
    if rename_coords:
        ds = ds.rename(rename_coords)
    return ds


def _ensure_time_subset(ds: xr.Dataset, step_hours: int) -> xr.Dataset:
    if "time" not in ds.coords:
        raise RuntimeError("GEE dataset has no `time` coordinate.")
    if step_hours <= 1:
        return ds
    times = pd.to_datetime(ds["time"].values)
    mask = (times.hour % step_hours) == 0
    keep = [i for i, flag in enumerate(mask) if bool(flag)]
    return ds.isel(time=keep)


def _month_bounds(ym: YearMonth) -> tuple[str, str]:
    start = f"{ym.year:04d}-{ym.month:02d}-01"
    if ym.month == 12:
        end = f"{ym.year + 1:04d}-01-01"
    else:
        end = f"{ym.year:04d}-{ym.month + 1:02d}-01"
    return start, end


def _init_ee(ee_project: str, ee_authenticate: bool, ee_high_volume: bool):
    if not ee_project:
        raise ValueError("`--ee-project` is required for Earth Engine initialization.")
    try:
        import ee
    except Exception as exc:
        raise ImportError(
            "earthengine-api is required. Install with `pip install earthengine-api`."
        ) from exc

    if ee_authenticate:
        ee.Authenticate()

    init_kwargs = {"project": ee_project}
    if ee_high_volume:
        init_kwargs["opt_url"] = "https://earthengine-highvolume.googleapis.com"
    ee.Initialize(**init_kwargs)
    return ee


def _load_month_from_gee(
    ee_module,
    ym: YearMonth,
    area: list[float],
    step_hours: int,
    scale_meters: float,
) -> xr.Dataset:
    try:
        import xee  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "xee is required for xarray<->GEE access. Install with `pip install xee`."
        ) from exc

    start_iso, end_iso = _month_bounds(ym)
    north, west, south, east = area
    geom = ee_module.Geometry.Rectangle([west, south, east, north], proj=None, geodesic=False)

    ic = (
        ee_module.ImageCollection(GEE_ERA5_HOURLY)
        .filterDate(start_iso, end_iso)
        .select(["mean_sea_level_pressure", "total_precipitation"])
    )

    ds = xr.open_dataset(
        ic,
        engine="ee",
        geometry=geom,
        crs="EPSG:4326",
        scale=scale_meters,
    )
    ds = ds.load()
    ds = _normalize_lat_lon(ds)
    ds = _ensure_time_subset(ds, step_hours=step_hours)

    if "mean_sea_level_pressure" not in ds.data_vars:
        raise RuntimeError(
            f"`mean_sea_level_pressure` not found in GEE ERA5 hourly data vars: {list(ds.data_vars)}"
        )
    if "total_precipitation" not in ds.data_vars:
        raise RuntimeError(
            f"`total_precipitation` not found in GEE ERA5 hourly data vars: {list(ds.data_vars)}"
        )

    out = xr.Dataset(
        data_vars={
            "msl": ds["mean_sea_level_pressure"].astype("float32"),
            "tp": ds["total_precipitation"].astype("float32"),
        }
    )
    out = out.sortby("time")
    if "longitude" in out.coords:
        out = out.sortby("longitude")
    return out


def _save_monthly_files(ds: xr.Dataset, instant_file: Path, accum_file: Path) -> None:
    instant_file.parent.mkdir(parents=True, exist_ok=True)
    accum_file.parent.mkdir(parents=True, exist_ok=True)
    ds[["msl"]].to_netcdf(instant_file)
    ds[["tp"]].to_netcdf(accum_file)


def _merge_monthly(files: Iterable[Path], output: Path, overwrite: bool) -> None:
    valid = [f for f in files if _is_valid_netcdf(f)]
    if not valid:
        raise RuntimeError("No valid monthly files to merge.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite-merged.")
        output.unlink()
    with xr.open_mfdataset([str(f) for f in sorted(valid)], combine="by_coords") as ds:
        ds = ds.sortby("time")
        if "longitude" in ds.coords:
            ds = ds.sortby("longitude")
        output.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output)


def run_download(
    start: YearMonth,
    end: YearMonth,
    cache_dir: Path,
    time_step_hours: int,
    area: list[float],
    chunk_months: int,
    chunk_sleep_seconds: float,
    ee_project: str,
    ee_authenticate: bool,
    ee_high_volume: bool,
    scale_meters: float,
    output_instant: Path,
    output_accum: Path,
    overwrite_merged: bool,
    pressure_file: Path | None,
    output_final: Path | None,
    overwrite_final: bool,
) -> None:
    ee_module = _init_ee(
        ee_project=ee_project,
        ee_authenticate=ee_authenticate,
        ee_high_volume=ee_high_volume,
    )
    months = month_range(start, end)

    monthly_dir = cache_dir / "monthly"
    monthly_inst_dir = monthly_dir / "instant"
    monthly_acc_dir = monthly_dir / "accum"
    monthly_inst_dir.mkdir(parents=True, exist_ok=True)
    monthly_acc_dir.mkdir(parents=True, exist_ok=True)

    print(f"GEE ERA5 single-level from {start.tag} to {end.tag} ({len(months)} months)")
    print(f"Dataset: {GEE_ERA5_HOURLY}")
    print(f"Time step: {time_step_hours}h")
    print(f"Area: north={area[0]}, west={area[1]}, south={area[2]}, east={area[3]}")
    print(f"Chunk months: {chunk_months}, chunk sleep: {chunk_sleep_seconds}s")
    print(f"Cache: {cache_dir}")

    monthly_inst_files: list[Path] = []
    monthly_acc_files: list[Path] = []
    month_chunks = chunked(months, chunk_months) if chunk_months > 0 else [months]
    for idx, chunk in enumerate(month_chunks, start=1):
        print(f"\n[chunk {idx}/{len(month_chunks)}] {chunk[0].tag} -> {chunk[-1].tag}")
        for ym in tqdm(chunk, desc=f"gee-chunk-{idx}", unit="month"):
            inst_file = monthly_inst_dir / f"single_instant_{ym.tag}.nc"
            acc_file = monthly_acc_dir / f"single_accum_{ym.tag}.nc"
            monthly_inst_files.append(inst_file)
            monthly_acc_files.append(acc_file)
            if _is_valid_netcdf(inst_file) and _is_valid_netcdf(acc_file):
                continue

            ds = _load_month_from_gee(
                ee_module=ee_module,
                ym=ym,
                area=area,
                step_hours=time_step_hours,
                scale_meters=scale_meters,
            )
            _save_monthly_files(ds=ds, instant_file=inst_file, accum_file=acc_file)

        if idx < len(month_chunks) and chunk_sleep_seconds > 0:
            print(f"[chunk-sleep] sleep {chunk_sleep_seconds:.1f}s")
            time.sleep(chunk_sleep_seconds)

    print("\nMerging single-level instant ...")
    _merge_monthly(monthly_inst_files, output=output_instant, overwrite=overwrite_merged)
    print(f"Saved instant: {output_instant}")

    print("Merging single-level accum ...")
    _merge_monthly(monthly_acc_files, output=output_accum, overwrite=overwrite_merged)
    print(f"Saved accum: {output_accum}")

    if pressure_file is not None and output_final is not None:
        print("\nBuilding LTG dataset with existing pressure file ...")
        ds = build_dataset(
            pressure_file=pressure_file,
            instant_file=output_instant,
            accum_file=output_accum,
        )
        save_dataset(ds, output=output_final, overwrite=overwrite_final)
        print(f"Saved final LTG dataset: {output_final}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download ERA5 single-level data from GEE (chunked by month) without changing CDS scripts."
    )
    p.add_argument("--start", required=True, help="YYYY-MM")
    p.add_argument("--end", required=True, help="YYYY-MM")
    p.add_argument("--cache-dir", default="data/era5_gee_single_cache")
    p.add_argument("--time-step-hours", type=int, default=1)
    p.add_argument("--chunk-months", type=int, default=2, help="Load N months per chunk.")
    p.add_argument("--chunk-sleep-seconds", type=float, default=10.0)
    p.add_argument("--area", nargs=4, type=float, default=[50.0, 100.0, 10.0, 170.0], metavar=("N", "W", "S", "E"))
    p.add_argument("--scale-meters", type=float, default=27830.0, help="Sampling scale for GEE xarray engine.")
    p.add_argument("--ee-project", required=True, help="GEE cloud project id.")
    p.add_argument("--ee-authenticate", action="store_true", help="Run ee.Authenticate() before initialization.")
    p.add_argument(
        "--ee-high-volume",
        action="store_true",
        help="Use high-volume endpoint https://earthengine-highvolume.googleapis.com",
    )
    p.add_argument("--output-instant", required=True, help="Merged instant file (.nc)")
    p.add_argument("--output-accum", required=True, help="Merged accum file (.nc)")
    p.add_argument("--overwrite-merged", action="store_true")
    p.add_argument("--pressure-file", default=None, help="Existing pressure-level file")
    p.add_argument("--output-final", default=None, help="Final LTG dataset output (.nc/.zarr)")
    p.add_argument("--overwrite-final", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    start = parse_year_month(args.start)
    end = parse_year_month(args.end)

    pressure_file = Path(args.pressure_file) if args.pressure_file else None
    output_final = Path(args.output_final) if args.output_final else None
    if (pressure_file is None) ^ (output_final is None):
        raise ValueError("Use --pressure-file and --output-final together, or neither.")

    run_download(
        start=start,
        end=end,
        cache_dir=Path(args.cache_dir),
        time_step_hours=int(args.time_step_hours),
        area=list(args.area),
        chunk_months=int(args.chunk_months),
        chunk_sleep_seconds=float(args.chunk_sleep_seconds),
        ee_project=args.ee_project or os.environ.get("EE_PROJECT", ""),
        ee_authenticate=bool(args.ee_authenticate),
        ee_high_volume=bool(args.ee_high_volume),
        scale_meters=float(args.scale_meters),
        output_instant=Path(args.output_instant),
        output_accum=Path(args.output_accum),
        overwrite_merged=bool(args.overwrite_merged),
        pressure_file=pressure_file,
        output_final=output_final,
        overwrite_final=bool(args.overwrite_final),
    )


if __name__ == "__main__":
    main()
