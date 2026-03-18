from __future__ import annotations

import argparse
import calendar
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import xarray as xr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_era5_from_manual import build_dataset, save_dataset


SINGLE_DATASET = "reanalysis-era5-single-levels"


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


def days_in_month(year: int, month: int) -> list[str]:
    count = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, count + 1)]


def chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_time_list(step_hours: int) -> list[str]:
    if step_hours < 1 or step_hours > 24 or 24 % step_hours != 0:
        raise ValueError("time_step_hours must divide 24, e.g. 1/2/3/4/6/8/12/24")
    return [f"{h:02d}:00" for h in range(0, 24, step_hours)]


def _normalize_ascii_token(value: str, name: str) -> str:
    cleaned = (
        value.strip()
        .replace("\ufeff", "")
        .replace("\u201c", "")
        .replace("\u201d", "")
        .replace("\u2018", "")
        .replace("\u2019", "")
        .replace("“", "")
        .replace("”", "")
        .replace("‘", "")
        .replace("’", "")
    )
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1].strip()

    if cleaned.isascii():
        return cleaned

    filtered = "".join(ch for ch in cleaned if ord(ch) < 128)
    if filtered and filtered.isascii():
        print(f"[warn] Non-ASCII chars found in {name}, auto-filtered.")
        return filtered

    bad = [f"U+{ord(ch):04X}" for ch in cleaned if ord(ch) >= 128]
    raise ValueError(
        f"{name} contains non-ASCII characters ({', '.join(bad)}). "
        "Please re-copy URL/KEY in plain ASCII."
    )


def _normalize_cds_key(key: str | None) -> str | None:
    if key is None:
        return None
    value = _normalize_ascii_token(key, name="cds_key")
    if ":" in value:
        left, right = value.split(":", 1)
        if left.isdigit() and right:
            return right.strip()
    return value


def _build_cds_url_candidates(cds_url: str | None) -> list[str | None]:
    if cds_url:
        base = _normalize_ascii_token(cds_url, name="cds_url").rstrip("/")
        if base.endswith("/api/v2"):
            return [base, base[:-3]]
        if base.endswith("/api"):
            return [base, f"{base}/v2"]
        return [base, f"{base}/api", f"{base}/api/v2"]
    return [None, "https://cds.climate.copernicus.eu/api", "https://cds.climate.copernicus.eu/api/v2"]


def _is_endpoint_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "endpoint not found" in msg or "does not exist" in msg or "api endpoint" in msg


def _build_cds_clients(cdsapi_module, cds_url: str | None, cds_key: str | None):
    key = _normalize_cds_key(cds_key)
    urls = _build_cds_url_candidates(cds_url)
    clients = []
    for url in urls:
        kwargs = {}
        if url is not None:
            kwargs["url"] = url
        if key:
            kwargs["key"] = key
        try:
            client = cdsapi_module.Client(**kwargs)
            clients.append((client, url or "<local-config>"))
        except Exception:
            continue
    if not clients:
        raise RuntimeError("Unable to initialize CDS API client with provided URL/key settings.")
    return clients


def _is_valid_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with xr.open_dataset(path):
            return True
    except Exception:
        return False


def _drop_auxiliary(ds: xr.Dataset) -> xr.Dataset:
    drop_vars = [v for v in ["number", "expver"] if v in ds.data_vars]
    if drop_vars:
        ds = ds.drop_vars(drop_vars)
    return ds


def _normalize_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def _retrieve_with_retry(
    clients,
    dataset: str,
    request: dict,
    target: Path,
    max_retries: int,
    sleep_seconds: float,
) -> None:
    if _is_valid_netcdf(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)

    variants = []
    req1 = dict(request)
    req1["data_format"] = "netcdf"
    req1["download_format"] = "unarchived"
    variants.append(req1)

    req2 = dict(request)
    req2["format"] = "netcdf"
    variants.append(req2)

    last_error = None
    for client, client_label in clients:
        for req in variants:
            for attempt in range(max_retries):
                try:
                    target.unlink(missing_ok=True)
                    client.retrieve(dataset, req).download(str(target))
                    if _is_valid_netcdf(target):
                        return
                    raise RuntimeError(f"Empty/broken file: {target}")
                except Exception as exc:
                    last_error = exc
                    if _is_endpoint_not_found_error(exc):
                        print(f"[switch-url] endpoint not found on {client_label}, try next URL...")
                        break
                    if attempt + 1 >= max_retries:
                        break
                    wait_s = sleep_seconds * (2**attempt)
                    print(
                        f"[retry] {dataset} -> {target.name} url={client_label} attempt {attempt + 1}/{max_retries} "
                        f"sleep {wait_s:.1f}s: {exc}"
                    )
                    time.sleep(wait_s)
    raise RuntimeError(f"Failed download: {dataset} -> {target}") from last_error


def _download_month_single(
    ym: YearMonth,
    clients,
    raw_dir: Path,
    monthly_instant_file: Path,
    monthly_accum_file: Path,
    times: list[str],
    area: list[float] | None,
    days_per_request: int,
    max_retries: int,
    sleep_seconds: float,
    cleanup_raw: bool,
) -> tuple[Path, Path]:
    if _is_valid_netcdf(monthly_instant_file) and _is_valid_netcdf(monthly_accum_file):
        return monthly_instant_file, monthly_accum_file

    month_days = days_in_month(ym.year, ym.month)
    day_groups = chunked(month_days, days_per_request)
    inst_parts: list[Path] = []
    accum_parts: list[Path] = []
    for group in day_groups:
        chunk_tag = f"d{group[0]}-{group[-1]}"
        inst_file = raw_dir / f"single_instant_{ym.tag}_{chunk_tag}.nc"
        accum_file = raw_dir / f"single_accum_{ym.tag}_{chunk_tag}.nc"
        inst_parts.append(inst_file)
        accum_parts.append(accum_file)

        req_common = {
            "product_type": "reanalysis",
            "year": f"{ym.year:04d}",
            "month": f"{ym.month:02d}",
            "day": group,
            "time": times,
        }
        if area is not None:
            req_common["area"] = area

        inst_req = dict(req_common)
        inst_req["variable"] = ["mean_sea_level_pressure"]

        accum_req = dict(req_common)
        accum_req["variable"] = ["total_precipitation"]

        _retrieve_with_retry(
            clients=clients,
            dataset=SINGLE_DATASET,
            request=inst_req,
            target=inst_file,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
        _retrieve_with_retry(
            clients=clients,
            dataset=SINGLE_DATASET,
            request=accum_req,
            target=accum_file,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )

    monthly_instant_file.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_mfdataset([str(p) for p in inst_parts], combine="by_coords") as ds_inst:
        ds_inst = _drop_auxiliary(_normalize_time(ds_inst)).sortby("time").sortby("longitude")
        ds_inst["msl"] = ds_inst["msl"].astype("float32")
        ds_inst[["msl"]].to_netcdf(monthly_instant_file)
    with xr.open_mfdataset([str(p) for p in accum_parts], combine="by_coords") as ds_acc:
        ds_acc = _drop_auxiliary(_normalize_time(ds_acc)).sortby("time").sortby("longitude")
        ds_acc["tp"] = ds_acc["tp"].astype("float32")
        ds_acc[["tp"]].to_netcdf(monthly_accum_file)

    if cleanup_raw:
        for f in inst_parts + accum_parts:
            f.unlink(missing_ok=True)
    return monthly_instant_file, monthly_accum_file


def _merge_monthly(files: Iterable[Path], output: Path, overwrite: bool) -> None:
    valid = [f for f in files if _is_valid_netcdf(f)]
    if not valid:
        raise RuntimeError("No valid monthly files to merge.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite-merged.")
        output.unlink()
    with xr.open_mfdataset([str(f) for f in sorted(valid)], combine="by_coords") as ds:
        ds = ds.sortby("time").sortby("longitude")
        output.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output)


def run_download(
    start: YearMonth,
    end: YearMonth,
    cache_dir: Path,
    time_step_hours: int,
    area: list[float] | None,
    days_per_request: int,
    chunk_months: int,
    chunk_sleep_seconds: float,
    max_retries: int,
    sleep_seconds: float,
    cleanup_raw: bool,
    cds_url: str | None,
    cds_key: str | None,
    output_instant: Path,
    output_accum: Path,
    overwrite_merged: bool,
    pressure_file: Path | None,
    output_final: Path | None,
    overwrite_final: bool,
) -> None:
    try:
        import cdsapi
    except Exception as exc:
        raise ImportError("cdsapi is required. Install with `pip install cdsapi`.") from exc

    clients = _build_cds_clients(cdsapi, cds_url=cds_url, cds_key=cds_key)
    times = build_time_list(time_step_hours)
    months = month_range(start, end)

    raw_dir = cache_dir / "raw"
    monthly_dir = cache_dir / "monthly"
    monthly_inst_dir = monthly_dir / "instant"
    monthly_acc_dir = monthly_dir / "accum"
    raw_dir.mkdir(parents=True, exist_ok=True)
    monthly_inst_dir.mkdir(parents=True, exist_ok=True)
    monthly_acc_dir.mkdir(parents=True, exist_ok=True)

    print(f"Download single-level ERA5 from {start.tag} to {end.tag} ({len(months)} months)")
    print(f"Time step: {time_step_hours}h -> {times}")
    print(f"Days per request: {days_per_request}, month chunk: {chunk_months}")
    if area is not None:
        print(f"Area: north={area[0]}, west={area[1]}, south={area[2]}, east={area[3]}")

    monthly_inst_files: list[Path] = []
    monthly_acc_files: list[Path] = []
    month_chunks = chunked(months, chunk_months) if chunk_months > 0 else [months]
    for chunk_idx, chunk in enumerate(month_chunks, start=1):
        print(f"\n[chunk {chunk_idx}/{len(month_chunks)}] {chunk[0].tag} -> {chunk[-1].tag}")
        for ym in tqdm(chunk, desc=f"chunk-{chunk_idx}", unit="month"):
            inst_m = monthly_inst_dir / f"single_instant_{ym.tag}.nc"
            acc_m = monthly_acc_dir / f"single_accum_{ym.tag}.nc"
            inst_m, acc_m = _download_month_single(
                ym=ym,
                clients=clients,
                raw_dir=raw_dir,
                monthly_instant_file=inst_m,
                monthly_accum_file=acc_m,
                times=times,
                area=area,
                days_per_request=days_per_request,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
                cleanup_raw=cleanup_raw,
            )
            monthly_inst_files.append(inst_m)
            monthly_acc_files.append(acc_m)
        if chunk_idx < len(month_chunks) and chunk_sleep_seconds > 0:
            print(f"[chunk-sleep] sleep {chunk_sleep_seconds:.1f}s to reduce queue pressure.")
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
    p = argparse.ArgumentParser(description="Chunked download for ERA5 single-level (msl/tp) with resume.")
    p.add_argument("--start", required=True, help="YYYY-MM")
    p.add_argument("--end", required=True, help="YYYY-MM")
    p.add_argument("--cache-dir", default="data/era5_single_download_cache")
    p.add_argument("--time-step-hours", type=int, default=1)
    p.add_argument("--days-per-request", type=int, default=15)
    p.add_argument("--chunk-months", type=int, default=3, help="Download N months per chunk.")
    p.add_argument("--chunk-sleep-seconds", type=float, default=30.0)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--sleep-seconds", type=float, default=20.0)
    p.add_argument("--area", nargs=4, type=float, default=None, metavar=("NORTH", "WEST", "SOUTH", "EAST"))
    p.add_argument("--cds-url", default=None)
    p.add_argument("--cds-key", default=None)
    p.add_argument("--cleanup-raw", action="store_true")
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
        area=list(args.area) if args.area is not None else None,
        days_per_request=int(args.days_per_request),
        chunk_months=int(args.chunk_months),
        chunk_sleep_seconds=float(args.chunk_sleep_seconds),
        max_retries=int(args.max_retries),
        sleep_seconds=float(args.sleep_seconds),
        cleanup_raw=bool(args.cleanup_raw),
        cds_url=args.cds_url or os.environ.get("CDSAPI_URL"),
        cds_key=args.cds_key or os.environ.get("CDSAPI_KEY"),
        output_instant=Path(args.output_instant),
        output_accum=Path(args.output_accum),
        overwrite_merged=bool(args.overwrite_merged),
        pressure_file=pressure_file,
        output_final=output_final,
        overwrite_final=bool(args.overwrite_final),
    )


if __name__ == "__main__":
    main()
