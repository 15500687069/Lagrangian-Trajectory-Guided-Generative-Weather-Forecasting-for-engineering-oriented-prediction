from __future__ import annotations

import argparse
import calendar
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import xarray as xr
from tqdm import tqdm


PRESSURE_DATASET = "reanalysis-era5-pressure-levels"
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
        # New CDS API expects key without deprecated <UID>: prefix.
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


def _retrieve_with_retry(
    clients,
    dataset: str,
    request: dict,
    target: Path,
    max_retries: int,
    sleep_seconds: float,
) -> None:
    if target.exists() and target.stat().st_size > 0:
        try:
            with xr.open_dataset(target):
                return
        except Exception:
            target.unlink(missing_ok=True)
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
                    if target.exists() and target.stat().st_size > 0:
                        with xr.open_dataset(target):
                            pass
                        return
                    raise RuntimeError(f"Empty download file: {target}")
                except Exception as exc:
                    last_error = exc
                    if _is_endpoint_not_found_error(exc):
                        print(f"[switch-url] endpoint not found on {client_label}, try next URL...")
                        break
                    if attempt + 1 >= max_retries:
                        break
                    wait_s = sleep_seconds * (2**attempt)
                    print(
                        f"[retry] {dataset} -> {target.name} url={client_label} attempt {attempt + 1}/{max_retries} failed, "
                        f"sleep {wait_s:.1f}s: {exc}"
                    )
                    time.sleep(wait_s)

    raise RuntimeError(f"Failed to download {dataset} -> {target}") from last_error


def _to_0360_longitude(ds: xr.Dataset) -> xr.Dataset:
    if "longitude" not in ds.coords:
        return ds
    lon = ds["longitude"]
    if float(lon.min()) < 0.0:
        ds = ds.assign_coords(longitude=(lon % 360)).sortby("longitude")
    return ds


def _standardize_month(
    pressure_files: list[Path],
    single_files: list[Path],
    monthly_out_file: Path,
) -> None:
    if monthly_out_file.exists() and monthly_out_file.stat().st_size > 0:
        return

    with xr.open_mfdataset([str(p) for p in pressure_files], combine="by_coords") as dp, xr.open_mfdataset(
        [str(p) for p in single_files], combine="by_coords"
    ) as ds:
        z500 = dp["z"].sel(pressure_level=500).rename("z500")
        z850 = dp["z"].sel(pressure_level=850).rename("z850")
        u850 = dp["u"].sel(pressure_level=850).rename("u850")
        v850 = dp["v"].sel(pressure_level=850).rename("v850")
        t850 = dp["t"].sel(pressure_level=850).rename("t850")
        q850 = dp["q"].sel(pressure_level=850).rename("q850")
        msl = ds["msl"].rename("msl")
        tp = ds["tp"].rename("tp")

        out = xr.Dataset(
            data_vars={
                "z500": z500,
                "z850": z850,
                "u850": u850,
                "v850": v850,
                "t850": t850,
                "q850": q850,
                "msl": msl,
                "tp": tp,
            }
        )
        out = _to_0360_longitude(out).sortby("time")
        for var_name in list(out.data_vars):
            out[var_name] = out[var_name].astype("float32")
        monthly_out_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(monthly_out_file)


def _is_valid_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with xr.open_dataset(path):
            return True
    except Exception:
        return False


def _merge_months(
    monthly_files: Iterable[Path],
    output_path: Path,
    overwrite: bool,
) -> None:
    files = sorted(monthly_files)
    if not files:
        raise RuntimeError("No monthly files to merge.")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace.")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    ds = xr.open_mfdataset([str(f) for f in files], combine="by_coords")
    ds = ds.sortby("time")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".zarr":
        try:
            import zarr  # noqa: F401
        except Exception as exc:
            raise ImportError("zarr is required for .zarr output. Install with `pip install zarr`.") from exc
        ds.to_zarr(str(output_path), mode="w")
    else:
        ds.to_netcdf(str(output_path))
    ds.close()


def download_era5(
    start: YearMonth,
    end: YearMonth,
    output_path: Path,
    cache_dir: Path,
    time_step_hours: int,
    area: list[float] | None,
    days_per_request: int,
    max_retries: int,
    sleep_seconds: float,
    overwrite: bool,
    cleanup_raw: bool,
    cleanup_monthly: bool,
    cds_url: str | None,
    cds_key: str | None,
) -> None:
    try:
        import cdsapi
    except Exception as exc:
        raise ImportError(
            "cdsapi is required. Install with `pip install cdsapi`, and configure ~/.cdsapirc."
        ) from exc

    months = month_range(start, end)
    times = build_time_list(time_step_hours)
    clients = _build_cds_clients(cdsapi, cds_url=cds_url, cds_key=cds_key)

    raw_dir = cache_dir / "raw"
    monthly_dir = cache_dir / "monthly"
    raw_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ERA5 from {start.tag} to {end.tag} ({len(months)} months)")
    if area is not None:
        print(f"Area: north={area[0]}, west={area[1]}, south={area[2]}, east={area[3]}")
    print(f"Time step: {time_step_hours}h -> {times}")
    print(f"Days per request: {days_per_request}")

    monthly_outputs: list[Path] = []
    for ym in tqdm(months, desc="monthly", unit="month"):
        tag = ym.tag
        m_file = monthly_dir / f"era5_{tag}.nc"
        if _is_valid_netcdf(m_file):
            monthly_outputs.append(m_file)
            continue
        month_days = days_in_month(ym.year, ym.month)
        day_groups = chunked(month_days, days_per_request)
        p_files: list[Path] = []
        s_files: list[Path] = []

        for group in day_groups:
            chunk_tag = f"d{group[0]}-{group[-1]}"
            p_file = raw_dir / f"pressure_{tag}_{chunk_tag}.nc"
            s_file = raw_dir / f"single_{tag}_{chunk_tag}.nc"
            p_files.append(p_file)
            s_files.append(s_file)

            req_common = {
                "product_type": "reanalysis",
                "year": f"{ym.year:04d}",
                "month": f"{ym.month:02d}",
                "day": group,
                "time": times,
            }
            if area is not None:
                req_common["area"] = area

            pressure_req = dict(req_common)
            pressure_req["variable"] = [
                "geopotential",
                "u_component_of_wind",
                "v_component_of_wind",
                "temperature",
                "specific_humidity",
            ]
            pressure_req["pressure_level"] = ["500", "850"]

            single_req = dict(req_common)
            single_req["variable"] = ["mean_sea_level_pressure", "total_precipitation"]

            _retrieve_with_retry(
                clients=clients,
                dataset=PRESSURE_DATASET,
                request=pressure_req,
                target=p_file,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
            _retrieve_with_retry(
                clients=clients,
                dataset=SINGLE_DATASET,
                request=single_req,
                target=s_file,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )

        _standardize_month(p_files, s_files, m_file)
        monthly_outputs.append(m_file)

        if cleanup_raw:
            for f in p_files + s_files:
                if f.exists():
                    f.unlink()

    print("Merging monthly files ...")
    _merge_months(monthly_outputs, output_path=output_path, overwrite=overwrite)
    print(f"Done: {output_path}")

    if cleanup_monthly:
        shutil.rmtree(monthly_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download ERA5 and build LTG-Net ready dataset.")
    parser.add_argument("--start", required=True, help="Start month, format YYYY-MM, e.g. 1990-01")
    parser.add_argument("--end", required=True, help="End month, format YYYY-MM, e.g. 2021-12")
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset path (.zarr or .nc), e.g. data/era5.zarr",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/era5_download_cache",
        help="Cache directory for raw and monthly files.",
    )
    parser.add_argument(
        "--time-step-hours",
        type=int,
        default=6,
        help="Temporal sampling interval in hours (must divide 24).",
    )
    parser.add_argument(
        "--days-per-request",
        type=int,
        default=7,
        help="Days per CDS request. Lower value is slower but more stable.",
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        default=None,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="Optional subregion [N, W, S, E], e.g. 60 80 -10 200",
    )
    parser.add_argument("--max-retries", type=int, default=6, help="Retry attempts per request.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=20.0,
        help="Base sleep between retries (exponential backoff).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if exists.")
    parser.add_argument(
        "--cleanup-raw",
        action="store_true",
        help="Delete raw monthly downloaded files after each monthly standardization.",
    )
    parser.add_argument(
        "--cleanup-monthly",
        action="store_true",
        help="Delete standardized monthly files after final merge.",
    )
    parser.add_argument(
        "--cds-url",
        default=None,
        help="Optional CDS API URL. If omitted, use CDSAPI_URL env or ~/.cdsapirc.",
    )
    parser.add_argument(
        "--cds-key",
        default=None,
        help="Optional CDS API key. If omitted, use CDSAPI_KEY env or ~/.cdsapirc.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start = parse_year_month(args.start)
    end = parse_year_month(args.end)
    output = Path(args.output)
    cache_dir = Path(args.cache_dir)

    download_era5(
        start=start,
        end=end,
        output_path=output,
        cache_dir=cache_dir,
        time_step_hours=int(args.time_step_hours),
        area=list(args.area) if args.area is not None else None,
        days_per_request=int(args.days_per_request),
        max_retries=int(args.max_retries),
        sleep_seconds=float(args.sleep_seconds),
        overwrite=bool(args.overwrite),
        cleanup_raw=bool(args.cleanup_raw),
        cleanup_monthly=bool(args.cleanup_monthly),
        cds_url=args.cds_url or os.environ.get("CDSAPI_URL"),
        cds_key=args.cds_key or os.environ.get("CDSAPI_KEY"),
    )


if __name__ == "__main__":
    main()

