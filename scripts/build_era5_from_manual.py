from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


def _open(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)


def _normalize_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def _drop_auxiliary(ds: xr.Dataset) -> xr.Dataset:
    drop_vars = [v for v in ["number", "expver"] if v in ds.data_vars]
    if drop_vars:
        ds = ds.drop_vars(drop_vars)
    return ds


def build_dataset(
    pressure_file: Path,
    instant_file: Path,
    accum_file: Path,
) -> xr.Dataset:
    with _open(pressure_file) as dp, _open(instant_file) as di, _open(accum_file) as da:
        dp = _drop_auxiliary(_normalize_time(dp))
        di = _normalize_time(di)
        da = _normalize_time(da)

        z500 = dp["z"].sel(pressure_level=500, drop=True).rename("z500")
        z850 = dp["z"].sel(pressure_level=850, drop=True).rename("z850")
        u850 = dp["u"].sel(pressure_level=850, drop=True).rename("u850")
        v850 = dp["v"].sel(pressure_level=850, drop=True).rename("v850")
        t850 = dp["t"].sel(pressure_level=850, drop=True).rename("t850")
        q850 = dp["q"].sel(pressure_level=850, drop=True).rename("q850")
        msl = di["msl"].rename("msl")
        tp = da["tp"].rename("tp")

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
        out = out.sortby("time")
        out = out.sortby("longitude")
        for v in out.data_vars:
            out[v] = out[v].astype("float32")
        return out


def save_dataset(ds: xr.Dataset, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite.")
        if output.is_dir():
            import shutil

            shutil.rmtree(output)
        else:
            output.unlink()

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".zarr":
        try:
            import zarr  # noqa: F401
        except Exception as exc:
            raise ImportError("zarr is required for .zarr output. Install `pip install zarr`.") from exc
        ds.to_zarr(str(output), mode="w")
    else:
        ds.to_netcdf(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LTG-Net ERA5 dataset from manually downloaded ERA5 NetCDF files."
    )
    parser.add_argument("--pressure", required=True, help="Pressure-level NetCDF (contains z/u/v/t/q).")
    parser.add_argument("--instant", required=True, help="Single-level instant NetCDF (contains msl).")
    parser.add_argument("--accum", required=True, help="Single-level accum NetCDF (contains tp).")
    parser.add_argument("--output", required=True, help="Output path (.nc or .zarr).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    args = parser.parse_args()

    ds = build_dataset(
        pressure_file=Path(args.pressure),
        instant_file=Path(args.instant),
        accum_file=Path(args.accum),
    )
    save_dataset(ds, output=Path(args.output), overwrite=bool(args.overwrite))
    print(f"Saved: {args.output}")
    print("Variables:", list(ds.data_vars))
    print("Dims:", dict(ds.sizes))
    print("Time:", str(ds.time.values[0]), "->", str(ds.time.values[-1]), "count", ds.sizes["time"])


if __name__ == "__main__":
    main()
