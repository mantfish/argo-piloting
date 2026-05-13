"""
Generate a complete synthetic test data directory for the profiling float simulator.

Produces:
  1. One or more CMEMS-format .nc velocity tiles (uo, vo)
  2. A manifest.yaml matching what data_loader.load_manifest() expects
  3. A synthetic GEBCO-format bathymetry file (D6_2024.nc)

After running, you can point main.py's data_dir at the output directory.

Usage:
    python generate_test_data.py --case uniform_east --output test_data/
    python generate_test_data.py --case rotating --output test_data/
    python generate_test_data.py --case shear_ns --output test_data/

Then in main.py:
    config = SimConfig(..., data_dir=Path("test_data"), ...)
"""

import argparse
import numpy as np
import xarray as xr
import yaml
from pathlib import Path


# ──────────────────────────────────────────────────────────────────
# Grid generation
# ──────────────────────────────────────────────────────────────────

def make_grid(
        lat_range=(54.0, 57.0),
        lon_range=(12.0, 18.0),
        depth_levels=(0.494025, 1.541375, 2.645669, 3.819495, 5.078224,
                      6.440614, 7.929560, 9.572997, 11.405000, 13.467137),
        hours=240,  # 10 days
        start_time="2025-01-15T00:00:00",
        spatial_res=0.028,  # ~1/36° like CMEMS Baltic
):
    """Create coordinate arrays matching CMEMS Baltic format.

    The depth levels are taken from the actual BALTICSEA_ANALYSISFORECAST_PHY_003_006
    product to ensure the interpolator works correctly with realistic depth values.
    """
    lats = np.arange(lat_range[0], lat_range[1], spatial_res).astype(np.float64)
    lons = np.arange(lon_range[0], lon_range[1], spatial_res).astype(np.float64)
    depths = np.array(depth_levels, dtype=np.float32)
    t0 = np.datetime64(start_time)
    times = np.array([t0 + np.timedelta64(h, "h") for h in range(hours)])
    return times, depths, lats, lons


# ──────────────────────────────────────────────────────────────────
# Current field test cases
# ──────────────────────────────────────────────────────────────────

def case_uniform_east(times, depths, lats, lons, speed=0.1):
    """Constant 0.1 m/s eastward everywhere, all depths.

    Verification:
      After 24h at surface: dx ≈ 8640m east, dy = 0
      At 55°N: dx ≈ 0.135° longitude
    """
    shape = (len(times), len(depths), len(lats), len(lons))
    uo = np.full(shape, speed, dtype=np.float32)
    vo = np.zeros(shape, dtype=np.float32)
    return uo, vo


def case_uniform_north(times, depths, lats, lons, speed=0.1):
    """Constant 0.1 m/s northward everywhere."""
    shape = (len(times), len(depths), len(lats), len(lons))
    uo = np.zeros(shape, dtype=np.float32)
    vo = np.full(shape, speed, dtype=np.float32)
    return uo, vo


def case_rotating(times, depths, lats, lons, speed=0.1, period_hours=24):
    """Uniform field that rotates with time. Full rotation in period_hours.

    Verification:
      After one full period, net displacement ≈ 0
      Trajectory should be roughly circular
    """
    shape = (len(times), len(depths), len(lats), len(lons))
    uo = np.zeros(shape, dtype=np.float32)
    vo = np.zeros(shape, dtype=np.float32)
    for i in range(len(times)):
        angle = 2 * np.pi * i / period_hours
        uo[i, :, :, :] = speed * np.cos(angle)
        vo[i, :, :, :] = speed * np.sin(angle)
    return uo, vo


def case_shear_ns(times, depths, lats, lons, max_speed=0.15):
    """Eastward velocity increases linearly with latitude. vo = 0.

    Verification:
      Float at southern edge: barely moves
      Float at northern edge: moves at max_speed eastward
    """
    shape = (len(times), len(depths), len(lats), len(lons))
    lat_frac = (lats - lats.min()) / (lats.max() - lats.min())
    uo = np.zeros(shape, dtype=np.float32)
    for i_lat, frac in enumerate(lat_frac):
        uo[:, :, i_lat, :] = max_speed * frac
    vo = np.zeros(shape, dtype=np.float32)
    return uo, vo


def case_surface_only(times, depths, lats, lons, speed=0.1):
    """Current only at the surface (first depth level); zero below.

    Verification:
      Float drifts only during surface phase
      Parking at depth = no horizontal motion
      This is the key test for the control strategy!
    """
    shape = (len(times), len(depths), len(lats), len(lons))
    uo = np.zeros(shape, dtype=np.float32)
    vo = np.zeros(shape, dtype=np.float32)
    # Only the shallowest depth level gets current
    uo[:, 0, :, :] = speed
    return uo, vo


CASES = {
    "uniform_east": case_uniform_east,
    "uniform_north": case_uniform_north,
    "rotating": case_rotating,
    "shear_ns": case_shear_ns,
    "surface_only": case_surface_only,
}


# ──────────────────────────────────────────────────────────────────
# Builders
# ──────────────────────────────────────────────────────────────────

def build_velocity_dataset(uo, vo, times, depths, lats, lons):
    """Assemble velocity arrays into an xarray Dataset matching CMEMS conventions."""
    return xr.Dataset(
        {
            "uo": (["time", "depth", "latitude", "longitude"], uo, {
                "standard_name": "eastward_sea_water_velocity",
                "long_name": "Eastward velocity",
                "units": "m s-1",
            }),
            "vo": (["time", "depth", "latitude", "longitude"], vo, {
                "standard_name": "northward_sea_water_velocity",
                "long_name": "Northward velocity",
                "units": "m s-1",
            }),
        },
        coords={
            "time": times,
            "depth": ("depth", depths, {"units": "m", "positive": "down"}),
            "latitude": ("latitude", lats, {"units": "degrees_north"}),
            "longitude": ("longitude", lons, {"units": "degrees_east"}),
        },
        attrs={
            "title": "SYNTHETIC TEST DATA — NOT REAL OCEAN DATA",
            "source": "generate_test_data.py",
            "Conventions": "CF-1.6",
        },
    )


def build_bathymetry_dataset(lats, lons, flat_depth=100.0):
    """Create a synthetic GEBCO-format bathymetry file.

    GEBCO stores elevation (negative = below sea level).
    A flat seabed at 100m means elevation = -100 everywhere.

    Parameters
    ----------
    flat_depth:
        Uniform seabed depth in metres. Default 100m — deep enough
        that the float won't hit bottom in most test configs,
        but shallow enough that park_on_bottom is fast.
    """
    elevation = np.full((len(lats), len(lons)), -flat_depth, dtype=np.float32)

    return xr.Dataset(
        {
            "elevation": (["lat", "lon"], elevation, {
                "standard_name": "height_above_reference_ellipsoid",
                "long_name": "Elevation relative to sea level",
                "units": "m",
            }),
        },
        coords={
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={
            "title": "SYNTHETIC BATHYMETRY — NOT REAL GEBCO DATA",
            "source": "generate_test_data.py",
        },
    )


def build_manifest(nc_filename, lats, lons, times):
    """Create a manifest.yaml entry matching what data_loader.load_manifest() expects."""
    return [{
        "file": nc_filename,
        "lat": [float(lats[0]), float(lats[-1])],
        "lon": [float(lons[0]), float(lons[-1])],
        "time": [
            str(times[0]),
            str(times[-1]),
        ],
    }]


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate complete synthetic test data for the float simulator"
    )
    parser.add_argument("--case", choices=list(CASES.keys()), default="uniform_east",
                        help="Current field test case")
    parser.add_argument("--output", type=str, default="test_data",
                        help="Output directory (will be created)")
    parser.add_argument("--hours", type=int, default=240,
                        help="Duration in hours (default: 240 = 10 days)")
    parser.add_argument("--speed", type=float, default=None,
                        help="Override default speed (m/s)")
    parser.add_argument("--seabed-depth", type=float, default=100.0,
                        help="Flat seabed depth in metres (default: 100)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate grid ---
    times, depths, lats, lons = make_grid(hours=args.hours)

    # --- Generate velocity field ---
    case_fn = CASES[args.case]
    kwargs = {}
    if args.speed is not None:
        kwargs["speed"] = args.speed
    uo, vo = case_fn(times, depths, lats, lons, **kwargs)

    # --- Write velocity NetCDF ---
    nc_filename = f"synthetic_{args.case}_{args.hours}h.nc"
    ds = build_velocity_dataset(uo, vo, times, depths, lats, lons)
    ds.to_netcdf(out_dir / nc_filename)
    print(f"  Velocity file: {out_dir / nc_filename}")
    print(f"    Shape: time={len(times)}, depth={len(depths)}, lat={len(lats)}, lon={len(lons)}")
    print(f"    uo range: [{uo.min():.4f}, {uo.max():.4f}] m/s")
    print(f"    vo range: [{vo.min():.4f}, {vo.max():.4f}] m/s")

    # --- Write manifest ---
    manifest = build_manifest(nc_filename, lats, lons, times)
    manifest_path = out_dir / "manifest.yaml"
    with manifest_path.open("w") as fh:
        yaml.dump(manifest, fh, default_flow_style=False)
    print(f"  Manifest: {manifest_path}")

    # --- Write bathymetry ---
    bathy_ds = build_bathymetry_dataset(lats, lons, flat_depth=args.seabed_depth)
    bathy_path = out_dir / "D6_2024.nc"
    bathy_ds.to_netcdf(bathy_path)
    print(f"  Bathymetry: {bathy_path} (flat seabed at {args.seabed_depth}m)")

    # --- Print expected displacements for easy verification ---
    s = args.speed if args.speed else 0.1
    lat_mid = (lats[0] + lats[-1]) / 2
    dx_m_24h = s * 86400
    print(f"\n  Reference displacements at {s} m/s:")
    print(f"    24h east:  {dx_m_24h:.0f} m = {dx_m_24h / (111320 * np.cos(np.radians(lat_mid))):.4f}° lon")
    print(f"    24h north: {dx_m_24h:.0f} m = {dx_m_24h / 111320:.4f}° lat")
    print(f"\n  Ready! Point your SimConfig.data_dir at '{out_dir}'")


if __name__ == "__main__":
    main()