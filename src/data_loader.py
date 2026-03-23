"""Data access layer for the profiling float simulator.

All I/O — ocean velocity fields, bathymetry, and forecast perturbation —
lives here. No simulation logic belongs in this module.

Imports from src/types.py only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


def load_ocean_data(data_dir: Path) -> xr.Dataset:
    """Open all NetCDF tiles in *data_dir* as a single lazy Dataset.

    Files are combined along their coordinates so overlapping tiles are
    merged correctly. Only the velocity variables ``uo`` and ``vo`` are
    kept; everything else is dropped to reduce memory pressure.

    No data is loaded into memory — the returned dataset is fully lazy.

    Parameters
    ----------
    data_dir:
        Directory containing the tiled ``.nc`` files produced by
        ``CopernicusDataGetter``.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables ``uo`` and ``vo`` on a
        (time, depth, latitude, longitude) grid.
    """
    paths = sorted(data_dir.glob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"No .nc files found in {data_dir}")

    ds = xr.open_mfdataset(paths, combine="by_coords")
    drop = [v for v in ds.data_vars if v not in ("uo", "vo")]
    if drop:
        ds = ds.drop_vars(drop)
    return ds


def load_manifest(data_dir: Path) -> list[dict]:
    """Read the tile manifest produced by CopernicusDataGetter.

    Parameters
    ----------
    data_dir:
        Directory that contains ``manifest.yaml``.

    Returns
    -------
    list[dict]
        One dict per tile with keys ``file``, ``lat``, ``lon``, ``time``, etc.

    Raises
    ------
    FileNotFoundError
        If ``manifest.yaml`` is absent (data_getter has not been run yet).
    """
    path = data_dir / "manifest.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"manifest.yaml not found in {data_dir}. Run data_getter first."
        )
    with path.open() as fh:
        return yaml.safe_load(fh)


def select_tiles(
    manifest: list[dict],
    data_dir: Path,
    lat: float,
    lon: float,
    start_time: datetime,
    end_time: datetime,
    spatial_margin_deg: float = 0.5,
) -> xr.Dataset:
    """Open only the manifest tiles that overlap the requested spatiotemporal window.

    Parameters
    ----------
    manifest:
        List returned by :func:`load_manifest`.
    data_dir:
        Base directory for resolving tile file paths.
    lat:
        Centre latitude of the region of interest.
    lon:
        Centre longitude of the region of interest.
    start_time:
        Start of the time window.
    end_time:
        End of the time window.
    spatial_margin_deg:
        Half-width of the spatial selection box. Default 1.0°.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables ``uo`` and ``vo`` for the matching tiles.

    Raises
    ------
    RuntimeError
        If no tiles overlap the requested window.
    """
    paths = []
    for entry in manifest:
        lat_min, lat_max = entry["lat"]
        lon_min, lon_max = entry["lon"]
        t_start = datetime.fromisoformat(entry["time"][0])
        t_end = datetime.fromisoformat(entry["time"][1])

        if (
            lat_min <= lat + spatial_margin_deg
            and lat_max >= lat - spatial_margin_deg
            and lon_min <= lon + spatial_margin_deg
            and lon_max >= lon - spatial_margin_deg
            and t_start <= end_time
            and t_end >= start_time
        ):
            paths.append(data_dir / entry["file"])

    if not paths:
        raise RuntimeError(
            f"No tiles found for lat={lat:.3f} lon={lon:.3f} "
            f"time=[{start_time}, {end_time}] margin={spatial_margin_deg}°. "
            "Float may have left the dataset domain."
        )

    logger.debug("select_tiles: loading %d tile(s) for lat=%.3f lon=%.3f", len(paths), lat, lon)
    ds = xr.open_mfdataset(paths, combine="by_coords")
    drop = [v for v in ds.data_vars if v not in ("uo", "vo")]
    if drop:
        ds = ds.drop_vars(drop)
    return ds


def load_bathymetry(gebco_path: Path) -> xr.Dataset:
    """Open the GEBCO bathymetry NetCDF file lazily.

    GEBCO stores sea-floor elevation in a variable called ``elevation``,
    with negative values below sea level. The dataset is returned as-is;
    use :func:`get_bathymetry_depth` to convert to positive depth at a
    specific location.

    Parameters
    ----------
    gebco_path:
        Path to the GEBCO ``.nc`` file.

    Returns
    -------
    xr.Dataset
        Dataset containing at minimum the ``elevation`` variable on a
        (lat, lon) grid.
    """
    if not gebco_path.exists():
        raise FileNotFoundError(f"GEBCO file not found: {gebco_path}")
    return xr.open_dataset(gebco_path)


def build_velocity_interpolator(
    ds: xr.Dataset,
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    """Build fast scipy interpolators for uo and vo from an in-memory dataset.

    Converts the xarray working window into a pair of
    ``RegularGridInterpolator`` objects (one per velocity component) that
    can be queried thousands of times per dive cycle at negligible cost —
    far faster than calling ``xr.Dataset.interp`` on every timestep.

    Parameters
    ----------
    ds:
        Fully computed in-memory dataset from :func:`load_working_window`.

    Returns
    -------
    tuple[RegularGridInterpolator, RegularGridInterpolator]
        ``(interp_u, interp_v)`` — call each with a ``[[t_s, depth, lat, lon]]``
        array where *t_s* is seconds since the Unix epoch.
    """
    times  = ds.time.values.astype("datetime64[s]").astype(np.float64)
    depths = ds.depth.values.astype(np.float64)
    lats   = ds.latitude.values.astype(np.float64)
    lons   = ds.longitude.values.astype(np.float64)

    kw = dict(method="linear", bounds_error=False, fill_value=np.nan)
    interp_u = RegularGridInterpolator((times, depths, lats, lons), ds["uo"].values.astype(np.float64), **kw)
    interp_v = RegularGridInterpolator((times, depths, lats, lons), ds["vo"].values.astype(np.float64), **kw)
    return interp_u, interp_v


def build_bathymetry_interpolator(bathy_ds: xr.Dataset):
    """Build a fast scipy interpolator for seabed depth from the GEBCO dataset.

    Returns a callable ``f(lat, lon) -> depth_m`` that is orders of magnitude
    faster than calling xarray interp on every timestep.

    Parameters
    ----------
    bathy_ds:
        Dataset from :func:`load_bathymetry`.

    Returns
    -------
    callable
        ``f(lat, lon) -> float`` — seabed depth in metres, positive down.
        Returns ``0.0`` for land (positive GEBCO elevation).
    """
    lats = bathy_ds.lat.values.astype(np.float64)
    lons = bathy_ds.lon.values.astype(np.float64)
    elev = bathy_ds["elevation"].values.astype(np.float64)  # (lat, lon)

    interp = RegularGridInterpolator(
        (lats, lons), elev, method="linear", bounds_error=False, fill_value=np.nan,
    )

    def query(lat: float, lon: float) -> float:
        depth = float(interp([[lat, lon]])[0]) * -1.0
        return max(depth, 0.0)

    return query


def load_working_window(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    start_time: datetime,
    end_time: datetime,
    spatial_margin_deg: float = 1.0,
) -> xr.Dataset:
    """Slice the lazy dataset to a small spatiotemporal window and load it into memory.

    Call this once before each dive, then pass the result to
    :func:`build_velocity_interpolator` for fast per-step lookups.

    Parameters
    ----------
    ds:
        Full lazy ocean dataset from :func:`load_ocean_data`.
    lat:
        Centre latitude of the window in decimal degrees.
    lon:
        Centre longitude of the window in decimal degrees.
    start_time:
        Start of the time window (inclusive).
    end_time:
        End of the time window (inclusive).
    spatial_margin_deg:
        Degrees of padding added around *lat* / *lon* in each direction.
        Default 1.0°.

    Returns
    -------
    xr.Dataset
        Fully computed (in-memory) dataset covering the requested window.
    """
    window = ds.sel(
        latitude=slice(lat - spatial_margin_deg, lat + spatial_margin_deg),
        longitude=slice(lon - spatial_margin_deg, lon + spatial_margin_deg),
        time=slice(
            np.datetime64(start_time, "ns"),
            np.datetime64(end_time, "ns"),
        ),
    ).compute()

    logger.debug(
        "Working window loaded: lat=[%.2f, %.2f] lon=[%.2f, %.2f] time=[%s, %s] shape=%s",
        lat - spatial_margin_deg, lat + spatial_margin_deg,
        lon - spatial_margin_deg, lon + spatial_margin_deg,
        start_time, end_time,
        {v: window[v].shape for v in window.data_vars},
    )
    return window


def get_forecast_field(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    time: datetime,
    hours_ahead: float,
    noise_std: float,
    seed: int,
    spatial_margin_deg: float = 1.0,
) -> xr.Dataset:
    """Return a small, noisy forecast window centred on the current position.

    Called once at each surfacing event to give the control module a
    "forecast" to plan with. The spatial and temporal slicing mirrors
    :func:`load_working_window`; Gaussian noise is added to ``uo`` and
    ``vo`` to simulate forecast error.

    .. note::
        The noise model is intentionally simple (i.i.d. Gaussian, uniform
        across space, depth, and time). It is designed to be extended later
        with more physically realistic error models — e.g. spatially
        correlated noise (via a covariance kernel), depth-varying bias, or
        ensemble spread derived from NWP output.

    Parameters
    ----------
    ds:
        Full lazy ocean dataset from :func:`load_ocean_data`.
    lat:
        Current latitude in decimal degrees.
    lon:
        Current longitude in decimal degrees.
    time:
        Current UTC datetime (start of the forecast window).
    hours_ahead:
        Length of the forecast horizon in hours.
    noise_std:
        Standard deviation of the Gaussian noise added to velocity (m/s).
        Pass ``0.0`` for a perfect (noise-free) forecast.
    seed:
        Random seed for reproducibility of the noise field.
    spatial_margin_deg:
        Degrees of padding around *lat* / *lon*. Default 1.0°.

    Returns
    -------
    xr.Dataset
        Fully computed dataset covering the forecast window, with
        ``forecast_noise_std`` and ``forecast_noise_seed`` set as
        dataset-level attributes (even when ``noise_std == 0.0``).
    """
    end_time = time + timedelta(hours=hours_ahead)

    window = ds.sel(
        latitude=slice(lat - spatial_margin_deg, lat + spatial_margin_deg),
        longitude=slice(lon - spatial_margin_deg, lon + spatial_margin_deg),
        time=slice(
            np.datetime64(time, "ns"),
            np.datetime64(end_time, "ns"),
        ),
    ).compute()

    if noise_std == 0.0:
        window.attrs = {**window.attrs, "forecast_noise_std": 0.0, "forecast_noise_seed": seed}
        return window

    rng = np.random.default_rng(seed)

    uo_base = window["uo"]
    vo_base = window["vo"]

    noise_u = xr.DataArray(
        rng.normal(loc=0.0, scale=noise_std, size=uo_base.shape).astype(np.float32),
        coords=uo_base.coords,
        dims=uo_base.dims,
        attrs=uo_base.attrs,
    )
    noise_v = xr.DataArray(
        rng.normal(loc=0.0, scale=noise_std, size=vo_base.shape).astype(np.float32),
        coords=vo_base.coords,
        dims=vo_base.dims,
        attrs=vo_base.attrs,
    )

    noisy = window.assign(uo=uo_base + noise_u, vo=vo_base + noise_v)
    noisy.attrs = {
        **window.attrs,
        "forecast_noise_std": noise_std,
        "forecast_noise_seed": seed,
    }
    return noisy
