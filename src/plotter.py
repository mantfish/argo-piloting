"""Visualisation for the profiling float simulator.

Reads trajectory data from parquet files — never from simulation objects
directly. All plots are produced with matplotlib.

Static output
-------------
:func:`plot_trajectory` produces a two-panel figure:
- Left: a map view of the float track, line-coloured by phase.
  Uses cartopy (PlateCarree + coastlines) when available, falling back
  to plain lat/lon axes otherwise.
- Right: depth vs time, y-axis inverted so the surface is at the top.

Animation
---------
:func:`animate_trajectory` is not yet implemented. It will animate the
float track overlaid on the current velocity field, with one frame per
surfacing event.

Imports: src/types.py only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from sim_types import SimConfig

# ---------------------------------------------------------------------------
# Phase colour map
# ---------------------------------------------------------------------------

_PHASE_COLOURS: dict[str, str] = {
    "ascending":   "cornflowerblue",
    "descending":  "tomato",
    "parking":     "gold",
    "on_seabed":   "saddlebrown",
    "at_surface":  "limegreen",
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_trajectory(path: Path) -> pd.DataFrame:
    """Read a trajectory parquet file and return it as a DataFrame.

    The ``time`` column is parsed as datetime if it is not already.

    Parameters
    ----------
    path:
        Path to the ``.parquet`` file written by ``run_simulation()``.

    Returns
    -------
    pd.DataFrame
        Trajectory with columns: time, lat, lon, depth, phase, u, v,
        bathymetry_depth, on_seabed.
    """
    df = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])
    return df


def plot_trajectory(
    df: pd.DataFrame,
    config: SimConfig,
    save_path: Path | None = None,
    show: bool = True,
    bathy_ds: xr.Dataset | None = None,
) -> plt.Figure:
    """Produce a two-panel static map of the float trajectory.

    Left panel shows the spatial track coloured by phase, optionally
    underlaid with a shaded bathymetry map; right panel shows depth vs
    time coloured by the same scheme.

    Parameters
    ----------
    df:
        Trajectory DataFrame from :func:`load_trajectory` or directly
        from ``run_simulation()``.
    config:
        Simulation config — used for the figure title metadata.
    save_path:
        If given, the figure is saved here at 150 dpi.
    show:
        If ``True``, call ``plt.show()`` before returning.
    bathy_ds:
        Optional GEBCO bathymetry dataset from
        :func:`~data_loader.load_bathymetry`. When provided, ocean
        depth is shaded on the map panel.

    Returns
    -------
    plt.Figure
        The completed figure object.
    """
    # ------------------------------------------------------------------
    # Figure and axes setup
    # ------------------------------------------------------------------
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(16, 6))
        ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax_depth = fig.add_subplot(1, 2, 2)
    else:
        fig, (ax_map, ax_depth) = plt.subplots(1, 2, figsize=(16, 6))

    # ------------------------------------------------------------------
    # Left panel — map view
    # ------------------------------------------------------------------
    _draw_map(ax_map, df, bathy_ds=bathy_ds)

    # ------------------------------------------------------------------
    # Right panel — depth vs time
    # ------------------------------------------------------------------
    _draw_depth_profile(ax_depth, df)

    # ------------------------------------------------------------------
    # Figure title with run metadata
    # ------------------------------------------------------------------
    title = (
        f"Strategy: {config.control_strategy}\n"
        f"Start: {config.start_state.time:%Y-%m-%d %H:%M}  |  "
        f"Lat: {config.start_state.lat:.3f}  "
        f"Lon: {config.start_state.lon:.3f}\n"
        f"Forecast noise \u03c3: {config.forecast_noise_std} m/s  |  "
        f"Seed: {config.forecast_noise_seed}"
    )
    fig.suptitle(title, fontsize=10, y=1.02, va="bottom")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig



def animate_trajectory(
    df: pd.DataFrame,
    ds: xr.Dataset,
    config: SimConfig,
    save_path: Path | None = None,
) -> None:
    """Animate the float trajectory overlaid on the current velocity field.

    Each frame corresponds to one surfacing event. The animation will
    show the float track building up over time, with background arrows
    (quiver) drawn from the ``uo`` / ``vo`` fields in *ds* at the
    matching time slice.

    Not yet implemented.
    """
    raise NotImplementedError("Animation coming in a later prompt.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _colour_segments(ax, x, y, phases: pd.Series, transform=None) -> None:
    """Draw a multi-coloured line by plotting each phase segment separately."""
    for phase, colour in _PHASE_COLOURS.items():
        mask = phases == phase
        if not mask.any():
            continue
        # Plot connected segments within this phase by inserting NaN breaks
        # wherever the phase is interrupted.
        xs = np.where(mask, x, np.nan)
        ys = np.where(mask, y, np.nan)
        kwargs: dict = dict(color=colour, linewidth=1.5)
        if transform is not None:
            kwargs["transform"] = transform
        ax.plot(xs, ys, **kwargs)


def _phase_legend_handles() -> list[mlines.Line2D]:
    """Return a list of legend handles for the phase colour map."""
    return [
        mlines.Line2D([], [], color=colour, linewidth=2, label=phase)
        for phase, colour in _PHASE_COLOURS.items()
    ]


def _draw_bathymetry(ax, bathy_ds: xr.Dataset, traj_lats: np.ndarray, traj_lons: np.ndarray,
                     margin: float = 1.0) -> None:
    """Shade ocean depth behind the trajectory, with a colorbar."""
    lat_min = traj_lats.min() - margin
    lat_max = traj_lats.max() + margin
    lon_min = traj_lons.min() - margin
    lon_max = traj_lons.max() + margin

    sub = bathy_ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )
    # Downsample to keep rendering fast (GEBCO is ~0.001° resolution).
    step = max(1, len(sub.lat) // 400)
    sub = sub.isel(lat=slice(None, None, step), lon=slice(None, None, step))

    elev = sub["elevation"].values.astype(np.float32)
    lats = sub.lat.values
    lons = sub.lon.values

    # Ocean depth = positive values where elevation < 0; land → NaN.
    depth = np.where(elev < 0, -elev, np.nan)
    vmax = float(np.nanmax(depth)) if not np.all(np.isnan(depth)) else 1.0

    cmap = plt.cm.Blues
    kwargs: dict = dict(cmap=cmap, vmin=0, vmax=vmax, zorder=0, alpha=0.85, shading="auto")
    if HAS_CARTOPY:
        import cartopy.crs as _ccrs
        mesh = ax.pcolormesh(lons, lats, depth, transform=_ccrs.PlateCarree(), **kwargs)
    else:
        mesh = ax.pcolormesh(lons, lats, depth, **kwargs)

    plt.colorbar(mesh, ax=ax, label="Depth (m)", fraction=0.025, pad=0.04)

    # Depth contours for orientation.
    contour_levels = [200, 500, 1000, 2000]
    valid = [lv for lv in contour_levels if lv < vmax]
    if valid:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        ct_kwargs: dict = dict(levels=valid, colors="steelblue", linewidths=0.4, alpha=0.6)
        if HAS_CARTOPY:
            ax.contour(lon_grid, lat_grid, depth, transform=_ccrs.PlateCarree(), **ct_kwargs)
        else:
            ax.contour(lon_grid, lat_grid, depth, **ct_kwargs)


def _draw_map(ax, df: pd.DataFrame, bathy_ds: xr.Dataset | None = None) -> None:
    """Draw the spatial trajectory on *ax*, with cartopy if available."""
    lons = df["lon"].to_numpy()
    lats = df["lat"].to_numpy()
    phases = df["phase"]

    if bathy_ds is not None:
        _draw_bathymetry(ax, bathy_ds, lats, lons)

    if HAS_CARTOPY:
        ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=1)
        ax.coastlines(resolution="10m", linewidth=0.6, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="grey",
                          alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        transform = ccrs.PlateCarree()
        _colour_segments(ax, lons, lats, phases, transform=transform)
        ax.plot(lons[0],  lats[0],  marker="*", markersize=14,
                color="black", transform=transform, zorder=5, label="Start")
        ax.plot(lons[-1], lats[-1], marker="s", markersize=10,
                color="black", transform=transform, zorder=5, label="End")
    else:
        _colour_segments(ax, lons, lats, phases)
        ax.plot(lons[0],  lats[0],  marker="*", markersize=14,
                color="black", zorder=5, label="Start")
        ax.plot(lons[-1], lats[-1], marker="s", markersize=10,
                color="black", zorder=5, label="End")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

    handles = _phase_legend_handles() + [
        mlines.Line2D([], [], marker="*", color="black",
                      linestyle="None", markersize=10, label="Start"),
        mlines.Line2D([], [], marker="s", color="black",
                      linestyle="None", markersize=8,  label="End"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="best")
    ax.set_title("Float trajectory")


def _draw_depth_profile(ax, df: pd.DataFrame) -> None:
    """Draw depth vs time on *ax*, y-axis inverted (surface at top)."""
    # Convert datetimes to matplotlib float dates so that _colour_segments
    # can safely insert np.nan as break markers (np.nan cannot go into a
    # datetime64 array and would corrupt or crash the plot).
    times_float = mdates.date2num(pd.to_datetime(df["time"]))
    depths = df["depth"].to_numpy()
    phases = df["phase"]

    _colour_segments(ax, times_float, depths, phases)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    ax.axhline(y=0.0, color="grey", linestyle="--", linewidth=1.0, label="Surface")
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_ylabel("Depth (m)")
    ax.tick_params(axis="x", rotation=45)

    handles = _phase_legend_handles() + [
        mlines.Line2D([], [], color="grey", linestyle="--",
                      linewidth=1.5, label="Surface"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="best")
    ax.set_title("Depth profile")
