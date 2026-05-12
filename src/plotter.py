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
:func:`animate_trajectory` saves an MP4 of the track building up over time.
Each frame advances by a configurable number of records; the title shows the
current simulation date, phase, and position.

CLI usage::

    python plotter.py <results_dir> <output.mp4> [--fps 10] [--step N]

``results_dir`` must contain exactly one ``.parquet`` trajectory file.

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

from sim_types import EKFRecord, SimConfig

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
        f"Lat: {config.start_state.location.lat:.3f}  "
        f"Lon: {config.start_state.location.lon:.3f}\n"
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



def plot_ekf_statistics(
    ekf_records: list[EKFRecord],
    Q: np.ndarray,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Produce a three-panel EKF diagnostic figure.

    Panel 1 — Covariance evolution
        Sawtooth of position variance trace(P[:2,:2]) across cycles. Each
        bar shows how much uncertainty grew during the dive (peak = just
        before GPS fix). After each fix P collapses to zero, so the plot
        shows the worst-case uncertainty at the moment of surfacing.

    Panel 2 — Innovation time series
        Innovations v_k = GPS_k − H x̂_{k|k−1} in x (east) and y (north),
        overlaid with ±2σ bounds derived from P_{k|k−1}. For a well-tuned
        filter, ~95 % of innovations should fall within these bounds.
        Running mean is plotted to expose any systematic bias.

    Panel 3 — Normalised Innovation Squared (NIS)
        NIS_k = vᵀ S⁻¹ v where S = P[:2,:2].  For a 2-D state with R=0
        this should follow χ²(2) with expected value 2.  The plot shows
        NIS per cycle alongside the χ²(2) 95 % consistency band [0.10, 7.38]
        and the theoretical mean.  Values persistently above the band mean
        Q is too small (filter overconfident); below means Q is too large.

    Parameters
    ----------
    ekf_records:
        List of :class:`EKFRecord` snapshots collected at each surfacing.
    Q:
        (4, 4) process noise covariance used during the run (for annotation).
    save_path:
        If given, saves the figure here at 150 dpi.
    show:
        If True, calls ``plt.show()`` before returning.
    """
    if not ekf_records:
        raise ValueError("ekf_records is empty — nothing to plot.")

    times  = [r.time for r in ekf_records]
    cycles = [r.cycle for r in ekf_records]
    vx     = np.array([r.innovation_x for r in ekf_records])
    vy     = np.array([r.innovation_y for r in ekf_records])
    Pxx    = np.array([r.P_xx for r in ekf_records])
    Pyy    = np.array([r.P_yy for r in ekf_records])
    trace_P = Pxx + Pyy

    # ------------------------------------------------------------------
    # Figure layout — 3 rows
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    ax1, ax2, ax3 = axes

    # ------------------------------------------------------------------
    # Panel 1: covariance sawtooth
    # ------------------------------------------------------------------
    # Build sawtooth: for each cycle, trace rises from 0 at the start of the
    # dive to trace_P[k] at the moment of surfacing, then drops to 0 again.
    saw_t: list = []
    saw_v: list = []
    for i, (t, tr) in enumerate(zip(times, trace_P)):
        if i > 0:
            saw_t.append(t)
            saw_v.append(0.0)   # immediately after previous GPS update
        saw_t.append(t)
        saw_v.append(tr)        # peak just before this GPS update

    ax1.plot(saw_t, [v / 1e6 for v in saw_v], color="steelblue", linewidth=1.5)
    ax1.fill_between(saw_t, [v / 1e6 for v in saw_v], alpha=0.18, color="steelblue")
    ax1.scatter(times, trace_P / 1e6, color="steelblue", s=30, zorder=5,
                label="peak trace(P) at surfacing")
    ax1.set_ylabel("trace(P[:2,:2])  (km²)")
    ax1.set_xlabel("")
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(fontsize=8)
    ax1.set_title("Covariance evolution — position uncertainty grows during dive, collapses at GPS fix")

    # ------------------------------------------------------------------
    # Panel 2: innovation time series with ±2σ and running mean
    # ------------------------------------------------------------------
    sigma_x = np.sqrt(np.maximum(Pxx, 0.0))
    sigma_y = np.sqrt(np.maximum(Pyy, 0.0))

    ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax2.plot(cycles, vx / 1e3, color="tab:blue",   marker="o", markersize=4,
             linewidth=1.0, label="v_x (east)")
    ax2.plot(cycles, vy / 1e3, color="tab:orange", marker="s", markersize=4,
             linewidth=1.0, label="v_y (north)")
    ax2.fill_between(cycles, -2 * sigma_x / 1e3, 2 * sigma_x / 1e3,
                     alpha=0.15, color="tab:blue",   label="±2σ_x from P")
    ax2.fill_between(cycles, -2 * sigma_y / 1e3, 2 * sigma_y / 1e3,
                     alpha=0.12, color="tab:orange", label="±2σ_y from P")

    # Running mean (cumulative)
    cum_mean_x = np.cumsum(vx) / np.arange(1, len(vx) + 1)
    cum_mean_y = np.cumsum(vy) / np.arange(1, len(vy) + 1)
    ax2.plot(cycles, cum_mean_x / 1e3, color="tab:blue",   linewidth=2.0,
             linestyle="--", label="running mean v_x")
    ax2.plot(cycles, cum_mean_y / 1e3, color="tab:orange", linewidth=2.0,
             linestyle="--", label="running mean v_y")

    ax2.set_ylabel("Innovation  (km)")
    ax2.set_xlabel("Cycle")
    ax2.legend(fontsize=7, ncol=3)
    ax2.set_title(
        "Innovation v = GPS − H x̂  |  "
        f"overall mean: ({np.mean(vx)/1e3:+.2f}, {np.mean(vy)/1e3:+.2f}) km"
    )

    # ------------------------------------------------------------------
    # Panel 3: Normalised Innovation Squared (NIS) vs χ²(2)
    # ------------------------------------------------------------------
    # NIS_k = v^T S^{-1} v  where  S = diag(Pxx, Pyy)  (off-diag pos cov
    # is zero after GPS reset, and we have independent x/y here)
    with np.errstate(divide="ignore", invalid="ignore"):
        nis = np.where(
            (Pxx > 0) & (Pyy > 0),
            vx ** 2 / Pxx + vy ** 2 / Pyy,
            np.nan,
        )

    chi2_mean = 2.0               # expected value for χ²(2)
    chi2_lo   = 0.1026            # 2.5th percentile χ²(2)
    chi2_hi   = 7.3778            # 97.5th percentile χ²(2)

    ax3.axhline(chi2_mean, color="green",  linewidth=1.5, linestyle="-",
                label=f"χ²(2) mean = {chi2_mean}")
    ax3.axhline(chi2_lo,   color="green",  linewidth=1.0, linestyle="--",
                label=f"95 % band [{chi2_lo:.2f}, {chi2_hi:.2f}]")
    ax3.axhline(chi2_hi,   color="green",  linewidth=1.0, linestyle="--")
    ax3.fill_between([cycles[0], cycles[-1]], chi2_lo, chi2_hi,
                     color="green", alpha=0.08)

    ax3.plot(cycles, nis, color="tab:red", marker="o", markersize=4,
             linewidth=1.0, label="NIS per cycle")

    # Rolling mean over a 5-cycle window
    if len(nis) >= 3:
        w = min(5, len(nis))
        rolling = np.convolve(np.nan_to_num(nis), np.ones(w) / w, mode="valid")
        ax3.plot(cycles[w - 1:], rolling, color="darkred", linewidth=2.0,
                 linestyle="--", label=f"{w}-cycle rolling mean")

    overall_mean_nis = float(np.nanmean(nis))
    ax3.set_ylabel("NIS = vᵀ S⁻¹ v")
    ax3.set_xlabel("Cycle")
    ax3.legend(fontsize=8)
    ax3.set_title(
        f"Normalised Innovation Squared  |  "
        f"mean NIS = {overall_mean_nis:.2f}  (target ≈ 2.0)  |  "
        f"σ_pos = {np.sqrt(Q[0, 0]):.1f} m/√hr"
    )

    # Annotate tuning guidance
    if overall_mean_nis > chi2_hi:
        guidance = "NIS > band → Q too small (filter over-confident), increase σ_pos"
    elif overall_mean_nis < chi2_lo:
        guidance = "NIS < band → Q too large (filter under-confident), decrease σ_pos"
    else:
        guidance = "NIS within band → filter consistent with Q"
    ax3.annotate(guidance, xy=(0.02, 0.95), xycoords="axes fraction",
                 fontsize=8, va="top",
                 color="green" if chi2_lo <= overall_mean_nis <= chi2_hi else "tab:red")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def animate_trajectory(
    df: pd.DataFrame,
    save_path: Path,
    fps: int = 10,
    step: int | None = None,
    bathy_ds: xr.Dataset | None = None,
) -> None:
    """Animate the float trajectory and save as an MP4 video.

    The track builds up frame-by-frame, coloured by phase. The title of
    each frame shows the current simulation date, phase, and position.
    Requires ``ffmpeg`` to be installed on the system.

    Parameters
    ----------
    df:
        Trajectory DataFrame from :func:`load_trajectory`.
    save_path:
        Output ``.mp4`` path.
    fps:
        Frames per second in the output video. Default 10.
    step:
        Records to advance per frame. Defaults to ``len(df) // 200`` so
        the video is roughly 200 frames regardless of trajectory length.
    bathy_ds:
        Optional GEBCO bathymetry dataset for depth shading behind the track.
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    if step is None:
        step = max(1, len(df) // 200)

    lons = df["lon"].to_numpy()
    lats = df["lat"].to_numpy()
    times = pd.to_datetime(df["time"])
    phases = df["phase"]

    frame_starts = list(range(0, len(df), step))
    n_frames = len(frame_starts)

    # ------------------------------------------------------------------
    # Figure and axes
    # ------------------------------------------------------------------
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=1)
        ax.coastlines(resolution="10m", linewidth=0.6, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="grey",
                          alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        transform = None

    kw: dict = dict(transform=transform) if transform else {}

    # ------------------------------------------------------------------
    # Static elements — drawn once before animation starts
    # ------------------------------------------------------------------
    if bathy_ds is not None:
        _draw_bathymetry(ax, bathy_ds, lats, lons)

    ax.plot(lons[0], lats[0], marker="*", markersize=14, color="black", zorder=5, **kw)

    # Fix the view over the full trajectory so the frame never jumps
    pad = 0.5
    if transform:
        ax.set_extent(
            [lons.min() - pad, lons.max() + pad, lats.min() - pad, lats.max() + pad],
            crs=transform,
        )
    else:
        ax.set_xlim(lons.min() - pad, lons.max() + pad)
        ax.set_ylim(lats.min() - pad, lats.max() + pad)
        ax.set_aspect("equal")

    current_dot, = ax.plot([], [], marker="o", markersize=10, color="black", zorder=7, **kw)

    handles = _phase_legend_handles() + [
        mlines.Line2D([], [], marker="*", color="black",
                      linestyle="None", markersize=10, label="Start"),
        mlines.Line2D([], [], marker="o", color="black",
                      linestyle="None", markersize=8,  label="Current"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="best")
    title = ax.set_title("")

    # ------------------------------------------------------------------
    # Per-frame update — appends each new track segment
    # ------------------------------------------------------------------
    def _update(frame_i: int):
        start = frame_starts[frame_i]
        end = min(start + step, len(df))

        _colour_segments(
            ax,
            df["lon"].to_numpy()[start:end],
            df["lat"].to_numpy()[start:end],
            phases.iloc[start:end],
            transform=transform,
        )

        current_dot.set_data([lons[end - 1]], [lats[end - 1]])

        title.set_text(
            f"{times.iloc[end - 1]:%Y-%m-%d %H:%M}  |  {phases.iloc[end - 1]}  |  "
            f"{lats[end - 1]:.3f}°N  {lons[end - 1]:.3f}°E"
        )
        return [current_dot, title]

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000 // fps, blit=False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Profiler trajectory"})
    anim.save(str(save_path), writer=writer)
    plt.close(fig)
    print(f"Saved {n_frames} frames → {save_path}")


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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser(
        description="Animate a profiler simulation trajectory to MP4.",
    )
    parser.add_argument(
        "results_dir", type=Path,
        help="Folder containing the trajectory .parquet file.",
    )
    parser.add_argument(
        "output", type=Path,
        help="Output video path, e.g. trajectory.mp4",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second (default: 10)",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Records per frame (default: auto, ~200 frames total)",
    )
    args = parser.parse_args()

    parquet_files = sorted(args.results_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)
    if len(parquet_files) > 1:
        print(f"Multiple .parquet files found, using: {parquet_files[0].name}")

    trajectory_df = load_trajectory(parquet_files[0])
    animate_trajectory(trajectory_df, save_path=args.output, fps=args.fps, step=args.step)
