"""Single-cycle integrator for a profiling float.

Advances a float from the moment it starts descending until it returns to
the surface, recording the full trajectory. All ocean data access is
delegated to data_loader; all type definitions come from types.

Imports: src/types.py, src/data_loader.py only.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from data_loader import build_bathymetry_interpolator, build_velocity_interpolator, load_working_window, select_tiles
from sim_types import ControlAction, Phase, ProfilerState, TrajectoryRecord

logger = logging.getLogger(__name__)

_MAX_SIMULATED_DAYS = 365


def run_until_surface(
    state: ProfilerState,
    data_dir: Path,
    manifest: list[dict],
    bathy_interp,
    action: ControlAction,
    dt_seconds: float = 60.0,
    spatial_margin_deg: float = 0.5,
    reload_margin_deg: float = 0.2,
) -> tuple[list[TrajectoryRecord], ProfilerState]:
    """Step a profiling float through one full dive cycle.

    Receives a state with ``phase="at_surface"`` (surface waiting time has
    already elapsed before this call). Integrates forward using forward
    Euler until the float re-surfaces, then returns.

    Parameters
    ----------
    state:
        Current profiler state. Must have ``phase="at_surface"``.
    data_dir:
        Directory containing tile NetCDF files.
    manifest:
        Tile manifest from :func:`~data_loader.load_manifest`.
    bathy_ds:
        GEBCO bathymetry dataset from
        :func:`~data_loader.load_bathymetry`.
    action:
        Control instruction for this cycle (speeds, park mode, depths,
        cycle duration).
    dt_seconds:
        Integration timestep in seconds. Default 600 s (10 min).
    spatial_margin_deg:
        Spatial padding in degrees used for tile selection and working
        window slicing. Default 1.0°.
    reload_margin_deg:
        Trigger a tile reload when the float comes within this many
        degrees of the loaded window edge. Default 0.2°.

    Returns
    -------
    tuple[list[TrajectoryRecord], ProfilerState]
        ``(records, final_state)`` where *records* is the full trajectory
        for this cycle and *final_state* has ``phase="at_surface"``.

    Raises
    ------
    RuntimeError
        If the float has not surfaced after
        ``_MAX_SIMULATED_DAYS`` of simulated time (catches runaway loops).
    """
    if state.phase != "at_surface":
        raise ValueError(f"run_until_surface expects phase='at_surface', got {state.phase!r}")

    time = state.time
    lat = state.lat
    lon = state.lon
    depth = state.depth
    x = state.x
    y = state.y

    phase: Phase = "descending"
    dive_start = time
    ascent_start_time: datetime | None = None

    # Estimate the dive window end time so we can load a working window
    # into memory once upfront, avoiding repeated lazy reads during the loop.
    max_depth_m = (
        action.target_depth
        if (action.park_mode == "parking_depth" and action.target_depth is not None)
        else 500.0  # conservative upper bound for park_on_bottom
    )
    window_end = (
        dive_start
        + timedelta(hours=action.cycle_hours)
        + timedelta(seconds=(max_depth_m / action.descent_speed_ms) + (max_depth_m / action.ascent_speed_ms))
        + timedelta(hours=6)  # safety buffer
    )
    lazy_ds = select_tiles(manifest, data_dir, lat, lon, dive_start, window_end, spatial_margin_deg)
    working_ds = load_working_window(lazy_ds, lat, lon, start_time=dive_start, end_time=window_end,
                                     spatial_margin_deg=spatial_margin_deg)
    interp_u, interp_v = build_velocity_interpolator(working_ds)

    # Inner boundary: trigger reload before reaching the window edge.
    def _win_bounds(clat, clon):
        m = spatial_margin_deg - reload_margin_deg
        return clat - m, clat + m, clon - m, clon + m

    wlat_min, wlat_max, wlon_min, wlon_max = _win_bounds(lat, lon)

    records: list[TrajectoryRecord] = []
    deadline = dive_start + timedelta(days=_MAX_SIMULATED_DAYS)

    while True:
        if time > deadline:
            raise RuntimeError(
                f"Float has not surfaced after {_MAX_SIMULATED_DAYS} simulated days "
                f"(dive started {dive_start}). Check your ControlAction configuration."
            )

        # --- velocity lookup via pre-built scipy interpolator ---
        t_s = np.datetime64(time, "s").astype(np.float64)
        u = float(interp_u([[t_s, depth, lat, lon]])[0])
        v = float(interp_v([[t_s, depth, lat, lon]])[0])
        if np.isnan(u) or np.isnan(v):
            logger.warning("NaN velocity at lat=%.4f lon=%.4f depth=%.1f time=%s — defaulting to 0", lat, lon, depth, time)
            u, v = 0.0, 0.0

        bathy_depth = bathy_interp(lat, lon)

        # --- horizontal update (forward Euler) ---
        dx_m = u * dt_seconds
        dy_m = v * dt_seconds
        lat += dy_m / 111320.0
        lon += dx_m / (111320.0 * math.cos(math.radians(lat)))
        x += dx_m
        y += dy_m

        # --- vertical update and phase transitions ---
        phase, depth, ascent_start_time = _step_phase(
            phase=phase, depth=depth, bathy_depth=bathy_depth,
            action=action, time=time, dt_seconds=dt_seconds,
            ascent_start_time=ascent_start_time,
        )
        time += timedelta(seconds=dt_seconds)

        # Float on seabed is stationary — skip ahead to ascent rather than
        # stepping through every minute of park time.
        if phase == "on_seabed":
            print(f"  {time:%Y-%m-%d %H:%M}  {'on_seabed':<12}  lat={lat:+.4f}  lon={lon:+.4f}  x={x:+8.0f}m  y={y:+8.0f}m  depth={depth:6.1f}m  [fast-forward to {ascent_start_time:%Y-%m-%d %H:%M}]")
            records.append(TrajectoryRecord(
                time=time, lat=lat, lon=lon, x=x, y=y,
                depth=depth, phase="on_seabed", u=u, v=v,
                bathymetry_depth=bathy_depth, on_seabed=True,
            ))
            time = ascent_start_time
            phase = "ascending"
            continue

        print(f"  {time:%Y-%m-%d %H:%M}  {phase:<12}  lat={lat:+.4f}  lon={lon:+.4f}  x={x:+8.0f}m  y={y:+8.0f}m  depth={depth:6.1f}m")

        records.append(TrajectoryRecord(
            time=time, lat=lat, lon=lon, x=x, y=y,
            depth=depth, phase=phase, u=u, v=v,
            bathymetry_depth=bathy_depth, on_seabed=(phase == "on_seabed"),
        ))

        if phase == "at_surface":
            break

        # --- reload tiles if float nears working window edge ---
        if lat < wlat_min or lat > wlat_max or lon < wlon_min or lon > wlon_max:
            logger.info("Float near window boundary — reloading tiles for lat=%.3f lon=%.3f", lat, lon)
            lazy_ds = select_tiles(manifest, data_dir, lat, lon, time, window_end, spatial_margin_deg)
            working_ds = load_working_window(lazy_ds, lat, lon, start_time=time, end_time=window_end,
                                              spatial_margin_deg=spatial_margin_deg)
            interp_u, interp_v = build_velocity_interpolator(working_ds)
            wlat_min, wlat_max, wlon_min, wlon_max = _win_bounds(lat, lon)

    final_state = ProfilerState(time=time, lat=lat, lon=lon, depth=depth, phase="at_surface", x=x, y=y)
    return records, final_state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ascent_start_from_parking_depth(
    park_reached_time: datetime,
    action: ControlAction,
) -> datetime:
    """Return the datetime at which ascent should begin.

    ``cycle_hours`` is measured from the moment the float reaches its
    parking depth or the seabed — so ascent begins exactly that many hours
    after *park_reached_time*.
    """
    return park_reached_time + timedelta(hours=action.cycle_hours)


def _step_phase(
    phase: Phase,
    depth: float,
    bathy_depth: float,
    action: ControlAction,
    time: datetime,
    dt_seconds: float,
    ascent_start_time: datetime | None,
) -> tuple[Phase, float, datetime | None]:
    """Apply one timestep of vertical dynamics and return updated (phase, depth, ascent_start_time)."""

    if phase == "descending":
        depth += action.descent_speed_ms * dt_seconds

        if depth >= bathy_depth:
            depth = bathy_depth
            phase = "on_seabed"
            ascent_start_time = _ascent_start_from_parking_depth(time, action)
            logger.debug("Float hit seabed at %.1f m; ascent starts %s", depth, ascent_start_time)

        elif action.park_mode == "parking_depth" and action.target_depth is not None and depth >= action.target_depth:
            depth = action.target_depth
            phase = "parking"
            ascent_start_time = _ascent_start_from_parking_depth(time, action)
            logger.debug("Float reached parking depth %.1f m; ascent starts %s", depth, ascent_start_time)

        # park_on_bottom with no seabed hit yet: keep descending, no action needed.

    elif phase in ("parking", "on_seabed"):
        if ascent_start_time is not None and time >= ascent_start_time:
            phase = "ascending"
            logger.debug("Float beginning ascent from %.1f m at %s", depth, time)

    elif phase == "ascending":
        depth -= action.ascent_speed_ms * dt_seconds
        if depth <= 0.0:
            depth = 0.0
            phase = "at_surface"

    return phase, depth, ascent_start_time
