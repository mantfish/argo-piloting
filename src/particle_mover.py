"""Single-cycle integrator for a profiling float.

Advances a float from the moment it starts descending until it returns to
the surface, recording the full trajectory. All ocean data access is
delegated to data_loader; all type definitions come from types.

Imports: src/types.py, src/data_loader.py, src/kalman.py only.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from data_loader import build_bathymetry_interpolator, build_velocity_interpolator, load_working_window, select_tiles
from kalman import build_F, estimate_forecast_gradient, propagate_P
from sim_types import ControlAction, GeoLocation, Phase, ProfilerState, RealProfilerState, StateVector, TrajectoryRecord

logger = logging.getLogger(__name__)

_MAX_SIMULATED_DAYS = 365


def run_until_next_action(
    state: RealProfilerState,
    data_dir: Path,
    manifest: list[dict],
    bathy_interp,
    action: ControlAction,
    bias_fn: Callable[[datetime], list[float]],
    Q: np.ndarray,
    est_state: ProfilerState | None = None,
    dt_vertical_seconds: float = 30.0,
    dt_parking_seconds: float = 3000.0,
    dt_surface_seconds: float = 120.0,
    spatial_margin_deg: float = 0.5,
    reload_margin_deg: float = 0.2,
    use_rk4: bool = True,
) -> tuple[list[TrajectoryRecord], RealProfilerState, ProfilerState | None]:
    """Step a profiling float through one full dive cycle.

    Receives a state with ``phase="communicating"`` (surface waiting time has
    already elapsed before this call). Integrates forward until the float
    re-surfaces, then returns.

    Parameters
    ----------
    state:
        Current real profiler state. Must have ``phase="communicating"``.
    data_dir:
        Directory containing tile NetCDF files.
    manifest:
        Tile manifest from :func:`~data_loader.load_manifest`.
    bathy_interp:
        Bathymetry interpolator from :func:`~data_loader.build_bathymetry_interpolator`.
    action:
        Control instruction for this cycle.
    bias_fn:
        Known deterministic bias function. Returns [bx, by] in m/hr.
    Q:
        Process noise covariance matrix (4×4), per unit time (per hour).
    est_state:
        EKF estimated state at cycle start. If None, no EKF propagation
        is performed and the third return value is None.
    dt_vertical_seconds:
        Integration timestep for descent and ascent phases. Default 30 s.
    dt_parking_seconds:
        Integration timestep for parking and surface-drift phases. Default 3000 s.
    dt_surface_seconds:
        Integration timestep for the initial transmission window. Default 120 s.
    spatial_margin_deg:
        Spatial padding for tile selection and working window slicing.
    reload_margin_deg:
        Trigger a tile reload when float comes within this many degrees of the window edge.
    use_rk4:
        Use RK4 integration (default). False falls back to forward Euler.

    Returns
    -------
    tuple[list[TrajectoryRecord], RealProfilerState, ProfilerState | None]
        ``(records, final_real_state, final_est_state)`` where *records* is the
        full trajectory for this cycle. *final_est_state* is None when
        *est_state* input was None.
    """
    if state.phase != "communicating":
        raise ValueError(f"run_until_surface expects phase='communicating', got {state.phase!r}")

    start_time = state.time
    next_action_time = start_time + timedelta(hours=action.cycle_hours)
    lat = state.location.lat
    lon = state.location.lon
    depth = state.depth
    x = state.x
    y = state.y
    phase = state.phase
    window_end = (
        start_time
        + timedelta(hours=action.cycle_hours)
        + timedelta(hours=12)  # safety buffer
    )

    lazy_ds = select_tiles(manifest, data_dir, lat, lon, start_time, window_end, spatial_margin_deg)
    working_ds = load_working_window(lazy_ds, lat, lon, start_time=start_time, end_time=window_end,
                                     spatial_margin_deg=spatial_margin_deg)
    interp_u, interp_v = build_velocity_interpolator(working_ds)

    def _win_bounds(clat, clon):
        m = spatial_margin_deg - reload_margin_deg
        return clat - m, clat + m, clon - m, clon + m

    wlat_min, wlat_max, wlon_min, wlon_max = _win_bounds(lat, lon)

    records: list[TrajectoryRecord] = []
    deadline = start_time + timedelta(days=_MAX_SIMULATED_DAYS)
    _nan_vel_count = 0
    _NAN_VEL_SEABED_THRESHOLD = 5

    # --- transmission window drift ---
    logger.info("Simulating drift during transmission window")
    time = start_time

    while time < start_time + timedelta(minutes=action.transmission_duration_minutes):
        t_s = np.datetime64(time, "s").astype(np.float64)
        u_surface = float(interp_u([[t_s, depth, lat, lon]])[0])
        v_surface = float(interp_v([[t_s, depth, lat, lon]])[0])
        if use_rk4:
            dlat, dlon, dx_m, dy_m = _rk4_horizontal_step(
                interp_u, interp_v, time, dt_surface_seconds, depth, phase, action, lat, lon
            )
        else:
            dx_m = u_surface * dt_surface_seconds
            dy_m = v_surface * dt_surface_seconds
            dlat = dy_m / 111320.0
            dlon = dx_m / (111320.0 * math.cos(math.radians(lat)))
        lat += dlat
        lon += dlon

        # Real state: add bias and process noise scaled by timestep.
        dt_hours_step = dt_surface_seconds / 3600.0
        b = bias_fn(time)
        noise = np.random.multivariate_normal(np.zeros(4), Q * dt_hours_step)
        x += dx_m + b[0] * dt_hours_step + noise[0]
        y += dy_m + b[1] * dt_hours_step + noise[1]

        records.append(TrajectoryRecord(
            time=time, lat=lat, lon=lon, x=x, y=y,
            depth=depth, phase=phase, u=u_surface, v=v_surface,
            bathymetry_depth=np.nan, on_seabed=False,
        ))
        time += timedelta(seconds=dt_surface_seconds)
        logger.info("Simulated surface drift to %.1f m, %.1f m, drifted for %.1f minutes",
                    x, y, (time - start_time).total_seconds() / 60.0)

    phase = _start_action(action, phase)

    # --- main dive loop ---
    while True:
        if time > deadline:
            raise RuntimeError(
                f"Float has not surfaced after {_MAX_SIMULATED_DAYS} simulated days "
                f"(action started {start_time}). Check your ControlAction configuration."
            )

        if phase in ("parking", "on_seabed"):
            dt = dt_parking_seconds
        elif phase == "drift_on_surface":
            dt = dt_surface_seconds
        else:
            dt = dt_vertical_seconds

        t_s = np.datetime64(time, "s").astype(np.float64)
        u = float(interp_u([[t_s, depth, lat, lon]])[0])
        v = float(interp_v([[t_s, depth, lat, lon]])[0])
        bathy_depth = bathy_interp(lat, lon)
        if np.isnan(u) or np.isnan(v):
            logger.warning(
                "NaN velocity at lat=%.4f lon=%.4f depth=%.1f bathy=%.1f time=%s — defaulting to 0 during phase %s",
                lat, lon, depth, bathy_depth, time, phase,
            )
            u, v = 0.0, 0.0
            if phase in ("parking", "on_seabed"):
                _nan_vel_count += 1
                if _nan_vel_count >= _NAN_VEL_SEABED_THRESHOLD:
                    logger.warning(
                        "5 consecutive NaN velocities at depth=%.1f (bathy=%.1f) during phase %s "
                        "— treating as on_seabed and skipping to %s",
                        depth, bathy_depth, phase, next_action_time,
                    )
                    records.append(TrajectoryRecord(
                        time=time, lat=lat, lon=lon, x=x, y=y,
                        depth=depth, phase="on_seabed", u=u, v=v,
                        bathymetry_depth=bathy_depth, on_seabed=True,
                    ))
                    time = next_action_time
                    phase = "ascending"
                    continue
        else:
            _nan_vel_count = 0

        # --- horizontal update ---
        if use_rk4:
            dlat, dlon, dx_m, dy_m = _rk4_horizontal_step(
                interp_u, interp_v, time, dt, depth, phase, action, lat, lon
            )
        else:
            dx_m = u * dt
            dy_m = v * dt
            dlat = dy_m / 111320.0
            dlon = dx_m / (111320.0 * math.cos(math.radians(lat)))
        lat += dlat
        lon += dlon

        # Real state: add bias and process noise scaled by timestep.
        dt_hours_step = dt / 3600.0
        b = bias_fn(time)
        noise = np.random.multivariate_normal(np.zeros(4), Q * dt_hours_step)
        x += dx_m + b[0] * dt_hours_step + noise[0]
        y += dy_m + b[1] * dt_hours_step + noise[1]

        # EKF covariance propagation during dominant drift phases.
        if est_state is not None and phase in ("parking", "on_seabed", "drift_on_surface"):
            J = estimate_forecast_gradient(interp_u, interp_v, lat, lon, time, depth)
            F = build_F(dt_hours_step, J)
            new_P = propagate_P(est_state.P, F, Q, dt_hours_step)
            new_x = est_state.X.x + dx_m + b[0] * dt_hours_step
            new_y = est_state.X.y + dy_m + b[1] * dt_hours_step
            new_X = StateVector(x=new_x, y=new_y, bx=est_state.X.bx, by=est_state.X.by)
            est_state = ProfilerState(time=time, lat=lat, lon=lon, X=new_X, P=new_P)

        # --- vertical update and phase transitions ---
        phase, depth = _step_phase(
            phase=phase, depth=depth, bathy_depth=bathy_depth,
            action=action, time=time, dt_seconds=dt,
            next_action_time=next_action_time,
        )
        time += timedelta(seconds=dt)

        if phase == "on_seabed":
            logger.debug(
                "  %s  %-12s  lat=%+.4f  lon=%+.4f  x=%+8.0fm  y=%+8.0fm  depth=%6.1fm  "
                "[fast-forward to %s]",
                time, "on_seabed", lat, lon, x, y, depth, next_action_time,
            )
            records.append(TrajectoryRecord(
                time=time, lat=lat, lon=lon, x=x, y=y,
                depth=depth, phase="on_seabed", u=u, v=v,
                bathymetry_depth=bathy_depth, on_seabed=True,
            ))
            time = next_action_time
            phase = "ascending"
            continue

        logger.debug(
            "  %s  %-12s  lat=%+.4f  lon=%+.4f  x=%+8.0fm  y=%+8.0fm  depth=%6.1fm",
            time, phase, lat, lon, x, y, depth,
        )

        records.append(TrajectoryRecord(
            time=time, lat=lat, lon=lon, x=x, y=y,
            depth=depth, phase=phase, u=u, v=v,
            bathymetry_depth=bathy_depth, on_seabed=(phase == "on_seabed"),
        ))

        if phase == "communicating":
            break

        # --- reload tiles if float nears working window edge ---
        if lat < wlat_min or lat > wlat_max or lon < wlon_min or lon > wlon_max:
            logger.info("Float near window boundary — reloading tiles for lat=%.3f lon=%.3f", lat, lon)
            lazy_ds = select_tiles(manifest, data_dir, lat, lon, time, window_end, spatial_margin_deg)
            working_ds = load_working_window(lazy_ds, lat, lon, start_time=time, end_time=window_end,
                                              spatial_margin_deg=spatial_margin_deg)
            interp_u, interp_v = build_velocity_interpolator(working_ds)
            wlat_min, wlat_max, wlon_min, wlon_max = _win_bounds(lat, lon)

    final_real = RealProfilerState(
        time=time,
        location=GeoLocation(lat=lat, lon=lon),
        depth=depth,
        phase="communicating",
        x=x,
        y=y,
        bathymetry_depth=bathy_depth,
    )
    if est_state is not None:
        est_state = ProfilerState(time=time, lat=lat, lon=lon, X=est_state.X, P=est_state.P)
    return records, final_real, est_state


def quickly_estimate_next_surface_position(
    state: RealProfilerState,
    action: ControlAction,
    forecast,
    use_rk4: bool = True,
    assumed_park_depth: float | None = None,
    dt_vertical_seconds: float = 600.0,
    dt_parking_seconds: float = 3600.0,
    dt_surface_seconds: float = 3600.0,
) -> RealProfilerState:
    """Estimate the float's state after one complete dive cycle.

    Designed to be called hundreds of times inside a control strategy's
    ``get_action``. Reuses the same RK4/Euler integration math as
    ``run_until_next_action`` but strips everything else: no tile loading,
    no trajectory recording, no logging, no window reloading, no noise.

    Parameters
    ----------
    state:
        Current float state. Must have ``phase="communicating"``.
    action:
        The control action to evaluate.
    forecast:
        Fully computed forecast ``xr.Dataset`` (e.g. from
        ``get_forecast_field``). Used directly to build velocity
        interpolators.
    use_rk4:
        Use RK4 integration (default).
    assumed_park_depth:
        Depth (m) used as seabed stand-in for ``park_on_bottom`` actions.
        Defaults to ``state.bathymetry_depth`` if available, otherwise 2000 m.
    dt_vertical_seconds / dt_parking_seconds / dt_surface_seconds:
        Timesteps for the respective phases.

    Returns
    -------
    RealProfilerState
        Estimated state when the float next reaches the surface,
        with ``phase="communicating"``.
    """
    if assumed_park_depth is None:
        assumed_park_depth = (
            state.bathymetry_depth
            if not math.isnan(state.bathymetry_depth)
            else 2000.0
        )

    interp_u, interp_v = build_velocity_interpolator(forecast)

    lat = state.location.lat
    lon = state.location.lon
    depth = state.depth
    time = state.time
    x = 0.0
    y = 0.0

    next_action_time = time + timedelta(hours=action.cycle_hours)

    # --- transmission drift ---
    transmission_end = time + timedelta(minutes=action.transmission_duration_minutes)
    while time < transmission_end:
        dt = min(dt_surface_seconds, (transmission_end - time).total_seconds())
        if use_rk4:
            dlat, dlon, dx_m, dy_m = _rk4_horizontal_step(
                interp_u, interp_v, time, dt, depth, "drift_on_surface", action, lat, lon
            )
        else:
            u, v = _query_velocity(interp_u, interp_v, time, depth, lat, lon)
            dy_m = v * dt
            dx_m = u * dt
            dlat = dy_m / 111320.0
            dlon = dx_m / (111320.0 * math.cos(math.radians(lat)))
        lat += dlat
        lon += dlon
        x += dx_m
        y += dy_m
        time += timedelta(seconds=dt)

    phase = _start_action(action, "communicating")

    # --- main dive loop ---
    while phase != "communicating":
        if phase in ("parking", "on_seabed", "drift_on_surface"):
            dt = dt_parking_seconds
        else:
            dt = dt_vertical_seconds

        if use_rk4:
            dlat, dlon, dx_m, dy_m = _rk4_horizontal_step(
                interp_u, interp_v, time, dt, depth, phase, action, lat, lon
            )
        else:
            u, v = _query_velocity(interp_u, interp_v, time, depth, lat, lon)
            dy_m = v * dt
            dx_m = u * dt
            dlat = dy_m / 111320.0
            dlon = dx_m / (111320.0 * math.cos(math.radians(lat)))
        lat += dlat
        lon += dlon
        x += dx_m
        y += dy_m

        phase, depth = _step_phase(
            phase=phase, depth=depth, bathy_depth=assumed_park_depth,
            action=action, time=time, dt_seconds=dt,
            next_action_time=next_action_time,
        )
        time += timedelta(seconds=dt)

        if phase == "on_seabed":
            time = next_action_time
            phase = "ascending"

    return RealProfilerState(
        time=time,
        location=GeoLocation(lat=lat, lon=lon),
        depth=0.0,
        phase="communicating",
        x=state.x + x,
        y=state.y + y,
        bathymetry_depth=assumed_park_depth,
    )


def propagate_covariance_through_cycle(
    est_state: ProfilerState,
    action: ControlAction,
    forecast,
    Q: np.ndarray,
) -> np.ndarray:
    """Approximate P propagation for one full dive cycle (for MPC use).

    Uses a single-step EKF prediction with F computed at the current estimated
    position. This is a first-order approximation suitable for the trace(P)
    penalty in the MPC cost function — much cheaper than full per-timestep
    integration when called 13+ times per decision cycle.

    Parameters
    ----------
    est_state : current EKF estimated state (provides lat, lon, time, P)
    action    : candidate control action (provides cycle_hours)
    forecast  : xr.Dataset from get_forecast_field()
    Q         : (4,4) process noise covariance per hour

    Returns
    -------
    np.ndarray of shape (4, 4): propagated covariance after action.cycle_hours
    """
    interp_u, interp_v = build_velocity_interpolator(forecast)
    J = estimate_forecast_gradient(
        interp_u, interp_v,
        lat=est_state.lat,
        lon=est_state.lon,
        t=est_state.time,
        depth=0.0,  # surface approximation for MPC rollout
    )
    F = build_F(action.cycle_hours, J)
    return propagate_P(est_state.P, F, Q, action.cycle_hours)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _query_velocity(
    interp_u,
    interp_v,
    t: datetime,
    depth: float,
    lat: float,
    lon: float,
) -> tuple[float, float]:
    """Query interpolated (u, v) at a single point, returning (0, 0) for NaN."""
    t_s = np.datetime64(t, "s").astype(np.float64)
    u = float(interp_u([[t_s, depth, lat, lon]])[0])
    v = float(interp_v([[t_s, depth, lat, lon]])[0])
    if math.isnan(u) or math.isnan(v):
        return 0.0, 0.0
    return u, v


def _rk4_horizontal_step(
    interp_u,
    interp_v,
    t: datetime,
    dt: float,
    depth: float,
    phase: Phase,
    action: ControlAction,
    lat: float,
    lon: float,
) -> tuple[float, float, float, float]:
    """Compute horizontal displacement via RK4, returning (dlat, dlon, dx_m, dy_m).

    Vertical depth at each sub-stage is derived analytically from the
    prescribed ascent/descent speed — no phase-transition logic is applied
    mid-step.
    """

    def depth_at(frac: float) -> float:
        if phase == "descending":
            return min(depth + action.descent_speed_ms * dt * frac,
                       action.target_depth if action.target_depth is not None else float("inf"))
        elif phase == "ascending":
            return max(depth - action.ascent_speed_ms * dt * frac, 0.0)
        else:
            return depth

    def vel(frac: float, lat_: float, lon_: float) -> tuple[float, float]:
        return _query_velocity(interp_u, interp_v, t + timedelta(seconds=dt * frac), depth_at(frac), lat_, lon_)

    def to_deg(u_: float, v_: float, ref_lat: float) -> tuple[float, float]:
        dlat = v_ * dt / 111320.0
        dlon = u_ * dt / (111320.0 * math.cos(math.radians(ref_lat)))
        return dlat, dlon

    u1, v1 = vel(0.0, lat, lon)
    dlat1, dlon1 = to_deg(u1, v1, lat)

    u2, v2 = vel(0.5, lat + dlat1 / 2, lon + dlon1 / 2)
    dlat2, dlon2 = to_deg(u2, v2, lat)

    u3, v3 = vel(0.5, lat + dlat2 / 2, lon + dlon2 / 2)
    dlat3, dlon3 = to_deg(u3, v3, lat)

    u4, v4 = vel(1.0, lat + dlat3, lon + dlon3)
    dlat4, dlon4 = to_deg(u4, v4, lat)

    dlat = (dlat1 + 2 * dlat2 + 2 * dlat3 + dlat4) / 6
    dlon = (dlon1 + 2 * dlon2 + 2 * dlon3 + dlon4) / 6
    dx_m = (u1 + 2 * u2 + 2 * u3 + u4) * dt / 6
    dy_m = (v1 + 2 * v2 + 2 * v3 + v4) * dt / 6

    return dlat, dlon, dx_m, dy_m


def _start_action(action: ControlAction, current_phase: Phase) -> Phase:
    if current_phase == "communicating":
        if action.park_mode in ("parking_depth", "park_on_bottom"):
            return "descending"
        elif action.park_mode == "drift_on_surface":
            return "drift_on_surface"
    raise ValueError(f"Invalid phase {current_phase!r} for action {action!r}")


def _step_phase(
    phase: Phase,
    depth: float,
    bathy_depth: float,
    action: ControlAction,
    time: datetime,
    dt_seconds: float,
    next_action_time: datetime | None,
) -> tuple[Phase, float]:
    """Apply one timestep of vertical dynamics and return updated (phase, depth)."""

    if phase == "descending":
        depth += action.descent_speed_ms * dt_seconds

        if (
            action.park_mode == "parking_depth"
            and next_action_time is not None
            and action.ascent_speed_ms is not None
            and action.ascent_speed_ms > 0
        ):
            time_to_ascend_s = depth / action.ascent_speed_ms
            time_remaining_s = (next_action_time - time).total_seconds()
            if time_to_ascend_s >= time_remaining_s:
                phase = "ascending"

        if phase == "ascending":
            pass
        elif depth >= bathy_depth:
            depth = bathy_depth
            phase = "on_seabed"
            logger.debug("Float hit seabed at %.1f m; ascent starts %s", depth, next_action_time)
        elif action.park_mode == "parking_depth" and action.target_depth is not None and depth >= action.target_depth:
            depth = action.target_depth
            phase = "parking"
            logger.debug("Float reached parking depth %.1f m; ascent starts %s", depth, next_action_time)

    elif phase in ("parking", "on_seabed"):
        if next_action_time is not None and time >= next_action_time:
            phase = "ascending"
            logger.debug("Float beginning ascent from %.1f m at %s", depth, time)

    elif phase == "ascending":
        depth -= action.ascent_speed_ms * dt_seconds
        if depth <= 0.0:
            depth = 0.0
            phase = "communicating"

    elif phase == "drift_on_surface":
        if time >= next_action_time:
            phase = "communicating"
            logger.debug("Float has started communicating at %s", time)

    return phase, depth
