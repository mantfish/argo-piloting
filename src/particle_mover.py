"""Single-cycle integrator for a profiling float."""
from __future__ import annotations

import copy
import logging
import math
from datetime import datetime, timedelta
from typing import Callable

import numpy as np

from sim_types import ControlAction, EstimatedState, GeoLocation, ProfilerState

logger = logging.getLogger(__name__)

_DESCENDING = "descending"
_PARKING = "parking"
_ASCENDING = "ascending"
_COMMUNICATING = "communicating"


def xy_to_latlon(x: float, y: float, start_lat: float, start_lon: float) -> tuple[float, float]:
    lat = start_lat + y / 111_000.0
    lon = start_lon + x / (111_000.0 * math.cos(math.radians(start_lat)))
    return lat, lon


def _query_uv(x, y, z, t, interp_u, interp_v, start_lat, start_lon) -> tuple[float, float]:
    lat, lon = xy_to_latlon(x, y, start_lat, start_lon)
    t_s = np.datetime64(t, "s").astype(np.float64)
    u = float(interp_u([[t_s, z, lat, lon]])[0])
    v = float(interp_v([[t_s, z, lat, lon]])[0])
    if math.isnan(u):
        u = 0.0
    if math.isnan(v):
        v = 0.0
    return u, v


def _compute_jacobian(x, y, z, t, interp_u, interp_v, start_lat, start_lon, eps=500.0) -> np.ndarray:
    """4x4 linearised dynamics Jacobian F for state [x, y, bx, by]."""
    def qv(xi, yi):
        return _query_uv(xi, yi, z, t, interp_u, interp_v, start_lat, start_lon)

    u_xp, v_xp = qv(x + eps, y)
    u_xm, v_xm = qv(x - eps, y)
    u_yp, v_yp = qv(x, y + eps)
    u_ym, v_ym = qv(x, y - eps)

    return np.array([
        [(u_xp - u_xm) / (2 * eps), (u_yp - u_ym) / (2 * eps), 1.0, 0.0],
        [(v_xp - v_xm) / (2 * eps), (v_yp - v_ym) / (2 * eps), 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])


def simulate_estimate_forward(
    estimated_state: EstimatedState,
    control_action: ControlAction,
    interp_u: Callable,
    interp_v: Callable,
    bathy_interp: Callable,
    Q: np.ndarray,
    start_lat: float,
    start_lon: float,
    dt: float = 3600.0,
) -> list[EstimatedState]:
    state = copy.deepcopy(estimated_state)
    state.phase = _DESCENDING
    history = [copy.deepcopy(state)]

    action_start_time = state.time
    n_steps = int(control_action.duration_hours * 3600.0 / dt)

    for _ in range(n_steps):
        lat, lon = xy_to_latlon(state.x, state.y, start_lat, start_lon)
        bottom_depth = bathy_interp(lat, lon)

        # Phase transitions
        if state.phase == _DESCENDING:
            if state.depth >= control_action.parking_depth or state.depth >= bottom_depth:
                state.phase = _PARKING
            else:
                state.depth += control_action.descent_speed_ms * dt

        if state.phase == _PARKING:
            elapsed = (state.time - action_start_time).total_seconds()
            time_remaining = control_action.duration_hours * 3600.0 - elapsed
            if time_remaining <= state.depth / control_action.ascent_speed_ms:
                state.phase = _ASCENDING

        if state.phase == _ASCENDING:
            state.depth = max(0.0, state.depth - control_action.ascent_speed_ms * dt)

        # Horizontal drift (not when resting on seabed)
        parked_on_bottom = state.phase == _PARKING and state.depth >= bottom_depth
        if not parked_on_bottom:
            u, v = _query_uv(state.x, state.y, state.depth, state.time,
                              interp_u, interp_v, start_lat, start_lon)
            F = _compute_jacobian(state.x, state.y, state.depth, state.time,
                                  interp_u, interp_v, start_lat, start_lon)  # pre-step
            state.x += (u + state.bx) * dt
            state.y += (v + state.by) * dt
            # Exact discrete propagation: always maintains PSD (Euler form does not).
            # Valid because |F*dt| << 1 for typical ocean gradients and dt=3600s.
            Phi = np.eye(4) + F * dt
            state.P = Phi @ state.P @ Phi.T + Q * dt
        else:
            # Float is stationary so position and cross-covariance don't evolve,
            # but bias keeps drifting in the real world regardless
            state.P[2, 2] += Q[2, 2] * dt
            state.P[3, 3] += Q[3, 3] * dt

        state.time += timedelta(seconds=dt)
        state.location = GeoLocation(*xy_to_latlon(state.x, state.y, start_lat, start_lon))
        history.append(copy.deepcopy(state))

    return history


def simulate_real(
    real_state: ProfilerState,
    control_action: ControlAction,
    interp_u: Callable,
    interp_v: Callable,
    bathy_interp: Callable,
    bias_function: Callable,
    Q: np.ndarray,
    start_lat: float,
    start_lon: float,
    noise_seed: int = 42,
    dt: float = 3600.0,
) -> list[ProfilerState]:
    state = copy.deepcopy(real_state)
    state.phase = _DESCENDING
    history = [copy.deepcopy(state)]

    action_start_time = state.time
    n_steps = int(control_action.duration_hours * 3600.0 / dt)
    rng = np.random.default_rng(noise_seed)

    for _ in range(n_steps):
        lat, lon = xy_to_latlon(state.x, state.y, start_lat, start_lon)
        bottom_depth = bathy_interp(lat, lon)

        # Phase transitions
        if state.phase == _DESCENDING:
            if state.depth >= control_action.parking_depth or state.depth >= bottom_depth:
                state.phase = _PARKING
            else:
                state.depth += control_action.descent_speed_ms * dt

        if state.phase == _PARKING:
            elapsed = (state.time - action_start_time).total_seconds()
            time_remaining = control_action.duration_hours * 3600.0 - elapsed
            if time_remaining <= state.depth / control_action.ascent_speed_ms:
                state.phase = _ASCENDING

        if state.phase == _ASCENDING:
            state.depth = max(0.0, state.depth - control_action.ascent_speed_ms * dt)

        # Horizontal drift
        parked_on_bottom = state.phase == _PARKING and state.depth >= bottom_depth
        if not parked_on_bottom:
            u, v = _query_uv(state.x, state.y, state.depth, state.time,
                              interp_u, interp_v, start_lat, start_lon)
            bias = bias_function(state.time)
            # Process noise on position only (first 2 components of Q)
            noise = rng.multivariate_normal(np.zeros(2), Q[:2, :2] * dt)
            state.x += (u + bias[0]) * dt + noise[0]
            state.y += (v + bias[1]) * dt + noise[1]


        state.time += timedelta(seconds=dt)
        state.location = GeoLocation(*xy_to_latlon(state.x, state.y, start_lat, start_lon))
        history.append(copy.deepcopy(state))

    return history
