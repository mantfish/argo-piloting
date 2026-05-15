from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import (
    build_bathymetry_interpolator,
    build_velocity_interpolator,
    load_bathymetry,
    load_manifest,
    load_working_window,
)
from sim_types import EstimatedState, GeoLocation, ProfilerState, SimConfig
from control import KFMPC
from particle_mover import simulate_estimate_forward, simulate_real, xy_to_latlon
from noise import bias, Q_DEFAULT
from plotter import DebugPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

_MAX_ACTION_HOURS = 120  # for sizing the working window


def _load_cycle_interpolators(ocean_lazy, lat, lon, t_start, t_end):
    """Load a small spatiotemporal window and return (interp_u, interp_v)."""
    window = load_working_window(
        ocean_lazy, lat, lon, t_start, t_end, spatial_margin_deg=1.0,
    )
    return build_velocity_interpolator(window)


def run_simulation(config: SimConfig):
    start_lat = config.start_location.lat
    start_lon = config.start_location.lon

    # Bathymetry — static, load once
    bathy_ds = load_bathymetry(config.data_dir / "D6_2024.nc")
    bathy_interp = build_bathymetry_interpolator(bathy_ds)

    # Open all current tiles lazily via manifest (skips D6_2024.nc and manifest.yaml)
    manifest = load_manifest(config.data_dir)
    tile_paths = [config.data_dir / entry["file"] for entry in manifest]
    ocean_lazy = xr.open_mfdataset(tile_paths, combine="by_coords")
    logger.info("Manifest loaded: %d tiles (lazy).", len(manifest))

    plotter = DebugPlotter(config, start_lat, start_lon) if config.debug else None

    # Initialise states
    real_state = ProfilerState(
        time=config.start_time,
        location=config.start_location,
        depth=0.0,
        phase="communicating",
    )
    estimated_state = EstimatedState(
        time=config.start_time,
        location=config.start_location,
        depth=0.0,
        phase="communicating",
        P=np.diag([
            100.0 ** 2,  # x std = 100 m
            100.0 ** 2,  # y std = 100 m
            0.05 ** 2,  # bx std = 5 cm/s
            0.05 ** 2,  # by std = 5 cm/s
        ])
    )

    real_history: list[ProfilerState] = [real_state]
    estimated_history: list[EstimatedState] = [estimated_state]

    cycle = 0
    while real_history[-1].time < config.time_end:
        cycle += 1
        current_real = real_history[-1]
        current_est = estimated_history[-1]
        logger.info("Cycle %d | t=%s | pos=(%.3f, %.3f)",
                    cycle, current_real.time, current_real.location.lat, current_real.location.lon)

        # Load velocity data for this cycle (window sized for longest possible action)
        t_window_end = current_real.time + timedelta(hours=_MAX_ACTION_HOURS + 24)
        interp_u, interp_v = _load_cycle_interpolators(
            ocean_lazy,
            current_real.location.lat, current_real.location.lon,
            current_real.time, t_window_end,
        )

        # MPC: evaluate all candidate actions
        best_action = None
        best_cost = float("inf")
        best_est_history: list[EstimatedState] = []
        all_action_results: list[tuple] = []

        best_subcosts: tuple = ()
        for action in config.control.possible_actions:
            est_traj = simulate_estimate_forward(
                current_est, action, interp_u, interp_v,
                bathy_interp, config.process_noise,
                start_lat, start_lon, config.dt,
            )
            cost, flow, dist, sci, var = config.control.evaluate_cost(
                est_traj[-1], current_real, action, interp_u, interp_v, start_lat, start_lon,
            )
            all_action_results.append((action, est_traj, cost))
            logger.debug("  action depth=%.0f dur=%.0f h  cost=%.3f",
                         action.parking_depth, action.duration_hours, cost)
            if cost < best_cost:
                best_cost = cost
                best_action = action
                best_est_history = est_traj
                best_subcosts = (flow, dist, sci, var)

        flow, dist, sci, var = best_subcosts
        logger.info(
            "  chosen: depth=%.0f m  dur=%.0f h  "
            "cost=%.4f  [flow=%.4f  dist=%.4f  sci=%.4f  var=%.4f]",
            best_action.parking_depth, best_action.duration_hours,
            best_cost, flow, dist, sci, var,
        )

        # Simulate real float
        real_traj = simulate_real(
            current_real, best_action, interp_u, interp_v,
            bathy_interp, config.bias_function, config.process_noise,
            start_lat, start_lon, config.noise_seed, config.dt,
        )

        # EKF update at surface — GPS position fix zeros position uncertainty
        surf_est = best_est_history[-1]
        surf_real = real_traj[-1]

        P_XX = surf_est.P[:2, :2]
        P_Xb = surf_est.P[:2, 2:]
        P_bX = surf_est.P[2:, :2]
        P_bb = surf_est.P[2:, 2:]

        innovation = np.array([surf_real.x - surf_est.x, surf_real.y - surf_est.y])
        P_XX_inv = np.linalg.inv(P_XX + np.eye(2) * 1e-10)
        bias_correction = P_bX @ P_XX_inv @ innovation

        updated = EstimatedState(
            time=surf_real.time,
            location=surf_real.location,
            depth=0.0,
            phase="communicating",
            x=surf_real.x,
            y=surf_real.y,
            bx=surf_est.bx + bias_correction[0],
            by=surf_est.by + bias_correction[1],
        )
        P_bb_new = P_bb - P_bX @ P_XX_inv @ P_Xb
        P_new = np.zeros((4, 4))
        P_new[:2, :2] = np.eye(2) * 1e-10
        P_new[2:, 2:] = P_bb_new
        updated.P = P_new

        real_history.extend(real_traj[1:])
        estimated_history.extend(best_est_history[1:])
        estimated_history.append(updated)

        if plotter:
            plotter.update(cycle, real_traj, best_est_history,
                           all_action_results, real_history, estimated_history, updated)

    logger.info("Simulation complete. %d real states, %d estimated states.",
                len(real_history), len(estimated_history))
    if plotter:
        out = config.data_dir.parent / "processed" / "debug_plot.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plotter.save(out)
        logger.info("Debug plot saved to %s", out)
        plt.ioff()
        plt.show()
    return real_history, estimated_history


if __name__ == "__main__":
    control = KFMPC(
        target_location=[55.2, 15.5],  # somewhere in the Baltic dataset
        flow_weight=-0.8,
        distance_weight=1.0,
        science_weight=0*1.0,
        variance_weight=100,
        radius_std_m=8000
    )
    config = SimConfig(
        start_time=datetime(2023, 10, 1),
        start_location=GeoLocation(lat=55.2, lon=15.5),  # within Baltic dataset (lat 53.5-56.5, lon 12.5-18)
        noise_seed=42,
        bias_function=bias,
        process_noise=Q_DEFAULT,
        debug=True,
        data_dir=Path("./data/raw"),
        time_end=datetime(2025, 1, 24),
        control=control,
        dt=300.0,
    )
    run_simulation(config)
