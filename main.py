"""Profiling float simulator — entry point.

Wires all modules together into a complete simulation run. Reading this
file top-to-bottom gives a full picture of the system:

  1. Load lazy ocean + bathymetry datasets from NetCDF tiles.
  2. Loop over dive cycles until the configured end time:
       a. Integrate the float through one descent → park → ascent.
       b. Get a (possibly noisy) forecast window at the surface.
       c. Ask the control module for the next ControlAction.
       d. Advance the clock by the surface waiting time.
  3. Collect all TrajectoryRecords into a DataFrame, save to parquet,
     write a JSON config snapshot, and produce a trajectory plot.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from control import get_action
from data_loader import build_bathymetry_interpolator, get_forecast_field, load_bathymetry, load_manifest, select_tiles
from particle_mover import run_until_surface
from plotter import plot_trajectory
from sim_types import ControlAction, ProfilerState, SimConfig, TrajectoryRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_simulation(config: SimConfig) -> pd.DataFrame:
    """Execute a full profiler simulation and return the trajectory as a DataFrame."""

    # ------------------------------------------------------------------
    # Step 1 — Load data
    # ------------------------------------------------------------------
    manifest = load_manifest(config.data_dir)
    bathy_ds = load_bathymetry(config.data_dir / "D6_2024.nc")
    bathy_interp = build_bathymetry_interpolator(bathy_ds)
    logger.info("Manifest loaded (%d tiles) and bathymetry loaded.", len(manifest))

    # ------------------------------------------------------------------
    # Step 2 — Initialise
    # ------------------------------------------------------------------
    state: ProfilerState = config.start_state
    action: ControlAction = config.default_action
    all_records: list[TrajectoryRecord] = []
    cycle_number = 0

    # ------------------------------------------------------------------
    # Step 3 — Main simulation loop
    # ------------------------------------------------------------------
    while state.time < config.end_time:

        cycle_number += 1
        logger.info(
            "Cycle %d | %s | lat=%.4f lon=%.4f depth=%.1fm",
            cycle_number,
            state.time.strftime("%Y-%m-%d %H:%M"),
            state.lat,
            state.lon,
            state.depth,
        )

        # a. Run particle simulation for one dive cycle.
        records, state = run_until_surface(state, config.data_dir, manifest, bathy_interp, action)
        all_records.extend(records)
        logger.info(
            "  Surfaced at %s | lat=%.4f lon=%.4f",
            state.time.strftime("%Y-%m-%d %H:%M"),
            state.lat,
            state.lon,
        )

        # b. Check if simulation is complete.
        if state.time >= config.end_time:
            break

        # c. Get forecast window for the control decision.
        forecast_end = state.time + timedelta(hours=action.cycle_hours * 2)
        forecast_ds = select_tiles(manifest, config.data_dir, state.lat, state.lon, state.time, forecast_end)
        forecast = get_forecast_field(
            forecast_ds,
            lat=state.lat,
            lon=state.lon,
            time=state.time,
            hours_ahead=action.cycle_hours * 2,
            noise_std=config.forecast_noise_std,
            seed=config.forecast_noise_seed,
        )

        # d. Ask the control module what to do next.
        action = get_action(
            state=state,
            forecast=forecast,
            current_action=action,
            strategy=config.control_strategy,
        )

        # e. Simulate surface waiting time before the next dive.
        state = ProfilerState(
            time=state.time + timedelta(hours=action.surface_duration_hours),
            lat=state.lat,
            lon=state.lon,
            depth=0.0,
            phase="at_surface",
            x=state.x,
            y=state.y,
        )
        logger.info(
            "  Surface wait complete. Next dive at %s",
            state.time.strftime("%Y-%m-%d %H:%M"),
        )

    # ------------------------------------------------------------------
    # Step 4 — Build DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame([vars(r) for r in all_records])
    logger.info(
        "Simulation complete. %d records across %d cycles.",
        len(df),
        cycle_number,
    )

    # ------------------------------------------------------------------
    # Step 5 — Save outputs
    # ------------------------------------------------------------------
    config.output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = config.output_dir / f"{config.control_strategy}_trajectory.parquet"
    df.to_parquet(parquet_path)
    logger.info("Trajectory saved to %s", parquet_path)

    meta_path = config.output_dir / f"{config.control_strategy}_config.json"
    meta = {
        "control_strategy": config.control_strategy,
        "forecast_noise_std": config.forecast_noise_std,
        "forecast_noise_seed": config.forecast_noise_seed,
        "start_lat": config.start_state.lat,
        "start_lon": config.start_state.lon,
        "start_time": config.start_state.time.isoformat(),
        "end_time": config.end_time.isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Config saved to %s", meta_path)

    # ------------------------------------------------------------------
    # Step 6 — Plot
    # ------------------------------------------------------------------
    plot_trajectory(
        df=df,
        config=config,
        save_path=config.output_dir / f"{config.control_strategy}_trajectory.png",
        show=False,
        bathy_ds=bathy_ds,
    )
    logger.info("Plot saved.")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_state = ProfilerState(
        time=datetime(2025, 1, 15, 0, 0, 0),
        lat=55.2,
        lon=15.5,
        depth=0.0,
        phase="at_surface",
    )

    default_action = ControlAction(
        park_mode="parking_depth",
        target_depth=100.0,
        surface_duration_hours=6.0,
        ascent_speed_ms=0.05,
        descent_speed_ms=0.05,
        cycle_hours=120.0,
    )

    config = SimConfig(
        start_state=start_state,
        end_time=datetime(2025, 2, 15, 0, 0, 0),
        control_strategy="move_towards",
        default_action=default_action,
        forecast_noise_std=0.0,
        forecast_noise_seed=42,
        data_dir=Path("data/raw"),
        output_dir=Path("data/processed"),
    )

    df = run_simulation(config)
    print(f"Done. {len(df)} trajectory records.")
