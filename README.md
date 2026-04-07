# Profiling Float Environmental Navigation

This project develops control strategies for navigating profiling floats using operational ocean forecasts.

## Overview

A simulation environment models the full dive cycle of a profiling float, from descent through parking to ascent and surface communication. Users can select a control strategy, define a starting state, and configure output directories via `main.py`.

## Running the simulation

```bash
uv run python main.py
```

## Project structure

```
.
├── data/
│   ├── argo_data/       # Downloaded Argo float profiles
│   ├── external/
│   ├── processed/       # Simulation outputs (one timestamped folder per run)
│   ├── raw/             # CMEMS ocean model tiles + manifest
│   └── test_data/
├── notebooks/           # Exploratory analysis
├── src/
│   ├── argo_data_getter.py
│   ├── control.py
│   ├── data_getter.py
│   ├── data_loader.py
│   ├── generate_test_data.py
│   ├── particle_mover.py
│   ├── plotter.py
│   └── sim_types.py
├── tests/
├── main.py
├── pyproject.toml
└── README.md
```

## Source files

### `main.py`
Entry point. Configure the control strategy, profiler starting state, and output directory here.

### `src/particle_mover.py`
Core simulation loop for one dive cycle:

1. Load bathymetry data.
2. While end time is not reached:
   - Fetch a forecast window. Gaussian noise can optionally be added to the model velocities to mimic real forecast error.
   - Query the control strategy for a `ControlAction` given the current profiler state and forecast.
   - Log the cycle.
   - Run the simulation to the next communication interval:
     - Load a small spatiotemporal window of ocean model data to keep memory usage low.
     - Build a linear interpolator over the velocity field (depth × lat × lon × time).
     - Drift the float on the surface for the communication duration using forward Euler integration (10-minute timestep).
     - Determine the next phase from the control action: descend, ascend, drift on surface, park at depth, or park on seabed.
     - Step the float forward in 10-minute increments, updating position and phase. If the float hits the seabed it parks and fast-forwards to the ascent time.
   - Stop if the simulation end time is reached or the float leaves the data domain.
3. Save the trajectory as a Parquet file, a parameter summary as JSON, and a plot — all in a new timestamped output folder.

### `src/control.py`
Control strategies available to the user. Each strategy subclasses `ControlStrategy` and must implement:

- `get_action(profiler_state, forecast) -> ControlAction` — returns the next action for the float.
- `get_log() -> dict` — returns a JSON-serialisable summary of strategy parameters.

### `src/data_getter.py`
Downloads and tiles CMEMS ocean model data into small regional NetCDF files. Also writes a `manifest.yaml` that the simulation uses to efficiently select and load relevant tiles.

### `src/data_loader.py`
Memory-efficient helpers for loading tiled CMEMS data and GEBCO bathymetry into the simulation.

### `src/argo_data_getter.py`
Small utility for downloading real Argo float profile data.

### `src/generate_test_data.py`
Generates synthetic test data to validate the simulation loop without requiring a full CMEMS download.

### `src/plotter.py`
Helpers for plotting float trajectories.

### `src/sim_types.py`
Shared type definitions used across all modules:

| Class | Description |
|---|---|
| `ProfilerState` | Float state at a single instant: time, lat, lon, depth, phase, and x/y displacement from start (metres). |
| `ControlAction` | Instruction for one cycle: `park_mode` (`"drift_on_surface"`, `"park_on_bottom"`, `"parking_depth"`), cycle hours, transmission duration (minutes), target depth, ascent/descent speed (m/s). |
| `TrajectoryRecord` | One row of the recorded trajectory: time, lat, lon, x, y, depth, phase, u/v velocity (m/s), seabed depth, and whether the float is on the seabed. |
| `ControlStrategy` | Abstract base class for control strategies (see `control.py`). |
| `SimConfig` | Top-level simulation configuration: `start_state`, `end_time`, `control_strategy`, forecast noise parameters, forecast horizon, data directory, output directory. |

The notebook in `/notebooks` is the backbone for validating simulation performance against real Argo float data. The workflow is:

1. Download trajectory and measurement data for a float from the Argo database.
2. Build a YAML file of *actions* (one per dive cycle) from the downloaded data.
3. Run the simulation using those actions. Simulations can either be re-initialised at each surfacing position, or chained across N consecutive cycles without resetting position.
4. Plot position errors between the simulated and observed trajectories.

Ascent and descent speeds are taken as the mean speed observed during each phase. Surface duration is calculated from the time between the last GPS fix of one cycle and the first GPS fix of the next.

**Known limitation:** the simulated float surfaces and descends with a slight timing offset relative to the real float.

