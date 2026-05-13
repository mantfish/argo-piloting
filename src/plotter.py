"""Live debug plotter for the profiling float simulator."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from particle_mover import xy_to_latlon

if TYPE_CHECKING:
    from sim_types import ControlAction, EstimatedState, ProfilerState, SimConfig

# Phase colours for the depth panel
_PHASE_COLOUR = {
    "descending": "#4e9af1",
    "parking": "#888888",
    "ascending": "#e05c5c",
    "communicating": "#2ecc71",
    "on_seabed": "#8B4513",
}


class DebugPlotter:

    def __init__(self, config: SimConfig, start_lat: float, start_lon: float) -> None:
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.target_lat = config.control.target.lat
        self.target_lon = config.control.target.lon
        self.bias_truth_fn = config.bias_function

        # Per-cycle accumulators (one value per GPS fix)
        self._cycles: list[int] = []
        self._variance_history: list[float] = []
        self._bx_est_history: list[float] = []
        self._by_est_history: list[float] = []
        self._bx_truth_history: list[float] = []
        self._by_truth_history: list[float] = []

        plt.ion()
        self.fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        self.fig.suptitle("Float simulator — live debug", fontsize=12, fontweight="bold")

        self.ax_map, self.ax_cand, self.ax_cost = axes[0]
        self.ax_var, self.ax_bias, self.ax_depth = axes[1]

        self._setup_axes()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.pause(0.01)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_axes(self) -> None:
        self.ax_map.set_title("Trajectory (map)")
        self.ax_map.set_xlabel("Longitude")
        self.ax_map.set_ylabel("Latitude")
        self.ax_map.plot(self.target_lon, self.target_lat, "r*", ms=14, label="target", zorder=5)
        self.ax_map.legend(loc="upper left", fontsize=7)

        self.ax_cand.set_title("Candidate actions (current cycle)")
        self.ax_cand.set_xlabel("Longitude")
        self.ax_cand.set_ylabel("Latitude")

        self.ax_cost.set_title("Action costs (current cycle)")
        self.ax_cost.set_xlabel("Action")
        self.ax_cost.set_ylabel("Cost (lower = better)")

        self.ax_var.set_title("Position variance trace(P_xx)")
        self.ax_var.set_xlabel("Cycle")
        self.ax_var.set_ylabel("Variance (m²)")
        self.ax_var.set_yscale("log")

        self.ax_bias.set_title("Bias estimates vs truth")
        self.ax_bias.set_xlabel("Cycle")
        self.ax_bias.set_ylabel("Bias (m/s)")

        self.ax_depth.set_title("Real float depth")
        self.ax_depth.set_xlabel("Time")
        self.ax_depth.set_ylabel("Depth (m)")
        self.ax_depth.invert_yaxis()

    # ------------------------------------------------------------------
    # Public update — called once per cycle
    # ------------------------------------------------------------------

    def update(
        self,
        cycle: int,
        real_traj: list[ProfilerState],
        chosen_est_traj: list[EstimatedState],
        all_action_results: list[tuple[ControlAction, list[EstimatedState], float]],
        real_history: list[ProfilerState],
        estimated_history: list[EstimatedState],
        updated_est: EstimatedState,
    ) -> None:
        # Accumulate scalar series
        self._cycles.append(cycle)
        self._variance_history.append(float(np.trace(updated_est.P[:2, :2])))
        self._bx_est_history.append(updated_est.bx)
        self._by_est_history.append(updated_est.by)
        truth = self.bias_truth_fn(updated_est.time)
        self._bx_truth_history.append(truth[0])
        self._by_truth_history.append(truth[1])

        chosen_action = all_action_results[
            min(range(len(all_action_results)), key=lambda i: all_action_results[i][2])
        ][0]

        self._draw_map(real_history, estimated_history, updated_est)
        self._draw_candidates(all_action_results, chosen_action, updated_est)
        self._draw_costs(all_action_results, chosen_action)
        self._draw_variance()
        self._draw_bias()
        self._draw_depth(real_traj)

        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    # ------------------------------------------------------------------
    # Panel drawing helpers
    # ------------------------------------------------------------------

    def _latlon(self, x: float, y: float) -> tuple[float, float]:
        lat, lon = xy_to_latlon(x, y, self.start_lat, self.start_lon)
        return lat, lon

    def _draw_map(
        self,
        real_history: list[ProfilerState],
        estimated_history: list[EstimatedState],
        updated_est: EstimatedState,
    ) -> None:
        ax = self.ax_map
        ax.cla()
        ax.set_title("Trajectory (map)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.plot(self.target_lon, self.target_lat, "r*", ms=14, label="target", zorder=5)

        # Real trajectory
        r_lats = [s.location.lat for s in real_history]
        r_lons = [s.location.lon for s in real_history]
        ax.plot(r_lons, r_lats, color="#4e9af1", lw=1.2, label="real", zorder=2)

        # Estimated trajectory
        e_lats = [s.location.lat for s in estimated_history]
        e_lons = [s.location.lon for s in estimated_history]
        ax.plot(e_lons, e_lats, color="#f5a623", lw=1.2, linestyle="--", label="estimated", zorder=2)

        # GPS fix markers (communicating states in real history)
        fix_lats = [s.location.lat for s in real_history if s.phase == "communicating"]
        fix_lons = [s.location.lon for s in real_history if s.phase == "communicating"]
        ax.scatter(fix_lons, fix_lats, color="#4e9af1", s=20, zorder=3)

        # Uncertainty ellipse at current position
        self._draw_ellipse(ax, updated_est)

        ax.legend(loc="upper left", fontsize=7)

    def _draw_ellipse(self, ax: plt.Axes, state: EstimatedState) -> None:
        P = state.P[:2, :2]
        lat, lon = state.location.lat, state.location.lon
        lat_rad = math.radians(lat)

        # 1σ radii in degrees
        lon_std = math.sqrt(max(P[0, 0], 0.0)) / (111_000.0 * math.cos(lat_rad))
        lat_std = math.sqrt(max(P[1, 1], 0.0)) / 111_000.0

        ellipse = mpatches.Ellipse(
            (lon, lat), width=2 * lon_std, height=2 * lat_std,
            edgecolor="#f5a623", facecolor="#f5a623", alpha=0.2, zorder=4,
        )
        ax.add_patch(ellipse)
        ax.plot(lon, lat, "o", color="#f5a623", ms=5, zorder=5)

    def _draw_candidates(
        self,
        all_action_results: list[tuple[ControlAction, list[EstimatedState], float]],
        chosen_action: ControlAction,
        current_est: EstimatedState,
    ) -> None:
        ax = self.ax_cand
        ax.cla()
        ax.set_title("Candidate actions (current cycle)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        cur_lat, cur_lon = current_est.location.lat, current_est.location.lon

        for action, traj, cost in all_action_results:
            is_chosen = (action is chosen_action)
            lats = [s.location.lat for s in traj]
            lons = [s.location.lon for s in traj]
            colour = "#2ecc71" if is_chosen else "#aaaaaa"
            lw = 1.8 if is_chosen else 0.8
            alpha = 1.0 if is_chosen else 0.6
            ax.plot(lons, lats, color=colour, lw=lw, alpha=alpha, zorder=3 if is_chosen else 2)
            # Label at endpoint
            label = f"{action.parking_depth:.0f}m/{action.duration_hours:.0f}h\nc={cost:.1f}"
            ax.annotate(label, (lons[-1], lats[-1]), fontsize=6, color=colour,
                        textcoords="offset points", xytext=(3, 3))

        ax.plot(cur_lon, cur_lat, "ko", ms=6, zorder=5, label="current")
        ax.plot(self.target_lon, self.target_lat, "r*", ms=10, zorder=5, label="target")
        ax.legend(loc="upper left", fontsize=7)

        # Zoom to ~2° around current position
        margin = 1.0
        ax.set_xlim(cur_lon - margin, cur_lon + margin)
        ax.set_ylim(cur_lat - margin, cur_lat + margin)

    def _draw_costs(
        self,
        all_action_results: list[tuple[ControlAction, list[EstimatedState], float]],
        chosen_action: ControlAction,
    ) -> None:
        ax = self.ax_cost
        ax.cla()
        ax.set_title("Action costs (current cycle)")
        ax.set_xlabel("Action")
        ax.set_ylabel("Cost (lower = better)")

        labels = [f"{a.parking_depth:.0f}m\n{a.duration_hours:.0f}h" for a, _, _ in all_action_results]
        costs = [c for _, _, c in all_action_results]
        colours = ["#2ecc71" if a is chosen_action else "#aaaaaa" for a, _, _ in all_action_results]

        bars = ax.bar(labels, costs, color=colours, edgecolor="black", linewidth=0.5)
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{cost:.2f}", ha="center", va="bottom", fontsize=7)

    def _draw_variance(self) -> None:
        ax = self.ax_var
        ax.cla()
        ax.set_title("Position variance trace(P_xx)")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Variance (m²)")
        if self._variance_history:
            ax.set_yscale("log")
        ax.step(self._cycles, self._variance_history, where="post",
                color="#4e9af1", lw=1.5)
        if self._variance_history:
            ax.scatter(self._cycles, self._variance_history, s=20, color="#4e9af1", zorder=3)

    def _draw_bias(self) -> None:
        ax = self.ax_bias
        ax.cla()
        ax.set_title("Bias estimates vs truth")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Bias (m/s)")
        ax.plot(self._cycles, self._bx_est_history, color="#f5a623", lw=1.5, label="bx est")
        ax.plot(self._cycles, self._by_est_history, color="#2ecc71", lw=1.5, label="by est")
        ax.plot(self._cycles, self._bx_truth_history, color="#f5a623",
                lw=1.0, linestyle="--", label="bx truth", alpha=0.6)
        ax.plot(self._cycles, self._by_truth_history, color="#2ecc71",
                lw=1.0, linestyle="--", label="by truth", alpha=0.6)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)

    def _draw_depth(self, real_traj: list[ProfilerState]) -> None:
        ax = self.ax_depth
        ax.cla()
        ax.set_title("Real float depth (current cycle)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()

        if len(real_traj) < 2:
            return

        times = [s.time for s in real_traj]
        depths = [s.depth for s in real_traj]
        phases = [s.phase for s in real_traj]

        for i in range(len(real_traj) - 1):
            colour = _PHASE_COLOUR.get(phases[i], "#cccccc")
            ax.plot([times[i], times[i + 1]], [depths[i], depths[i + 1]],
                    color=colour, lw=1.5)

        # Legend patches
        seen = set(phases)
        patches = [mpatches.Patch(color=_PHASE_COLOUR.get(p, "#cccccc"), label=p)
                   for p in _PHASE_COLOUR if p in seen]
        if patches:
            ax.legend(handles=patches, loc="lower right", fontsize=6)

        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
