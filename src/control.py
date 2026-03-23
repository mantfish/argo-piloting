"""Control strategies for the profiling float simulator.

This module maps strategy names to functions that decide what the float
should do next, given its current state and a forecast of the velocity
field ahead.

Adding a new strategy
---------------------
1. Implement a function with the signature::

       def _my_strategy(
           state: ProfilerState,
           forecast: xr.Dataset,
           current_action: ControlAction,
       ) -> ControlAction: ...

2. Register it in ``STRATEGIES``::

       STRATEGIES["my_strategy"] = _my_strategy

That is all. No other module needs to change.

Notes
-----
- The *forecast* dataset is a small, fully computed ``xr.Dataset``
  returned by ``get_forecast_field()``. Strategies may inspect ``uo``
  and ``vo`` in it to inform their decision, or ignore it entirely.
- Every strategy must return a ``ControlAction``. Return
  *current_action* unchanged for a no-op, or build and return a new
  instance to change behaviour for the next cycle.

Imports: src/types.py only.
"""
from __future__ import annotations

from typing import Callable

import xarray as xr

from sim_types import ControlAction, ProfilerState


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, Callable[[ProfilerState, xr.Dataset, ControlAction], ControlAction]] = {}
"""Maps strategy name → strategy function.

Populated at module load time by the registrations at the bottom of this
file. Use :func:`get_action` to dispatch by name.
"""


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def get_action(
    state: ProfilerState,
    forecast: xr.Dataset,
    current_action: ControlAction,
    strategy: str,
) -> ControlAction:
    """Return the next :class:`~types.ControlAction` for the profiler.

    Looks up *strategy* in :data:`STRATEGIES` and calls the corresponding
    function. The returned action is used for the next dive cycle.

    Parameters
    ----------
    state:
        Current profiler state at the moment of surfacing.
    forecast:
        Small, fully computed ``xr.Dataset`` covering the expected
        spatial and temporal range of the next dive, as returned by
        ``get_forecast_field()``. Strategies may use ``uo`` / ``vo``
        from this dataset to plan ahead, or ignore it entirely.
    current_action:
        The action used for the cycle that just completed. Passed to
        the strategy so it can make incremental adjustments rather than
        computing everything from scratch.
    strategy:
        Name of the control strategy to use. Must be a key in
        :data:`STRATEGIES`.

    Returns
    -------
    ControlAction
        The action to execute on the next dive.

    Raises
    ------
    ValueError
        If *strategy* is not registered, with a message listing all
        available strategy names.
    """
    if strategy not in STRATEGIES:
        available = ", ".join(f'"{s}"' for s in sorted(STRATEGIES))
        raise ValueError(
            f"Unknown control strategy {strategy!r}. Available strategies: {available}"
        )
    return STRATEGIES[strategy](state, forecast, current_action)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _no_control(
    state: ProfilerState,
    forecast: xr.Dataset,
    current_action: ControlAction,
) -> ControlAction:
    """Baseline strategy: repeat the same dive parameters every cycle.

    The profiler dives to the same depth, parks for the same duration,
    and ascends on the same schedule regardless of its position or the
    forecast. The *forecast* argument is ignored entirely.

    This is the reference case for drift experiments — it lets the float
    go wherever the currents take it with no corrective action.
    """
    return current_action


def _move_towards(
    state: ProfilerState,
    forecast: xr.Dataset,
    current_action: ControlAction,
) -> ControlAction:
    """If  move towards it"""


STRATEGIES["no_control"] = _no_control
