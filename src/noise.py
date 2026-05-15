from __future__ import annotations

from datetime import datetime

import numpy as np

# All units in SI (metres, seconds, m/s).
#
# sigma_pos  = 50 m/sqrt(hr)  = 50/60  m/sqrt(s) → Q_pos  = (50/60)² m²/s ≈ 0.694 m²/s
# sigma_bias = 0.001 (m/s)/sqrt(hr) = 0.001/60 (m/s)/sqrt(s) → Q_bias ≈ 2.78e-10 (m/s)²/s
#
# For a 1-hour step: position noise std = sqrt(Q_pos * 3600) = 50 m  ✓
#                    bias noise std     = sqrt(Q_bias * 3600) = 0.001 m/s  ✓

_SIGMA_POS  = 120.0 / 60.0          # m/sqrt(s)
_SIGMA_BIAS = 0.5 / 60.0         # (m/s)/sqrt(s)

Q_DEFAULT: np.ndarray = np.diag([
    _SIGMA_POS  ** 2,   # x    (m²/s)
    _SIGMA_POS  ** 2,   # y    (m²/s)
    _SIGMA_BIAS ** 2,   # bx   ((m/s)²/s)
    _SIGMA_BIAS ** 2,   # by   ((m/s)²/s)
])


def constant_bias(t: datetime) -> list[float]:
    """Known deterministic current bias [bx, by] in m/s."""
    return [0.02, -0.05]

_BIAS_PERIOD_S = 10 * 24 * 3600  # 10 days in seconds
_EPOCH = datetime(2000, 1, 1)     # arbitrary fixed reference

def bias(t: datetime) -> list[float]:
    """Sinusoidal current bias with a 10-day period, in m/s."""
    elapsed = (t - _EPOCH).total_seconds()
    phase = 2 * np.pi * elapsed / _BIAS_PERIOD_S
    return [0.1*np.sin(phase), 0.1* np.cos(phase)]