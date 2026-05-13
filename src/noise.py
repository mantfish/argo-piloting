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

_SIGMA_POS  = 50.0 / 60.0          # m/sqrt(s)
_SIGMA_BIAS = 0.001 / 60.0         # (m/s)/sqrt(s)

Q_DEFAULT: np.ndarray = np.diag([
    _SIGMA_POS  ** 2,   # x    (m²/s)
    _SIGMA_POS  ** 2,   # y    (m²/s)
    _SIGMA_BIAS ** 2,   # bx   ((m/s)²/s)
    _SIGMA_BIAS ** 2,   # by   ((m/s)²/s)
])


def bias(t: datetime) -> list[float]:
    """Known deterministic current bias [bx, by] in m/s."""
    return [0.0, 0.0]
