from __future__ import annotations

from datetime import datetime

import numpy as np

# Process noise scale parameters.
# sigma_pos:  unmodelled position diffusion in metres per sqrt(hour).
# sigma_bias: bias random-walk rate in (m/hr) per sqrt(hour).
_SIGMA_POS  = 50.0
_SIGMA_BIAS = 0.01

Q_DEFAULT: np.ndarray = np.diag([
    _SIGMA_POS  ** 2,   # x variance rate  (m² / hr)
    _SIGMA_POS  ** 2,   # y variance rate
    _SIGMA_BIAS ** 2,   # bx variance rate ((m/hr)² / hr)
    _SIGMA_BIAS ** 2,   # by variance rate
])


def bias(t: datetime) -> list[float]:
    """Known deterministic bias [bx, by] in m/hr. Constant for now; t reserved for future use."""
    return [0.1, 0.1]
