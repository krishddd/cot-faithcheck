"""Small statistics helpers — Wilson score intervals for proportions.

Every agreement rate is a proportion estimated from a finite number of k-run
Bernoulli trials, so a point estimate alone is misleading at small ``k``. The
Wilson score interval is the right tool for proportions near 0 or 1 (where the
normal approximation fails) and for small ``n`` — exactly our regime.
"""

from __future__ import annotations

import math
from typing import Tuple

# z for a 95% two-sided interval.
Z_95 = 1.959963984540054


def wilson_interval(successes: float, n: int, *, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score interval for ``successes``/``n`` at confidence ``z``.

    ``successes`` may be fractional (e.g. a pooled mean times n) and is clamped to
    ``[0, n]``. With ``n == 0`` the interval is the whole range ``[0, 1]`` — no
    data, no confidence.
    """
    if n <= 0:
        return (0.0, 1.0)
    successes = max(0.0, min(float(n), float(successes)))
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)
