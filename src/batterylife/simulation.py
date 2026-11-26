"""Exact integration for piecewise-constant discharge."""

from math import isfinite

import numpy as np


def time_to_empty(start: float, rate: float) -> float | None:
    """Return hours, or None for positive charge at zero drain."""
    if not isfinite(start) or not 0 <= start <= 100 or not isfinite(rate) or rate < 0:
        raise ValueError("Invalid charge or drain rate")
    return 0.0 if start == 0 else (start / rate if rate else None)


def trajectory(start: float, durations, rates, *, include_crossing: bool = False) -> dict:
    """Integrate each segment and locate the first crossing of zero exactly."""
    empty = time_to_empty(start, 0)
    hours = [0.0]
    charges = [start]
    for duration, rate in zip(durations, rates, strict=True):
        crossing = time_to_empty(charges[-1], rate)
        if not isfinite(duration) or duration <= 0:
            raise ValueError("Durations must be finite and positive")
        if empty is None and crossing is not None and crossing <= duration:
            empty = float(hours[-1] + crossing)
        charge = float(max(0, charges[-1] - duration * rate))
        end = float(hours[-1] + duration)
        if include_crossing and crossing is not None and 0 < crossing < duration:
            hours.append(float(hours[-1] + crossing))
            charges.append(0.0)
        hours.append(end)
        charges.append(charge)
    return {"hours": hours, "soc": charges, "empty_hours": empty}


def percentile_summary(values) -> dict:
    """Order-statistic quantiles retain infinite time-to-empty outcomes."""
    p10, median, p90 = np.quantile(values, [0.1, 0.5, 0.9], method="inverted_cdf")
    return {
        "p10": float(p10) if np.isfinite(p10) else None,
        "median": float(median) if np.isfinite(median) else None,
        "p90": float(p90) if np.isfinite(p90) else None,
        "width": float(p90 - p10) if np.isfinite(p90) else None,
    }
