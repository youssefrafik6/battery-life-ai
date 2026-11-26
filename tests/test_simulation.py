import math

import pytest

from batterylife.simulation import percentile_summary, time_to_empty, trajectory


def test_constant_rate_and_physical_crossing():
    assert time_to_empty(75, 12) == 6.25
    assert time_to_empty(0, 0) == 0
    assert time_to_empty(75, 0) is None
    result = trajectory(75, [1 / 6, 8, 1], [12, 10, 0])
    assert result["soc"] == [75, 73, 0, 0]
    assert result["empty_hours"] == pytest.approx(1 / 6 + 7.3)


@pytest.mark.parametrize(
    "start,durations,rates",
    [
        (101, [1], [1]),
        (0, [0], [1]),
        (10, [1], [-1]),
        (10, [1], [math.inf]),
        (10, [1, 2], [1]),
    ],
)
def test_invalid_trajectory_inputs(start, durations, rates):
    with pytest.raises(ValueError):
        trajectory(start, durations, rates)


def test_unbounded_percentiles():
    result = percentile_summary([1, 2, math.inf, math.inf])
    assert result == {"p10": 1, "median": 2, "p90": None, "width": None}
