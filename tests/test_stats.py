"""Wilson score interval."""

from __future__ import annotations

import pytest

from cot_faithcheck.stats import wilson_interval


def test_no_data_is_full_range():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_bounds_are_ordered_and_clamped():
    for succ, n in [(0, 5), (3, 5), (5, 5), (1, 20), (19, 20)]:
        low, high = wilson_interval(succ, n)
        assert 0.0 <= low <= high <= 1.0


def test_interval_contains_point_estimate():
    low, high = wilson_interval(7, 10)
    assert low <= 0.7 <= high


def test_extremes_do_not_pin_to_zero_width():
    # All successes: Wilson keeps a lower bound below 1 (unlike the naive normal CI).
    low, high = wilson_interval(5, 5)
    assert high == pytest.approx(1.0)
    assert low < 1.0
    # All failures: upper bound above 0.
    low, high = wilson_interval(0, 5)
    assert low == pytest.approx(0.0)
    assert high > 0.0


def test_more_data_narrows_interval():
    low_small, high_small = wilson_interval(3, 5)
    low_big, high_big = wilson_interval(30, 50)
    assert (high_big - low_big) < (high_small - low_small)
