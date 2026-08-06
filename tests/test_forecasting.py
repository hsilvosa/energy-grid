from datetime import UTC, date, datetime

import numpy as np
import pytest

from energy_grid.domain import EventType
from energy_grid.forecasting import evaluate, materialize_demo_forecasts


def test_metrics_support_negative_prices_without_mape() -> None:
    actual = np.array([-5.0, 0.0, 10.0])
    point = np.array([-4.0, 1.0, 8.0])
    metrics = evaluate(actual, point, point - 2, point + 2, allow_wape=False)
    assert metrics.mae == pytest.approx(4 / 3)
    assert metrics.wape is None
    assert metrics.interval_coverage == 1.0


def test_empty_metrics_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        evaluate(np.array([]), np.array([]), np.array([]), np.array([]))


def test_demo_demand_forecast_has_24_ordered_quantile_steps() -> None:
    forecasts = materialize_demo_forecasts(
        EventType.DEMAND, issue_time=datetime(2025, 1, 1, tzinfo=UTC)
    )
    assert len(forecasts) == 24
    assert all(item.p10 <= item.point <= item.p90 for item in forecasts)


def test_day_ahead_forecast_respects_spring_dst() -> None:
    forecasts = materialize_demo_forecasts(
        EventType.PRICE,
        issue_time=datetime(2025, 3, 29, 9, tzinfo=UTC),
        delivery_date=date(2025, 3, 30),
    )
    assert len(forecasts) == 92

