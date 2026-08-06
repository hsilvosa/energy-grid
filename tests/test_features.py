from datetime import UTC, datetime

import pandas as pd
import pytest

from energy_grid.features import asof_features, validate_asof


def test_asof_join_never_uses_future_publication() -> None:
    measurements = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:15Z"]),
            "published_at": pd.to_datetime(["2025-01-01T00:10Z", "2025-01-01T00:30Z"]),
            "value": [100.0, 200.0],
        }
    )
    origins = pd.DataFrame(
        {
            "forecast_origin": pd.to_datetime(["2025-01-01T00:20Z", "2025-01-01T00:35Z"]),
            "valid_time": pd.to_datetime(["2025-01-01T00:30Z", "2025-01-01T00:45Z"]),
        }
    )
    result = asof_features(measurements, origins)
    assert result["value"].tolist() == [100.0, 200.0]
    assert (result["published_at"] <= result["forecast_origin"]).all()


def test_validate_asof_rejects_future_data() -> None:
    with pytest.raises(ValueError, match="after forecast origin"):
        validate_asof(
            datetime(2025, 1, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)
        )

