"""Tests for deterministic, latitude-weighted RMSE."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Union

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.rmse import RMSE


def _batch(
    forecast: Union[list[list[float]], xr.DataArray],
    observation: Union[list[list[float]], xr.DataArray],
    *,
    valid_mask: Optional[Union[list[list[bool]], xr.DataArray]] = None,
    weights: Optional[Union[list[list[float]], xr.DataArray]] = None,
) -> EvaluationBatch:
    """Build the smallest valid grid batch accepted by the metric contract."""
    if not isinstance(forecast, xr.DataArray):
        forecast = xr.DataArray(forecast, dims=("lat", "lon"), attrs={"units": "m"})
    if not isinstance(observation, xr.DataArray):
        observation = xr.DataArray(
            observation, dims=("lat", "lon"), attrs={"units": "m"}
        )
    if valid_mask is None:
        valid_mask = xr.ones_like(observation, dtype=bool)
    elif not isinstance(valid_mask, xr.DataArray):
        valid_mask = xr.DataArray(valid_mask, dims=observation.dims)
    if weights is not None and not isinstance(weights, xr.DataArray):
        weights = xr.DataArray(weights, dims=observation.dims)

    return EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=[{"case": "unit-test"}],
        valid_mask=valid_mask,
        weights=weights,
        alignment={"time": "same_valid_time", "space": "same_grid"},
        protocol_id="grid_valid_time",
    )


def test_rmse_identical_fields_returns_zero():
    """Identical forecast and observation fields have zero error."""
    result = RMSE().compute(
        _batch([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]])
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.value == pytest.approx(0.0)
    assert result.n_valid == 4
    assert result.unit == "m"


def test_rmse_known_values_matches_hand_calculation():
    """RMSE=sqrt((0^2+3^2+4^2+0^2)/4)=2.5 for this field."""
    result = RMSE().compute(
        _batch([[0.0, 3.0], [4.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.value == pytest.approx(2.5)
    assert result.weights_sum == pytest.approx(4.0)


def test_rmse_uses_cosine_latitude_weights():
    """Errors at 60 degrees latitude receive half the weight of equatorial errors."""
    weights = [[1.0, 1.0], [0.5, 0.5]]  # cos(0 deg), cos(60 deg)
    result = RMSE().compute(
        _batch(
            [[1.0, 1.0], [3.0, 3.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            weights=weights,
        )
    )

    expected = np.sqrt((2.0 * 1.0**2 + 2.0 * 0.5 * 3.0**2) / 3.0)
    assert result.value == pytest.approx(expected)
    assert result.weights_sum == pytest.approx(3.0)


def test_rmse_excludes_points_outside_valid_mask():
    """A large error at an invalid point must not affect the score."""
    result = RMSE().compute(
        _batch(
            [[1.0, 1.0], [1.0, 1000.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            valid_mask=[[True, True], [True, False]],
        )
    )

    assert result.value == pytest.approx(1.0)
    assert result.n_valid == 3
    assert result.weights_sum == pytest.approx(3.0)


def test_rmse_merged_chunks_equal_whole_batch():
    """Accumulating two chunks must give the same result as one whole batch."""
    metric = RMSE()
    whole = _batch([[0.0, 3.0], [4.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    top = _batch([[0.0, 3.0]], [[0.0, 0.0]])
    bottom = _batch([[4.0, 0.0]], [[0.0, 0.0]])

    whole_result = metric.compute(whole)
    merged_result = metric.finalize(
        metric.merge([metric.accumulate(top), metric.accumulate(bottom)])
    )

    assert merged_result.value == pytest.approx(whole_result.value)
    assert merged_result.n_valid == whole_result.n_valid
    assert merged_result.weights_sum == pytest.approx(whole_result.weights_sum)


def test_rmse_all_points_invalid_returns_no_valid_data():
    """An empty valid domain must be reported explicitly, not as RMSE=0."""
    result = RMSE().compute(
        _batch(
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            valid_mask=[[False, False], [False, False]],
        )
    )

    assert result.status is ResultStatus.NO_VALID_DATA
    assert np.isnan(result.value)
    assert result.n_valid == 0
    assert "No valid data points" in result.warnings


def test_rmse_zero_weight_sum_returns_no_valid_data():
    """A non-empty field with zero total weight has no defined weighted RMSE."""
    result = RMSE().compute(
        _batch(
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            weights=[[0.0, 0.0], [0.0, 0.0]],
        )
    )

    assert result.status is ResultStatus.NO_VALID_DATA
    assert np.isnan(result.value)
    assert result.n_valid == 0


def test_rmse_nan_and_inf_are_excluded_by_valid_mask():
    """Non-finite values marked invalid must not affect the score or count."""
    forecast = xr.DataArray(
        [[1.0, np.nan], [np.inf, 4.0]],
        dims=("lat", "lon"),
        attrs={"units": "m"},
    )
    observation = xr.zeros_like(forecast)
    observation.attrs["units"] = "m"
    valid_mask = np.isfinite(forecast) & np.isfinite(observation)

    result = RMSE().compute(
        _batch(forecast, observation, valid_mask=valid_mask)
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.value == pytest.approx(np.sqrt((1.0**2 + 4.0**2) / 2.0))
    assert result.n_valid == 2
    assert result.weights_sum == pytest.approx(2.0)


def test_rmse_rejects_incompatible_states():
    """States from another metric or implementation version cannot be merged."""
    metric = RMSE()
    state = metric.accumulate(_batch([[1.0]], [[0.0]]))

    with pytest.raises(MetricError, match="different metrics"):
        metric.merge([state, replace(state, metric_name="bias")])

    with pytest.raises(MetricError, match="different versions"):
        metric.merge([state, replace(state, metric_version="2.0.0")])
