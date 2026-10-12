"""Tests for deterministic, latitude-weighted mean error (Bias)."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Union

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.bias import Bias


def _batch(
    forecast: Union[list[list[float]], xr.DataArray, xr.Dataset],
    observation: Union[list[list[float]], xr.DataArray, xr.Dataset],
    *,
    valid_mask: Optional[Union[list[list[bool]], xr.DataArray]] = None,
    weights: Optional[Union[list[list[float]], xr.DataArray]] = None,
) -> EvaluationBatch:
    """Build the smallest valid grid batch accepted by the metric contract."""
    if not isinstance(forecast, (xr.DataArray, xr.Dataset)):
        forecast = xr.DataArray(forecast, dims=("lat", "lon"), attrs={"units": "K"})
    if not isinstance(observation, (xr.DataArray, xr.Dataset)):
        observation = xr.DataArray(
            observation, dims=("lat", "lon"), attrs={"units": "K"}
        )

    mask_template = (
        observation[list(observation.data_vars)[0]]
        if isinstance(observation, xr.Dataset)
        else observation
    )
    if valid_mask is None:
        valid_mask = xr.ones_like(mask_template, dtype=bool)
    elif not isinstance(valid_mask, xr.DataArray):
        valid_mask = xr.DataArray(valid_mask, dims=mask_template.dims)
    if weights is not None and not isinstance(weights, xr.DataArray):
        weights = xr.DataArray(weights, dims=mask_template.dims)

    return EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=[{"case": "unit-test"}],
        valid_mask=valid_mask,
        weights=weights,
        alignment={"time": "same_valid_time", "space": "same_grid"},
        protocol_id="grid_valid_time",
    )


def test_bias_identical_fields_returns_zero():
    """Identical forecast and observation fields have zero mean error."""
    result = Bias().compute(
        _batch([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]])
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.value == pytest.approx(0.0)
    assert result.n_valid == 4
    assert result.aggregation == "mean_over_samples"
    assert result.product_kind == "deterministic"


def test_bias_preserves_forecast_minus_observation_sign():
    """Positive and negative errors must not be converted to absolute errors."""
    result = Bias().compute(
        _batch([[3.0, 0.0], [5.0, 2.0]], [[1.0, 2.0], [4.0, 4.0]])
    )

    # Errors are [2, -2, 1, -2], whose arithmetic mean is -0.25.
    assert result.value == pytest.approx(-0.25)
    assert result.weights_sum == pytest.approx(4.0)


def test_bias_uses_cosine_latitude_weights():
    """Errors at 60 degrees latitude receive half the equatorial weight."""
    result = Bias().compute(
        _batch(
            [[2.0, 2.0], [-4.0, -4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            weights=[[1.0, 1.0], [0.5, 0.5]],
        )
    )

    expected = (2.0 * 2.0 + 2.0 * 0.5 * -4.0) / 3.0
    assert result.value == pytest.approx(expected)
    assert result.weights_sum == pytest.approx(3.0)


def test_bias_excludes_points_outside_valid_mask():
    """A large error at an invalid point must not affect the score."""
    result = Bias().compute(
        _batch(
            [[2.0, 2.0], [2.0, 1000.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            valid_mask=[[True, True], [True, False]],
        )
    )

    assert result.value == pytest.approx(2.0)
    assert result.n_valid == 3
    assert result.weights_sum == pytest.approx(3.0)


def test_bias_merged_chunks_equal_whole_batch():
    """Accumulating chunks must give the same result as one whole batch."""
    metric = Bias()
    whole = _batch([[3.0, 0.0], [5.0, 2.0]], [[1.0, 2.0], [4.0, 4.0]])
    top = _batch([[3.0, 0.0]], [[1.0, 2.0]])
    bottom = _batch([[5.0, 2.0]], [[4.0, 4.0]])

    whole_result = metric.compute(whole)
    merged_result = metric.finalize(
        metric.merge([metric.accumulate(top), metric.accumulate(bottom)])
    )

    assert merged_result.value == pytest.approx(whole_result.value)
    assert merged_result.n_valid == whole_result.n_valid
    assert merged_result.weights_sum == pytest.approx(whole_result.weights_sum)


def test_bias_all_points_invalid_returns_no_valid_data():
    """An empty valid domain must not be reported as zero Bias."""
    result = Bias().compute(
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


def test_bias_zero_weight_sum_returns_no_valid_data():
    """A non-empty field with zero total weight has no defined weighted Bias."""
    result = Bias().compute(
        _batch(
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            weights=[[0.0, 0.0], [0.0, 0.0]],
        )
    )

    assert result.status is ResultStatus.NO_VALID_DATA
    assert np.isnan(result.value)
    assert result.n_valid == 0


def test_bias_nan_and_inf_are_excluded_by_valid_mask():
    """Non-finite values marked invalid must not affect the score or count."""
    forecast = xr.DataArray(
        [[1.0, np.nan], [np.inf, -3.0]],
        dims=("lat", "lon"),
        attrs={"units": "K"},
    )
    observation = xr.zeros_like(forecast)
    valid_mask = np.isfinite(forecast) & np.isfinite(observation)

    result = Bias().compute(_batch(forecast, observation, valid_mask=valid_mask))

    assert result.status is ResultStatus.SUCCESS
    assert result.value == pytest.approx(-1.0)
    assert result.n_valid == 2
    assert result.weights_sum == pytest.approx(2.0)


def test_bias_accepts_single_variable_dataset_and_rejects_ambiguous_dataset():
    """Dataset input is valid only after routing it to exactly one variable."""
    forecast = xr.Dataset({"t2m": (("lat", "lon"), [[2.0]])})
    observation = xr.Dataset({"t2m": (("lat", "lon"), [[1.0]])})
    assert Bias().compute(_batch(forecast, observation)).value == pytest.approx(1.0)

    ambiguous = forecast.assign(z500=forecast["t2m"])
    with pytest.raises(MetricError, match="一次只评一个变量"):
        Bias().compute(_batch(ambiguous, observation))


def test_bias_rejects_empty_and_incompatible_states():
    """Merge must reject empty, cross-metric, and cross-version state lists."""
    metric = Bias()
    state = metric.accumulate(_batch([[1.0]], [[0.0]]))

    with pytest.raises(MetricError, match="empty states"):
        metric.merge([])
    with pytest.raises(MetricError, match="different metrics"):
        metric.merge([state, replace(state, metric_name="rmse")])
    with pytest.raises(MetricError, match="different versions"):
        metric.merge([state, replace(state, metric_version="2.0.0")])
