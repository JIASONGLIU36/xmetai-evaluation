"""Unit tests for centered and uncentered anomaly correlation."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Union

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.acc import ACC


ArrayLike = Union[list[list[float]], xr.DataArray]


def _field(values: ArrayLike) -> xr.DataArray:
    if isinstance(values, xr.DataArray):
        return values
    return xr.DataArray(values, dims=("lat", "lon"))


def _batch(
    forecast: ArrayLike,
    observation: ArrayLike,
    climatology: Optional[ArrayLike],
    *,
    valid_mask: Optional[Union[list[list[bool]], xr.DataArray]] = None,
    weights: Optional[ArrayLike] = None,
) -> EvaluationBatch:
    forecast_da = _field(forecast)
    observation_da = _field(observation)
    reference_da = _field(climatology) if climatology is not None else None
    if valid_mask is None:
        valid_mask_da = xr.ones_like(observation_da, dtype=bool)
    elif isinstance(valid_mask, xr.DataArray):
        valid_mask_da = valid_mask
    else:
        valid_mask_da = xr.DataArray(valid_mask, dims=observation_da.dims)
    weights_da = _field(weights) if weights is not None else None

    return EvaluationBatch(
        forecast=forecast_da,
        observation=observation_da,
        reference=reference_da,
        sample_keys=[{"case": "unit-test"}],
        valid_mask=valid_mask_da,
        weights=weights_da,
        alignment={"time": "same_valid_time", "space": "same_grid"},
        protocol_id="grid_valid_time",
    )


def test_acc_identical_anomalies_returns_one():
    """Identical non-constant anomalies have perfect positive correlation."""
    climatology = [[10.0, 10.0], [10.0, 10.0]]
    field = [[11.0, 12.0], [13.0, 14.0]]

    result = ACC().compute(_batch(field, field, climatology))

    assert result.status is ResultStatus.SUCCESS
    assert result.value == pytest.approx(1.0)
    assert result.unit == "1"


def test_acc_opposite_anomalies_returns_minus_one():
    """Opposite anomaly patterns have perfect negative correlation."""
    result = ACC().compute(
        _batch(
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        )
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.value == pytest.approx(-1.0)


def test_acc_uses_cosine_latitude_weights():
    """Weighted centered ACC matches a hand-confirmed two-latitude example."""
    result = ACC().compute(
        _batch(
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.0, 2.0], [1.0, 3.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            weights=[[1.0, 1.0], [0.5, 0.5]],
        )
    )

    assert result.value == pytest.approx(0.8004987358916188)
    assert result.weights_sum == pytest.approx(3.0)


def test_acc_uncentered_matches_formal_fuxi_configuration():
    """The formal FuXi configuration can use the uncentered anomaly cosine."""
    result = ACC(params={"centered": False}).compute(
        _batch(
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        )
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.value == pytest.approx(0.9938079899999065)


def test_acc_missing_climatology_is_partial_with_warning():
    """The documented zero-field fallback must be visible in result status."""
    field = [[1.0, 2.0], [3.0, 4.0]]

    result = ACC().compute(_batch(field, field, None))

    assert result.status is ResultStatus.PARTIAL
    assert result.value == pytest.approx(1.0)
    assert "No climatology provided, using zero anomaly" in result.warnings


def test_acc_zero_anomaly_variance_is_undefined():
    """A constant anomaly has zero variance, so ACC is undefined."""
    result = ACC().compute(
        _batch(
            [[2.0, 2.0], [2.0, 2.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        )
    )

    assert result.status is ResultStatus.UNDEFINED
    assert np.isnan(result.value)
    assert "Zero variance in forecast or observation anomaly" in result.warnings


def test_acc_all_points_invalid_returns_no_valid_data():
    """An empty valid domain must not be reported as a correlation."""
    result = ACC().compute(
        _batch(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            valid_mask=[[False, False], [False, False]],
        )
    )

    assert result.status is ResultStatus.NO_VALID_DATA
    assert np.isnan(result.value)
    assert result.n_valid == 0


def test_acc_nan_and_inf_are_excluded_by_valid_mask():
    """Non-finite points marked invalid must not affect ACC or valid count."""
    forecast = _field([[1.0, np.nan], [np.inf, 4.0]])
    observation = _field([[1.0, 2.0], [3.0, 4.0]])
    climatology = xr.zeros_like(observation)
    valid_mask = np.isfinite(forecast) & np.isfinite(observation)

    result = ACC().compute(
        _batch(forecast, observation, climatology, valid_mask=valid_mask)
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.value == pytest.approx(1.0)
    assert result.n_valid == 2


def test_acc_merged_chunks_equal_whole_batch():
    """Mergeable sufficient statistics must reproduce whole-field ACC."""
    metric = ACC()
    whole = _batch(
        [[1.0, 2.0], [4.0, 8.0]],
        [[1.0, 3.0], [5.0, 7.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    )
    top = _batch([[1.0, 2.0]], [[1.0, 3.0]], [[0.0, 0.0]])
    bottom = _batch([[4.0, 8.0]], [[5.0, 7.0]], [[0.0, 0.0]])

    whole_result = metric.compute(whole)
    merged_result = metric.finalize(
        metric.merge([metric.accumulate(top), metric.accumulate(bottom)])
    )

    assert merged_result.status is ResultStatus.SUCCESS
    assert merged_result.value == pytest.approx(whole_result.value)
    assert merged_result.n_valid == whole_result.n_valid


def test_acc_rejects_incompatible_states():
    """States from another metric or implementation version cannot be merged."""
    metric = ACC()
    state = metric.accumulate(_batch([[1.0]], [[1.0]], [[0.0]]))

    with pytest.raises(MetricError, match="different metrics"):
        metric.merge([state, replace(state, metric_name="rmse")])

    with pytest.raises(MetricError, match="different versions"):
        metric.merge([state, replace(state, metric_version="2.0.0")])
