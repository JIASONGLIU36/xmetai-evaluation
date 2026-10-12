"""Unit tests for deterministic-field spherical-band power metrics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.specialized import (
    DEFAULT_SPHERICAL_BANDS,
    SphericalBands,
    spherical_band_power,
    validate_global_lat,
)


LAT = np.linspace(90.0, -90.0, 17)
LON = np.arange(16, dtype="f8") * (360.0 / 16.0)
TEST_BANDS = ((1, 1), (2, 2), (3, 3))


def _components() -> tuple[np.ndarray, np.ndarray]:
    """Return real spherical-harmonic shapes with degrees 1 and 2."""
    theta = np.deg2rad(90.0 - LAT)[:, None]
    longitude = np.deg2rad(LON)[None, :]
    degree_1 = np.broadcast_to(np.cos(theta), (LAT.size, LON.size)).copy()
    degree_2 = np.sin(theta) ** 2 * np.cos(2.0 * longitude)
    return degree_1, degree_2


def _field(values: np.ndarray, *, include_lat: bool = True) -> xr.DataArray:
    array = np.asarray(values, dtype="f8")
    if array.ndim == 2:
        dims = ("lat", "lon") if include_lat else ("y", "lon")
        coords = {dims[0]: LAT, "lon": LON}
    elif array.ndim == 3:
        dims = ("sample", "lat", "lon")
        coords = {"sample": np.arange(array.shape[0]), "lat": LAT, "lon": LON}
    else:
        raise ValueError("test field must be two- or three-dimensional")
    return xr.DataArray(array, dims=dims, coords=coords)


def _batch(forecast: np.ndarray, observation: np.ndarray) -> EvaluationBatch:
    forecast_da = _field(forecast)
    observation_da = _field(observation)
    sample_count = 1 if forecast_da.ndim == 2 else forecast_da.sizes["sample"]
    return EvaluationBatch(
        forecast=forecast_da,
        observation=observation_da,
        sample_keys=[{"sample": index} for index in range(sample_count)],
        valid_mask=xr.ones_like(observation_da, dtype=bool),
        alignment={"time": "same_valid_time", "space": "same_global_grid"},
        protocol_id="grid_valid_time",
    )


def test_default_spherical_bands_match_formal_configuration():
    assert DEFAULT_SPHERICAL_BANDS == (
        (1, 4),
        (5, 20),
        (21, 40),
        (41, 64),
        (65, 128),
    )


def test_validate_global_lat_accepts_both_row_orders():
    validate_global_lat(LAT)
    validate_global_lat(LAT[::-1])


@pytest.mark.parametrize(
    ("lat", "message"),
    [
        (np.array([[90.0, 0.0, -90.0]]), "一维有限数组"),
        (np.array([90.0, np.nan, -90.0]), "一维有限数组"),
        (np.array([91.0, 0.0, -90.0]), r"\|lat\| <= 90"),
        (np.array([90.0, 30.0, 0.0, -90.0]), "等距网格"),
        (np.linspace(60.0, -60.0, 7), "全球含极网格"),
    ],
    ids=["not_1d", "not_finite", "outside_degrees", "not_regular", "regional"],
)
def test_validate_global_lat_rejects_invalid_grids(lat: np.ndarray, message: str):
    with pytest.raises(ValueError, match=message):
        validate_global_lat(lat)


def test_spherical_band_power_requires_matching_latitude_coordinates():
    field = np.ones((LAT.size, LON.size))

    with pytest.raises(ValueError, match="需要 lat"):
        spherical_band_power(field, bands=TEST_BANDS)
    with pytest.raises(ValueError, match="lat 长度"):
        spherical_band_power(field, lat=LAT[:-1], bands=TEST_BANDS)


def test_spherical_band_power_rejects_grid_too_small_for_requested_degree():
    lat = np.linspace(90.0, -90.0, 9)
    field = np.ones((9, 8))

    with pytest.raises(ValueError, match="grid too small"):
        spherical_band_power(field, lat=lat, bands=((1, 4),))


def test_spherical_band_power_places_known_waves_in_expected_degrees():
    degree_1, degree_2 = _components()

    power_1 = spherical_band_power(degree_1, lat=LAT, bands=TEST_BANDS)
    power_2 = spherical_band_power(degree_2, lat=LAT, bands=TEST_BANDS)

    assert power_1[0] > 0.0
    assert power_2[1] > 0.0
    np.testing.assert_allclose(power_1[1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(power_2[[0, 2]], 0.0, atol=1e-12)


def test_spherical_band_power_is_invariant_to_latitude_row_reversal():
    degree_1, degree_2 = _components()
    field = degree_1 + 0.5 * degree_2

    descending = spherical_band_power(field, lat=LAT, bands=TEST_BANDS)
    ascending = spherical_band_power(
        field[::-1], lat=LAT[::-1], bands=TEST_BANDS
    )

    np.testing.assert_allclose(ascending, descending, rtol=1e-6, atol=1e-12)


def test_spherical_band_power_scales_quadratically_and_preserves_batch_shape():
    degree_1, degree_2 = _components()
    field = degree_1 + degree_2
    samples = np.stack([field, 2.0 * field, 3.0 * field])

    actual = spherical_band_power(samples, lat=LAT, bands=TEST_BANDS, block=1)
    reference = spherical_band_power(field, lat=LAT, bands=TEST_BANDS)

    assert actual.shape == (3, len(TEST_BANDS))
    np.testing.assert_allclose(actual[0], reference)
    np.testing.assert_allclose(actual[1], 4.0 * reference, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(actual[2], 9.0 * reference, rtol=1e-6, atol=1e-12)


def test_spherical_bands_metric_reports_each_band_and_power_ratio():
    degree_1, degree_2 = _components()
    observation = degree_1 + degree_2

    result = SphericalBands(bands=TEST_BANDS).compute(
        _batch(2.0 * observation, observation)
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.aggregation == "spherical_band_power"
    assert result.n_valid == 1
    assert result.unit == "1"
    assert set(result.value) == {"1_1", "2_2", "3_3"}
    for key in ("1_1", "2_2", "3_3"):
        assert result.value[key]["spherical_pred"] == pytest.approx(
            4.0 * result.value[key]["spherical_obs"], rel=1e-6
        )
        assert result.value[key]["spherical_ratio"] == pytest.approx(4.0, rel=1e-6)
    resolved_power = (
        result.value["1_1"]["spherical_obs"]
        + result.value["2_2"]["spherical_obs"]
    )
    assert result.value["3_3"]["spherical_obs"] < 1e-12 * resolved_power


def test_spherical_bands_merged_states_equal_whole_batch():
    degree_1, degree_2 = _components()
    metric = SphericalBands(bands=TEST_BANDS, block=1)
    first = _batch(2.0 * degree_1, degree_1)
    second = _batch(3.0 * degree_2, degree_2)
    whole = _batch(
        np.stack([2.0 * degree_1, 3.0 * degree_2]),
        np.stack([degree_1, degree_2]),
    )

    merged = metric.finalize(
        metric.merge([metric.accumulate(first), metric.accumulate(second)])
    )
    expected = metric.compute(whole)

    assert merged.n_valid == expected.n_valid == 2
    for band in ("1_1", "2_2", "3_3"):
        for name in ("spherical_pred", "spherical_obs", "spherical_ratio"):
            assert merged.value[band][name] == pytest.approx(
                expected.value[band][name], nan_ok=True
            )


def test_spherical_bands_rejects_missing_lat_and_regional_grid():
    degree_1, _ = _components()
    missing_lat = _batch(degree_1, degree_1)
    missing_lat = replace(
        missing_lat,
        forecast=_field(degree_1, include_lat=False),
        observation=_field(degree_1, include_lat=False),
    )

    with pytest.raises(MetricError, match="需要 forecast 携带 lat"):
        SphericalBands(bands=TEST_BANDS).compute(missing_lat)

    regional = _batch(degree_1, degree_1)
    regional_lat = np.linspace(60.0, -60.0, LAT.size)
    regional.forecast.coords["lat"] = regional_lat
    regional.observation.coords["lat"] = regional_lat
    with pytest.raises(MetricError, match="全球含极网格"):
        SphericalBands(bands=TEST_BANDS).compute(regional)


def test_spherical_bands_rejects_empty_and_incompatible_states():
    degree_1, degree_2 = _components()
    metric = SphericalBands(bands=TEST_BANDS)
    state = metric.accumulate(_batch(degree_1 + degree_2, degree_1 + degree_2))

    with pytest.raises(MetricError, match="empty states"):
        metric.merge([])
    with pytest.raises(MetricError, match="different metrics"):
        metric.merge([state, replace(state, metric_name="zonal_spectrum")])
