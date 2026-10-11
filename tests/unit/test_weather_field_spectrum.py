"""Unit tests for deterministic-field power-spectrum metrics."""

from __future__ import annotations

from typing import Union

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.specialized import (
    PowerSpectrum,
    ZonalSpectrum,
    power_spectrum,
    zonal_spectrum,
)


ArrayLike = Union[np.ndarray, list[list[float]]]


def _field(values: ArrayLike, lat: np.ndarray | None = None) -> xr.DataArray:
    array = np.asarray(values, dtype="f8")
    if array.ndim == 2:
        latitude = np.arange(array.shape[0], dtype="f8") if lat is None else lat
        return xr.DataArray(
            array,
            dims=("lat", "lon"),
            coords={"lat": latitude, "lon": np.arange(array.shape[1])},
        )
    return xr.DataArray(array)


def _batch(
    forecast: ArrayLike,
    observation: ArrayLike,
    *,
    lat: np.ndarray | None = None,
) -> EvaluationBatch:
    forecast_da = _field(forecast, lat)
    observation_da = _field(observation, lat)
    return EvaluationBatch(
        forecast=forecast_da,
        observation=observation_da,
        reference=None,
        sample_keys=[{"case": "unit-test"}],
        valid_mask=xr.ones_like(observation_da, dtype=bool),
        weights=None,
        alignment={"time": "same_valid_time", "space": "same_grid"},
        protocol_id="grid_valid_time",
    )


def _direct_legacy_spectrum(values: np.ndarray, max_k: int) -> np.ndarray:
    """Small-grid implementation of the legacy A_t/B_t definition.

    This deliberately uses the documented double sum rather than the optimized
    FFT implementation under test.
    """
    field = np.asarray(values, dtype="f8")
    nlat, nlon = field.shape
    j = np.arange(nlat)[:, None]
    i = np.arange(nlon)[None, :]
    two_sided = np.zeros(nlon, dtype="f8")
    for k in range(nlon):
        for meridional_k in range(nlat):
            phase = 2.0 * np.pi * (
                meridional_k * (j - i) / nlat + k * i / nlon
            )
            coefficient = np.sum(field * np.exp(1j * phase))
            two_sided[k] += np.abs(coefficient) ** 2

    one_sided = np.zeros(max_k + 1, dtype="f8")
    for k in range(max_k + 1):
        if k == 0 or (nlon % 2 == 0 and k == nlon // 2):
            one_sided[k] = two_sided[k]
        else:
            one_sided[k] = two_sided[k] + two_sided[nlon - k]
    return one_sided


def test_power_spectrum_matches_documented_direct_formula():
    """The optimized two-dimensional FFT must match the legacy A_t/B_t sums."""
    field = np.array(
        [[1.0, 2.0, 0.0, -1.0, 3.0, 2.0], [4.0, 0.0, 1.0, 2.0, -2.0, 1.0]]
    )

    wavenumber, actual = power_spectrum(field, max_wavenumber=3)
    expected = _direct_legacy_spectrum(field, max_k=3)

    np.testing.assert_array_equal(wavenumber, np.arange(4))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_power_spectrum_puts_cosine_energy_at_its_wavenumber():
    """A pure zonal wave must not leak power into unrelated wavenumbers."""
    nlat, nlon, target_k = 3, 8, 2
    lon_index = np.arange(nlon)
    field = np.tile(np.cos(2.0 * np.pi * target_k * lon_index / nlon), (nlat, 1))

    _, power = power_spectrum(field)

    expected = np.zeros(nlon // 2 + 1)
    expected[target_k] = (nlat * nlon) ** 2 / 2.0
    np.testing.assert_allclose(power, expected, rtol=1e-12, atol=1e-10)


def test_power_spectrum_fills_nan_with_field_mean_and_truncates():
    """NaN handling and max-wavenumber truncation follow the legacy script."""
    field = np.array([[1.0, 2.0, np.nan, 4.0], [2.0, 3.0, 4.0, 5.0]])
    filled = np.where(np.isnan(field), np.nanmean(field), field)

    wavenumber, actual = power_spectrum(field, max_wavenumber=1)
    _, expected = power_spectrum(filled, max_wavenumber=1)

    np.testing.assert_array_equal(wavenumber, [0, 1])
    np.testing.assert_allclose(actual, expected)


def test_power_spectrum_rejects_non_two_dimensional_input():
    """The legacy two-dimensional spectrum has an explicit shape contract."""
    with pytest.raises(MetricError, match="需要二维场"):
        power_spectrum(np.ones((2, 3, 4)))


def test_power_spectrum_metric_identical_fields_have_ratio_one():
    """Identical fields produce identical curves and total-power ratio one."""
    field = [[1.0, 2.0, 4.0, 2.0], [0.0, 3.0, 1.0, 5.0]]

    result = PowerSpectrum(max_wavenumber=2).compute(_batch(field, field))

    assert result.status is ResultStatus.SUCCESS
    assert result.value["summary"]["power_ratio"] == pytest.approx(1.0)
    assert result.curve["kind"] == "wavenumber_spectrum"
    np.testing.assert_array_equal(result.curve["wavenumber"], [0, 1, 2])
    np.testing.assert_allclose(
        result.curve["power_forecast"], result.curve["power_observation"]
    )


def test_power_spectrum_metric_scales_quadratically():
    """Multiplying a field by three multiplies its spectral power by nine."""
    observation = np.array([[1.0, 2.0, 4.0, 2.0], [0.0, 3.0, 1.0, 5.0]])

    result = PowerSpectrum(max_wavenumber=2).compute(
        _batch(3.0 * observation, observation)
    )

    assert result.value["summary"]["power_ratio"] == pytest.approx(9.0)
    np.testing.assert_allclose(
        result.curve["power_forecast"],
        9.0 * result.curve["power_observation"],
    )


def test_power_spectrum_merged_states_average_sample_curves():
    """Merged per-case spectra equal the arithmetic mean of their curves."""
    metric = PowerSpectrum(max_wavenumber=2)
    first = _batch(
        [[1.0, 0.0, -1.0, 0.0], [1.0, 0.0, -1.0, 0.0]],
        [[1.0, 1.0, -1.0, -1.0], [1.0, 1.0, -1.0, -1.0]],
    )
    second = _batch(
        [[2.0, 0.0, -2.0, 0.0], [2.0, 0.0, -2.0, 0.0]],
        [[0.5, 0.5, -0.5, -0.5], [0.5, 0.5, -0.5, -0.5]],
    )

    first_result = metric.compute(first)
    second_result = metric.compute(second)
    merged = metric.finalize(metric.merge([metric.accumulate(first), metric.accumulate(second)]))

    np.testing.assert_allclose(
        merged.curve["power_forecast"],
        (first_result.curve["power_forecast"] + second_result.curve["power_forecast"])
        / 2.0,
    )
    np.testing.assert_allclose(
        merged.curve["power_observation"],
        (
            first_result.curve["power_observation"]
            + second_result.curve["power_observation"]
        )
        / 2.0,
    )
    assert merged.n_valid == 2


def test_zonal_spectrum_removes_mean_and_finds_pure_wave():
    """The formal zonal spectrum removes k=0 and retains the selected wave."""
    lat = np.array([0.0, 60.0])
    nlon, target_k = 8, 2
    wave = np.cos(2.0 * np.pi * target_k * np.arange(nlon) / nlon)
    field = np.stack([10.0 + wave, -4.0 + wave])

    power = zonal_spectrum(field, lat)

    expected = np.zeros(nlon // 2 + 1)
    expected[target_k] = 0.25
    np.testing.assert_allclose(power, expected, rtol=1e-12, atol=1e-12)


def test_zonal_spectrum_uses_cosine_latitude_weights():
    """Each latitude's wave power is averaged with cos(latitude) weights."""
    lat = np.array([0.0, 60.0])
    nlon, target_k = 8, 1
    wave = np.cos(2.0 * np.pi * target_k * np.arange(nlon) / nlon)
    field = np.stack([wave, 2.0 * wave])

    power = zonal_spectrum(field, lat)

    expected_power = (1.0 * (1.0**2 / 4.0) + 0.5 * (2.0**2 / 4.0)) / 1.5
    assert power[target_k] == pytest.approx(expected_power)


def test_zonal_spectrum_metric_returns_truncated_curve():
    """The formal metric exposes the requested wavenumbers through result.curve."""
    lat = np.array([0.0, 60.0])
    observation = np.array(
        [[1.0, 0.0, -1.0, 0.0], [2.0, 0.0, -2.0, 0.0]]
    )

    result = ZonalSpectrum(max_wavenumber=1).compute(
        _batch(2.0 * observation, observation, lat=lat)
    )

    assert result.status is ResultStatus.SUCCESS
    np.testing.assert_array_equal(result.curve["wavenumber"], [0, 1])
    assert result.value["summary"]["power_ratio"] == pytest.approx(4.0)
    np.testing.assert_allclose(
        result.curve["power_forecast"],
        4.0 * result.curve["power_observation"],
    )
