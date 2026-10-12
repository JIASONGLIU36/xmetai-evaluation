"""Unit tests for deterministic grid matching and alignment."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import (
    DataBundle,
    EvaluationBatch,
    Provenance,
    SemanticMetadata,
)
from xmetai_evaluation.core.errors import AlignmentError
from xmetai_evaluation.core.variables import DataKind, TemporalKind
from xmetai_evaluation.pipeline.matcher import (
    Matcher,
    add_derived_variables,
    align_lon_to,
    align_to_grid,
    narrow_batch,
    restrict_to_latitude_band,
)


def _grid_dataset(lat, lon, values=None, name="z500") -> xr.Dataset:
    lat = np.asarray(lat, dtype="f8")
    lon = np.asarray(lon, dtype="f8")
    if values is None:
        values = np.arange(lat.size * lon.size, dtype="f8").reshape(
            lat.size, lon.size
        )
    return xr.Dataset(
        {name: (("lat", "lon"), np.asarray(values, dtype="f8"))},
        coords={"lat": lat, "lon": lon},
    )


def _batch(
    forecast,
    observation,
    *,
    reference=None,
    members=None,
    valid_mask=None,
) -> EvaluationBatch:
    if valid_mask is None:
        if isinstance(observation, xr.Dataset):
            first = next(iter(observation.data_vars.values()))
            valid_mask = xr.ones_like(first, dtype=bool)
        else:
            valid_mask = xr.ones_like(observation, dtype=bool)
    return EvaluationBatch(
        forecast=forecast,
        observation=observation,
        reference=reference,
        members=members,
        sample_keys=[{"case": "grid-matcher-unit"}],
        valid_mask=valid_mask,
        alignment={"time": "same_valid_time", "space": "test_grid"},
        protocol_id="grid_valid_time",
    )


def _bundle(dataset: xr.Dataset, kind: DataKind, source_id: str) -> DataBundle:
    return DataBundle(
        payload=dataset,
        kind=kind,
        source_id=source_id,
        standard_vars={name: name for name in dataset.data_vars},
        semantic=SemanticMetadata(
            units={name: "1" for name in dataset.data_vars},
            temporal_kind=TemporalKind.INSTANTANEOUS,
            grid_type="regular_ll",
        ),
        provenance=Provenance(
            input_files=[],
            reader_id="unit-test",
            reader_version="1",
        ),
    )


def test_align_lon_to_wraps_negative_longitudes_and_reorders_values():
    dataset = _grid_dataset(
        [0.0],
        [-180.0, -90.0, 0.0, 90.0],
        [[-180.0, -90.0, 0.0, 90.0]],
    )

    actual = align_lon_to(dataset, [0.0, 90.0, 180.0, 270.0])

    np.testing.assert_array_equal(actual.lon, [0.0, 90.0, 180.0, 270.0])
    np.testing.assert_array_equal(actual.z500, [[0.0, 90.0, -180.0, -90.0]])


def test_align_lon_to_wraps_positive_longitudes_to_negative_circle():
    dataset = _grid_dataset(
        [0.0],
        [0.0, 90.0, 180.0, 270.0],
        [[0.0, 90.0, 180.0, 270.0]],
    )

    actual = align_lon_to(dataset, [-180.0, -90.0, 0.0, 90.0])

    np.testing.assert_array_equal(actual.lon, [-180.0, -90.0, 0.0, 90.0])
    np.testing.assert_array_equal(actual.z500, [[180.0, 270.0, 0.0, 90.0]])


def test_align_lon_to_leaves_same_convention_and_missing_coordinate_unchanged():
    dataset = _grid_dataset([0.0], [0.0, 90.0])
    without_lon = xr.Dataset({"value": ("point", [1.0, 2.0])})

    assert align_lon_to(dataset, [0.0, 180.0]) is dataset
    assert align_lon_to(without_lon, [0.0, 180.0]) is without_lon


def test_align_to_grid_rewrites_coordinates_within_tolerance():
    dataset = _grid_dataset(
        [-90.0 + 5e-7, 0.0, 90.0 - 5e-7],
        [0.0 + 5e-7, 90.0, 180.0, 270.0 - 5e-7],
    )
    target_lat = np.array([-90.0, 0.0, 90.0])
    target_lon = np.array([0.0, 90.0, 180.0, 270.0])

    actual = align_to_grid(dataset, target_lat, target_lon)

    assert actual is not None
    np.testing.assert_array_equal(actual.lat, target_lat)
    np.testing.assert_array_equal(actual.lon, target_lon)
    np.testing.assert_array_equal(actual.z500, dataset.z500)


def test_align_to_grid_flips_latitude_and_rolls_longitude_without_interpolation():
    source_lat = np.array([90.0, 0.0, -90.0])
    source_lon = np.array([180.0, 270.0, 0.0, 90.0])
    values = source_lat[:, None] * 1000.0 + source_lon[None, :]
    dataset = _grid_dataset(source_lat, source_lon, values)
    target_lat = source_lat[::-1]
    target_lon = np.array([0.0, 90.0, 180.0, 270.0])

    actual = align_to_grid(dataset, target_lat, target_lon)

    assert actual is not None
    expected = target_lat[:, None] * 1000.0 + target_lon[None, :]
    np.testing.assert_array_equal(actual.z500, expected)
    np.testing.assert_array_equal(actual.lat, target_lat)
    np.testing.assert_array_equal(actual.lon, target_lon)


@pytest.mark.parametrize(
    "dataset,target_lat,target_lon",
    [
        (xr.Dataset({"z500": ("point", [1.0])}), [0.0], [0.0]),
        (_grid_dataset([0.0, 1.0], [0.0, 1.0]), [0.0], [0.0, 1.0]),
        (_grid_dataset([0.0, 2.0], [0.0, 1.0]), [0.0, 1.0], [0.0, 1.0]),
        (_grid_dataset([0.0, 1.0], [0.0, 2.0]), [0.0, 1.0], [0.0, 1.0]),
    ],
    ids=["missing_coords", "shape_mismatch", "different_lat", "different_lon"],
)
def test_align_to_grid_returns_none_for_nonidentical_grids(
    dataset, target_lat, target_lon
):
    assert align_to_grid(dataset, target_lat, target_lon) is None


def test_add_derived_variables_builds_only_requested_complete_wind_speed():
    dataset = xr.Dataset(
        {
            "u10m": (("lat", "lon"), [[3.0, 5.0]]),
            "v10m": (("lat", "lon"), [[4.0, 12.0]]),
            "u850": (("lat", "lon"), [[8.0, 15.0]]),
            "v850": (("lat", "lon"), [[15.0, 8.0]]),
        },
        coords={"lat": [0.0], "lon": [0.0, 1.0]},
    )

    actual = add_derived_variables(dataset, ["ws10m"])

    np.testing.assert_allclose(actual.ws10m, [[5.0, 13.0]])
    assert actual.ws10m.attrs["units"] == "m/s"
    assert "ws850" not in actual


def test_add_derived_variables_builds_ws850_from_850_hpa_components():
    """The requested pressure-level wind speed is sqrt(u850² + v850²)."""
    dataset = xr.Dataset(
        {
            "u850": (("lat", "lon"), [[8.0, 15.0]]),
            "v850": (("lat", "lon"), [[15.0, 8.0]]),
        },
        coords={"lat": [0.0], "lon": [0.0, 1.0]},
    )

    actual = add_derived_variables(dataset, ["ws850"])

    np.testing.assert_allclose(actual.ws850, [[17.0, 17.0]])
    assert actual.ws850.attrs["units"] == "m/s"


def test_add_derived_variables_preserves_existing_and_skips_missing_components():
    existing = xr.Dataset(
        {
            "u10m": (("lat", "lon"), [[3.0]]),
            "v10m": (("lat", "lon"), [[4.0]]),
            "ws10m": (("lat", "lon"), [[99.0]]),
        },
        coords={"lat": [0.0], "lon": [0.0]},
    )
    incomplete = existing.drop_vars(["v10m", "ws10m"])

    assert add_derived_variables(existing, ["ws10m"]).ws10m.item() == 99.0
    assert "ws10m" not in add_derived_variables(incomplete, ["ws10m"])


def test_narrow_batch_selects_one_variable_and_recomputes_its_valid_mask():
    coords = {"lat": [0.0], "lon": [0.0, 1.0]}
    forecast = xr.Dataset(
        {
            "a": (("lat", "lon"), [[1.0, 2.0]]),
            "b": (("lat", "lon"), [[np.nan, 4.0]]),
        },
        coords=coords,
    )
    observation = xr.Dataset(
        {
            "a": (("lat", "lon"), [[1.0, np.nan]]),
            "b": (("lat", "lon"), [[3.0, 4.0]]),
        },
        coords=coords,
    )
    reference = xr.Dataset({"a": (("lat", "lon"), [[0.0, 0.0]])}, coords=coords)
    old_mask = xr.DataArray([[False, False]], dims=("lat", "lon"), coords=coords)

    actual = narrow_batch(
        _batch(forecast, observation, reference=reference, valid_mask=old_mask), "a"
    )

    assert actual is not None
    assert isinstance(actual.forecast, xr.DataArray)
    assert isinstance(actual.observation, xr.DataArray)
    assert isinstance(actual.reference, xr.DataArray)
    np.testing.assert_array_equal(actual.valid_mask, [[True, False]])


def test_narrow_batch_returns_none_when_either_side_lacks_variable():
    forecast = _grid_dataset([0.0], [0.0], name="a")
    observation = _grid_dataset([0.0], [0.0], name="b")

    assert narrow_batch(_batch(forecast, observation), "a") is None
    assert narrow_batch(_batch(forecast, observation), "b") is None


def test_restrict_to_latitude_band_masks_grid_points_with_closed_boundaries():
    field = xr.DataArray(
        np.ones((3, 2)),
        dims=("lat", "lon"),
        coords={"lat": [-30.0, 0.0, 30.0], "lon": [0.0, 1.0]},
    )
    batch = _batch(field, field)

    actual = restrict_to_latitude_band(batch, -30.0, 0.0)

    np.testing.assert_array_equal(
        actual.valid_mask,
        [[True, True], [True, True], [False, False]],
    )
    np.testing.assert_array_equal(actual.forecast, batch.forecast)


def test_restrict_to_latitude_band_uses_station_latitude_coordinate():
    field = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims="station",
        coords={"station": ["a", "b", "c"], "station_lat": ("station", [-20.0, 0.0, 25.0])},
    )

    actual = restrict_to_latitude_band(_batch(field, field), -20.0, 20.0)

    np.testing.assert_array_equal(actual.valid_mask, [True, True, False])


def test_restrict_to_latitude_band_rejects_unsupported_mask_shape():
    field = xr.DataArray(
        np.ones((2, 2)), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]}
    )

    with pytest.raises(AlignmentError, match="只能是格点"):
        restrict_to_latitude_band(_batch(field, field), -20.0, 20.0)


def test_matcher_builds_time_aligned_batches_with_derived_wind_and_common_mask():
    init = np.array(["2025-01-02T00:00:00"], dtype="datetime64[ns]")
    leads = np.array([6, 12])
    forecast_lat = np.array([90.0, 0.0, -90.0])
    forecast_lon = np.array([0.0, 90.0, 180.0, 270.0])
    shape = (1, 2, forecast_lat.size, forecast_lon.size)
    forecast_u = np.empty(shape, dtype="f8")
    forecast_u[:, 0] = 3.0
    forecast_u[:, 1] = 6.0
    forecast_v = np.empty(shape, dtype="f8")
    forecast_v[:, 0] = 4.0
    forecast_v[:, 1] = 8.0
    forecast = xr.Dataset(
        {
            "u10m": (("init_time", "lead_time", "lat", "lon"), forecast_u),
            "v10m": (("init_time", "lead_time", "lat", "lon"), forecast_v),
        },
        coords={
            "init_time": init,
            "lead_time": leads,
            "lat": forecast_lat,
            "lon": forecast_lon,
        },
    )

    observation_lat = forecast_lat[::-1]
    observation_lon = np.array([-180.0, -90.0, 0.0, 90.0])
    valid_times = init[0] + leads.astype("timedelta64[h]")
    observation_u = np.empty((2, 3, 4), dtype="f8")
    observation_u[0] = 3.0
    observation_u[1] = 6.0
    observation_v = np.empty((2, 3, 4), dtype="f8")
    observation_v[0] = 4.0
    observation_v[1] = 8.0
    observation_u[0, 0, 0] = np.nan
    observation = xr.Dataset(
        {
            "u10m": (("valid_time", "lat", "lon"), observation_u),
            "v10m": (("valid_time", "lat", "lon"), observation_v),
        },
        coords={
            "valid_time": valid_times,
            "lat": observation_lat,
            "lon": observation_lon,
        },
    )

    batches = Matcher().match(
        _bundle(forecast, DataKind.GRIDDED_FORECAST, "forecast"),
        _bundle(observation, DataKind.GRIDDED_OBSERVATION, "observation"),
        variables=["u10m", "v10m", "ws10m"],
        sample_by="init_lead",
    )

    assert len(batches) == 2
    for index, batch in enumerate(batches):
        assert set(batch.forecast.data_vars) == {"u10m", "v10m", "ws10m"}
        assert set(batch.observation.data_vars) == {"u10m", "v10m", "ws10m"}
        np.testing.assert_array_equal(batch.forecast.lat, forecast_lat)
        np.testing.assert_array_equal(batch.observation.lat, forecast_lat)
        np.testing.assert_array_equal(batch.observation.lon, forecast_lon)
        np.testing.assert_allclose(batch.forecast.ws10m, 5.0 * (index + 1))
        assert batch.sample_keys[0]["lead_h"] == pytest.approx(float(leads[index]))
        assert batch.alignment["spatial"] == "grids_identical"
        assert batch.alignment["lon_convention"] == "observation_wrapped_to_forecast_circle"
    assert int(batches[0].valid_mask.sum()) == 11
    assert int(batches[1].valid_mask.sum()) == 12


def test_matcher_raises_when_requested_variables_have_no_common_field():
    dataset = _grid_dataset([0.0, 1.0], [0.0, 1.0], name="z500")

    with pytest.raises(AlignmentError, match="No common variables"):
        Matcher().match(
            _bundle(dataset, DataKind.GRIDDED_FORECAST, "forecast"),
            _bundle(dataset, DataKind.GRIDDED_OBSERVATION, "observation"),
            variables=["missing"],
        )


def test_matcher_falls_back_to_linear_interpolation_for_different_grid():
    forecast_lat = np.array([0.0, 2.0])
    forecast_lon = np.array([0.0, 2.0])
    forecast = _grid_dataset(
        forecast_lat,
        forecast_lon,
        forecast_lat[:, None] + forecast_lon[None, :],
    )
    observation = _grid_dataset(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        np.zeros((3, 3)),
    )

    batch = Matcher().match(
        _bundle(forecast, DataKind.GRIDDED_FORECAST, "forecast"),
        _bundle(observation, DataKind.GRIDDED_OBSERVATION, "observation"),
        variables=["z500"],
    )[0]

    expected = np.add.outer([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(batch.forecast.z500, expected)
    assert batch.alignment["spatial"] == "forecast_regridded_to_observation"


def test_matcher_raises_when_forecast_and_observation_times_do_not_overlap():
    forecast = xr.Dataset(
        {
            "z500": (
                ("init_time", "lead_time", "lat", "lon"),
                np.ones((1, 1, 1, 1)),
            )
        },
        coords={
            "init_time": np.array(["2025-01-02T00:00:00"], dtype="datetime64[ns]"),
            "lead_time": [6],
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    observation = xr.Dataset(
        {"z500": (("valid_time", "lat", "lon"), np.ones((1, 1, 1)))},
        coords={
            "valid_time": np.array(["2025-01-03T00:00:00"], dtype="datetime64[ns]"),
            "lat": [0.0],
            "lon": [0.0],
        },
    )

    with pytest.raises(AlignmentError, match="No common valid_time"):
        Matcher().match(
            _bundle(forecast, DataKind.GRIDDED_FORECAST, "forecast"),
            _bundle(observation, DataKind.GRIDDED_OBSERVATION, "observation"),
            variables=["z500"],
        )
