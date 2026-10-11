"""Default-running synthetic end-to-end test for deterministic weather fields."""

from __future__ import annotations

import pandas as pd
import pytest

from tests import runner_weather_field as runner
from xmetai_evaluation.output.table import SCORE_COLUMNS


def _value(scores: pd.DataFrame, metric: str, lead: float) -> float:
    """Return the ``(metric, z500, lead)`` cell; ``region`` must be the global row."""
    rows = scores[
        (scores["metric"] == metric)
        & (scores["variable"] == "z500")
        & (scores["lead_h"] == lead)
        & scores["region"].isna()
    ]
    assert len(rows) == 1, f"expected exactly one {metric} @ {lead}h row, got {len(rows)}"
    return float(rows["value"].iloc[0])


def test_weather_field_chain_end_to_end(tmp_path):
    """The packaged config must run from native inputs through output writers."""
    init_time = pd.Timestamp("2025-01-02T00:00:00").to_pydatetime()
    forecast_root = runner.write_synthetic_fuxi_phys(
        tmp_path / "forecast", init_time
    )
    stores = runner.write_synthetic_era5_zarr(tmp_path / "era5")
    climatology = runner.write_synthetic_daily_climatology(
        tmp_path / "climatology.nc"
    )

    output = runner.run_synthetic_weather_field_config(
        forecast_root=forecast_root,
        stores=stores,
        climatology_path=climatology,
        output_dir=tmp_path / "output",
    )

    scores = runner.read_field_scores(output)
    assert list(scores.columns) == SCORE_COLUMNS
    assert set(scores["status"]) == {"success"}
    assert set(scores["variable"]) == {"z500", "t2m"}
    assert set(scores["protocol_id"]) == {"grid_valid_time"}

    # Metric routing comes from the real weather_field_scores template plus the
    # test's metric_options: the surface field must not leak into heavy metrics.
    assert set(scores.loc[scores["variable"] == "t2m", "metric"]) == {"rmse"}
    z500_metrics = set(scores.loc[scores["variable"] == "z500", "metric"])
    assert {
        "rmse",
        "acc",
        "activity_ratio",
        "activity_forecast",
        "activity_observation",
        "activity_bias",
        "spectrum_power_ratio",
        "spherical_band_power_forecast",
        "spherical_band_power_observation",
        "spherical_band_power_ratio",
    } <= z500_metrics

    expected_rmse = {
        ("z500", 6.0): (2.0, "m^2/s^2"),
        ("z500", 12.0): (6.0, "m^2/s^2"),  # obs 12h = z500 + 10, fc 12h = +4
        ("t2m", 6.0): (1.0, "K"),
        ("t2m", 12.0): (8.0, "K"),          # obs 12h = t2m + 10, fc 12h = +2
    }
    rmse = scores[scores["metric"] == "rmse"]
    assert len(rmse) == len(expected_rmse)
    for row in rmse.itertuples(index=False):
        value, unit = expected_rmse[(row.variable, float(row.lead_h))]
        assert float(row.value) == pytest.approx(value, abs=1e-5)
        assert row.unit == unit

    # Forecast = observation + a constant offset. The offset cancels in the
    # anomaly (vs the zero climatology) and in the de-zonal-meaning, so the
    # activity ratio and spectrum power ratio are 1 in closed form. The synthetic
    # observation store is float32 while the forecast is float64, so rounding
    # noise moves these ratios off 1: ~2e-6 for activity, ~4e-5 for spectrum.
    for lead in (6.0, 12.0):
        assert _value(scores, "activity_ratio", lead) == pytest.approx(1.0, rel=1e-4)
        assert _value(scores, "spectrum_power_ratio", lead) == pytest.approx(1.0, rel=1e-3)

    spherical = scores[
        scores["metric"].str.startswith("spherical_band_power_")
    ]
    assert set(spherical["group"]) == {"1_1", "2_3"}

    spectrum = runner.read_field_spectrum(output, "z500")
    assert list(spectrum.columns) == [
        "wavenumber",
        "wavelength_km",
        "pred_mean",
        "obs_mean",
    ]
    assert spectrum["wavenumber"].tolist() == [1, 2, 3]

    # The two spectra must agree (the offset is removed by de-zonal-meaning) and
    # all energy must sit in the injected cos(2·lon) harmonic: wavenumber 2 is the
    # peak, wavenumbers 1 and 3 are ~zero.
    by_k = spectrum.set_index("wavenumber")
    pred = by_k["pred_mean"].astype(float)
    obs = by_k["obs_mean"].astype(float)
    assert int(pred.idxmax()) == 2, f"spectrum peak should be wavenumber 2, got {int(pred.idxmax())}"
    assert float(pred.loc[2]) > 0
    assert float(pred.loc[1]) < float(pred.loc[2]) * 1e-3
    assert float(pred.loc[3]) < float(pred.loc[2]) * 1e-3
    assert float((pred - obs).abs().max()) < float(pred.loc[2]) * 1e-3

    by_init = pd.read_csv(output / "diagnostics" / "spectrum_by_init.csv")
    assert list(by_init.columns) == [
        "variable",
        "init_time",
        "wavenumber",
        "wavelength_km",
        "pred",
        "obs",
    ]
    assert set(by_init["variable"]) == {"z500"}
    assert by_init["wavenumber"].tolist() == [1, 2, 3]
