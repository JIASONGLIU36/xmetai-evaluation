"""Default-running synthetic end-to-end test for the ensemble weather field."""

from __future__ import annotations

import pandas as pd
import pytest

from tests import runner_weather_field as runner
from xmetai_evaluation.output.table import SCORE_COLUMNS

LEADS = (6.0, 12.0)


def _value(scores: pd.DataFrame, metric: str, lead: float) -> float:
    """Return the ``(metric, z500, lead)`` global cell.

    ``rmse`` legitimately appears twice — once from the ``rmse`` metric and once
    from ``spread_error``'s ``RMSE`` field — so the lookup tolerates duplicates
    but insists they agree (a real fork would still fail the consistency check).
    """
    rows = scores[
        (scores["metric"] == metric)
        & (scores["variable"] == "z500")
        & (scores["lead_h"] == lead)
        & scores["region"].isna()
    ]
    assert len(rows) >= 1, f"scores.csv 里没有 {metric} @ {lead}h"
    values = rows["value"].to_numpy(dtype=float)
    spread = (values.max() - values.min()) / max(abs(values[0]), 1e-12)
    assert spread <= 1e-4, f"{metric} @ {lead}h 的多行互相矛盾: {values!r}"
    return float(values[0])


def test_weather_ens_chain_end_to_end(tmp_path):
    """The packaged ensemble config must run from native inputs through writers."""
    init_time = pd.Timestamp("2025-01-02T00:00:00").to_pydatetime()
    forecast_root = runner.write_synthetic_fuxi_ens_phys(
        tmp_path / "forecast", init_time, {"member_000": 0.0, "member_001": 4.0}
    )
    stores = runner.write_synthetic_era5_zarr(tmp_path / "era5")
    climatology = runner.write_synthetic_daily_climatology(
        tmp_path / "climatology.nc"
    )

    output = runner.run_synthetic_weather_ens_config(
        forecast_root=forecast_root,
        stores=stores,
        climatology_path=climatology,
        output_dir=tmp_path / "output",
    )

    scores = runner.read_field_scores(output)
    assert list(scores.columns) == SCORE_COLUMNS
    assert set(scores["status"]) == {"success"}
    assert set(scores["variable"]) == {"z500"}
    assert set(scores["protocol_id"]) == {"grid_valid_time"}

    # Metric routing: the ensemble family plus the deterministic family on the
    # ensemble mean must all be present, and nothing beyond z500.
    z500_metrics = set(scores.loc[scores["variable"] == "z500", "metric"])
    assert {
        "rmse",
        "crps",
        "spread",
        "spread_error_ratio",
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

    # Ensemble mean = z500 + 2.  Observation 6h = z500, 12h = z500 + 10, so:
    #   rmse(mean) 6h = 2, 12h = 8;  spread = std({0, 4}) = 2 (ddof=0);
    #   ratio = spread / rmse;  crps = 1 (6h) and 7 (12h), closed form.
    expected_rmse = {6.0: 2.0, 12.0: 8.0}
    expected_crps = {6.0: 1.0, 12.0: 7.0}
    expected_ratio = {6.0: 1.0, 12.0: 0.25}
    for lead in LEADS:
        assert _value(scores, "rmse", lead) == pytest.approx(expected_rmse[lead], abs=1e-5)
        # CRPS uses absolute differences |f - o|, so the float32 rounding of the
        # observation does not cancel out the way it does in RMSE's squared error:
        # member_000 = base collides with obs 6h = float32(base), biasing CRPS by
        # ~5e-4.  rel=1e-3 covers that bias; a real CRPS formula bug moves it by O(1).
        assert _value(scores, "crps", lead) == pytest.approx(expected_crps[lead], rel=1e-3)
        assert _value(scores, "spread", lead) == pytest.approx(2.0, abs=1e-5)
        assert _value(scores, "spread_error_ratio", lead) == pytest.approx(
            expected_ratio[lead], abs=1e-4
        )

    # rmse appears exactly twice per lead (rmse metric + spread_error's RMSE).
    rmse_rows = scores[(scores["metric"] == "rmse") & (scores["variable"] == "z500")]
    assert len(rmse_rows) == 2 * len(LEADS)

    # Deterministic-family parity: the ensemble mean is obs + a constant offset,
    # so activity/spectrum ratios are 1 in closed form (see the single-field test).
    for lead in LEADS:
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

    # Peak in the injected cos(2·lon) harmonic; forecast and observation spectra
    # agree (the offset is removed by de-zonal-meaning).
    by_k = spectrum.set_index("wavenumber")
    pred = by_k["pred_mean"].astype(float)
    obs = by_k["obs_mean"].astype(float)
    assert int(pred.idxmax()) == 2, f"spectrum peak should be wavenumber 2, got {int(pred.idxmax())}"
    assert float(pred.loc[2]) > 0
    assert float(pred.loc[1]) < float(pred.loc[2]) * 1e-3
    assert float(pred.loc[3]) < float(pred.loc[2]) * 1e-3
    assert float((pred - obs).abs().max()) < float(pred.loc[2]) * 1e-3
