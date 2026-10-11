"""Default-running synthetic end-to-end test for ensemble station precipitation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from tests import runner_precipitation as runner

#: 2×2 预报网格；4 个站点摆在格点上，双线性插值精确取网格值。
LAT = np.array([10.0, 20.0])
LON = np.array([100.0, 110.0])

#: 起报 2025-01-01 00:00（UTC），offset 8（北京时）。24h TS 段的窗口末端
#: = 01-02 08:00，24h 实况窗 = [01-01 09:00, 01-02 08:00]；6h 概率段四个时效的
#: 验证时刻 = 01-01 14/20、01-02 02/08，对应 ref 文件 010114/010120/010202/010208。
INIT = datetime(2025, 1, 1, 0, 0)
OBS_START = datetime(2025, 1, 1, 9, 0)

#: 4 个站点：(站号, 经度, 纬度)；1/3 是事件站（湿）、2/4 是非事件站（干）。
STATIONS = [
    (10001, 100.0, 10.0),
    (10002, 100.0, 20.0),
    (10003, 110.0, 10.0),
    (10004, 110.0, 20.0),
]


def _write_obs(root):
    """24 个逐时站点文件；事件站每小时 1mm、非事件站 0（6h=6/0、24h=24/0）。"""
    for hour in range(24):
        moment = OBS_START + timedelta(hours=hour)
        rows = [
            (sid, lon, lat, 50.0, 1.0 if sid in (10001, 10003) else 0.0)
            for sid, lon, lat in STATIONS
        ]
        runner.write_diamond(runner.station_file(root, moment), moment, rows)
    return root


def _write_ref(root):
    """四个时次的 ref 文件；p≥0.1 恒 0.25（其余阈值列不用）。"""
    for when in (
        datetime(2025, 1, 1, 14, 0),
        datetime(2025, 1, 1, 20, 0),
        datetime(2025, 1, 2, 2, 0),
        datetime(2025, 1, 2, 8, 0),
    ):
        runner.write_ref_probability(
            root, when, [(sid, 0.25, 0.0, 0.0, 0.0) for sid, _, _ in STATIONS]
        )
    return root


def _value(scores, metric: str, threshold: float) -> float:
    rows = scores[
        (scores["metric"] == metric)
        & (scores["threshold"] == threshold)
        & scores["region"].isna()
    ]
    assert len(rows) == 1, f"scores.csv 里 {metric} @ ≥{threshold} 应该正好一行，实际 {len(rows)} 行"
    return float(rows["value"].iloc[0])


def test_weather_ts_ens_chain_end_to_end(tmp_path):
    """The packaged ensemble precip config must run from native inputs through writers."""
    # 2 成员：member_000 在站 1/3/4 湿、member_001 只在站 1 湿。
    member_values = {
        "member_000": np.array([[5.0, 5.0], [0.0, 5.0]]),
        "member_001": np.array([[5.0, 0.0], [0.0, 0.0]]),
    }
    forecast_root = runner.write_fuxi_ens_tp(
        tmp_path / "forecast", INIT, lat=LAT, lon=LON, member_values=member_values
    )
    obs_root = _write_obs(tmp_path / "obs")
    ref_root = _write_ref(tmp_path / "ref")

    output = runner.run_synthetic_ts_ens_config(
        forecast_root=forecast_root,
        obs_root=obs_root,
        ref_root=ref_root,
        output_dir=tmp_path / "output",
    )

    scores = runner.read_scores(output)
    assert set(scores["status"]) == {"success"}
    assert set(scores["variable"]) == {"precipitation"}
    assert set(scores["sample_unit"]) == {"station"}

    # ── 概率段（6h，阈值 ≥0.1）：四个时效都评，AROC/BS/BSS 手算值 ─────────
    # 直方图：p=1.0 命中 1、p=0.0 正确否定 1、p=0.5 命中/空报各 1。
    #   AROC = 0.875；BS = 0.125；BS_ref = mean((0.25-o)²) = 0.3125；BSS = 0.6。
    prob = scores[scores["metric"].isin(["aroc", "bs", "bss"])]
    assert set(prob["lead_h"]) == {6.0, 12.0, 18.0, 24.0}
    for metric, expected in (("aroc", 0.875), ("bs", 0.125), ("bss", 0.6)):
        vals = scores[
            (scores["metric"] == metric) & (scores["threshold"] == 0.1)
        ]["value"].astype(float)
        assert len(vals) == 4, f"{metric} 应该有 4 个时效各一行"
        for value in vals:
            assert value == pytest.approx(expected, abs=1e-6)

    # ── TS 段（24h，阈值 ≥10）：集合平均 24h = 20/0/10/10，实况 = 24/0/24/0 ──
    assert _value(scores, "ts", 10.0) == pytest.approx(2.0 / 3.0)
    assert _value(scores, "pod", 10.0) == pytest.approx(1.0)
    assert _value(scores, "far", 10.0) == pytest.approx(1.0 / 3.0)
    assert _value(scores, "frequency_bias", 10.0) == pytest.approx(1.5)

    # ── 宽表对拍 ────────────────────────────────────────────────────────────
    # probability_wide 每时效一行（4 行，数值相同）。
    wide = runner.read_probability_wide(output)
    prob_rows = wide[wide["threshold_mm"] == 0.1]
    assert len(prob_rows) == 4
    assert set(prob_rows["lead_h"]) == {6.0, 12.0, 18.0, 24.0}
    for column, expected in (("AROC", 0.875), ("BS_ref", 0.3125), ("base_rate", 0.5)):
        assert np.allclose(prob_rows[column].to_numpy(dtype=float), expected, atol=1e-6)
    assert (prob_rows["n_points"].to_numpy() == 4).all()

    cat = runner.read_categorical_wide(output)
    cat_row = cat[cat["threshold_mm"] == 10.0]
    assert len(cat_row) == 1
    assert int(cat_row["hits"].iloc[0]) == 2
    assert int(cat_row["false_alarms"].iloc[0]) == 1
    assert int(cat_row["misses"].iloc[0]) == 0
