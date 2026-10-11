"""Default-running synthetic end-to-end test for deterministic station precipitation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from tests import runner_precipitation as runner

#: 2×2 预报网格；站点摆在格点上，双线性插值精确取网格值。
LAT = np.array([10.0, 20.0])
LON = np.array([100.0, 110.0])

#: 起报 2025-01-01 00:00（UTC，offset 0）；时效 24 的窗口末端 = 01-02 00:00，
#: 24h 窗口 = [01-01 01:00, 01-02 00:00]。
INIT = datetime(2025, 1, 1, 0, 0)
WINDOW_START = datetime(2025, 1, 1, 1, 0)


def _write_obs(root):
    """写 24 个逐时站点文件；每个站的 24h 总量压在最后一个时次。"""
    for hour in range(24):
        moment = WINDOW_START + timedelta(hours=hour)
        put = hour == 23
        rows = [
            # (站号, 经度, 纬度, 海拔, 降水)；A/C 在湿格点、B 在干格点。
            (10001, 100.0, 10.0, 50.0, 15.0 if put else 0.0),
            (10002, 100.0, 20.0, 50.0, 5.0 if put else 0.0),
            (10003, 110.0, 10.0, 50.0, 3.0 if put else 0.0),
        ]
        runner.write_diamond(runner.station_file(root, moment), moment, rows)
    return root


def _value(scores, metric: str, threshold: float) -> float:
    rows = scores[
        (scores["metric"] == metric)
        & (scores["threshold"] == threshold)
        & scores["region"].isna()
    ]
    assert len(rows) == 1, f"scores.csv 里 {metric} @ ≥{threshold} 应该正好一行，实际 {len(rows)} 行"
    return float(rows["value"].iloc[0])


def test_weather_ts_chain_end_to_end(tmp_path):
    """The packaged precip config must run from native inputs through writers."""
    # 湿格点（lat=10）每 6h 5mm → 24h 20mm；干格点（lat=20）→ 0。
    values = np.array([[5.0, 5.0], [0.0, 0.0]])
    forecast_root = runner.write_fuxi_tp(
        tmp_path / "forecast", INIT, lat=LAT, lon=LON, values=values
    )
    obs_root = _write_obs(tmp_path / "obs")

    output = runner.run_synthetic_ts_config(
        forecast_root=forecast_root, obs_root=obs_root, output_dir=tmp_path / "output"
    )

    scores = runner.read_scores(output)
    assert set(scores["metric"]) == {
        "ts", "pod", "far", "miss_rate", "frequency_bias",
    }
    assert set(scores["status"]) == {"success"}
    assert set(scores["variable"]) == {"precipitation"}
    assert set(scores["sample_unit"]) == {"station"}
    assert set(scores["lead_h"]) == {24.0}

    # 阈值 ≥10：A 命中、C 空报、B 正确否定 → hits=1 fa=1 miss=0
    assert _value(scores, "ts", 10.0) == pytest.approx(0.5)
    assert _value(scores, "pod", 10.0) == pytest.approx(1.0)
    assert _value(scores, "far", 10.0) == pytest.approx(0.5)
    assert _value(scores, "frequency_bias", 10.0) == pytest.approx(2.0)
    assert _value(scores, "miss_rate", 10.0) == pytest.approx(0.0)

    # 阈值 ≥0.1：A/C 命中、B 漏报 → hits=2 miss=1 fa=0
    assert _value(scores, "ts", 0.1) == pytest.approx(2.0 / 3.0)
    assert _value(scores, "pod", 0.1) == pytest.approx(2.0 / 3.0)
    assert _value(scores, "far", 0.1) == pytest.approx(0.0)
    assert _value(scores, "frequency_bias", 0.1) == pytest.approx(2.0 / 3.0)

    # 整数列联表对拍（宽表）
    wide = runner.read_categorical_wide(output)
    by_thr = wide.set_index("threshold_mm")
    assert int(by_thr.loc[10.0, "hits"]) == 1
    assert int(by_thr.loc[10.0, "false_alarms"]) == 1
    assert int(by_thr.loc[10.0, "misses"]) == 0
    assert int(by_thr.loc[0.1, "hits"]) == 2
    assert int(by_thr.loc[0.1, "false_alarms"]) == 0
    assert int(by_thr.loc[0.1, "misses"]) == 1
