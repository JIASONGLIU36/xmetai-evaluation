"""观测窗口累积口径：``station_valid_time`` 的窗口求和。

钉三件事：

1. ``require_complete=False``（reference 口径）：站级缺测按 0 累计，样本去留只看
   **窗口末端时次**——这是 FDP 站点链与 ``ensemble_verifier.py`` 的 ``read_rain_obs``
   的 ``fillna(0)`` 对齐的那条。
2. 整时次不在观测索引里（文件缺失 / 解析失败 / 白名单过滤后为空）：**两种口径都直接
   报错**，与 ``observation_window`` 开关无关。
3. 正常路径不再往 ``warnings`` 里丢东西：那句提示原来是每次调用都刷的，正式跑会按
   「起报 x 时效」喊几万次，把日志淹掉。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xmetai_evaluation.transforms.temporal import window_sum_at

WINDOW_HOURS = 6
WINDOW_END = pd.Timestamp("2026-08-20 06:00")
FULL = pd.date_range("2026-08-20 01:00", periods=WINDOW_HOURS, freq="h")
STATIONS = ["10001", "10002"]


def _dataset(times, values):
    return xr.Dataset(
        {"p": (("time", "station"), np.asarray(values, dtype="f8"))},
        coords={"time": times, "station": STATIONS},
    )


def _ones(rows=WINDOW_HOURS):
    return np.ones((rows, len(STATIONS)))


def _call(dataset, *, require_complete):
    return window_sum_at(
        dataset, "p", WINDOW_END, WINDOW_HOURS, require_complete=require_complete
    )


def test_reference_convention_fills_station_gaps_with_zero():
    """站级缺测按 0 累计：站 10002 只有最后两小时有值，窗口合计 = 2。"""
    values = _ones()
    values[0:4, 1] = np.nan

    assert _call(_dataset(FULL, values), require_complete=False).values.tolist() == [6.0, 2.0]


def test_reference_convention_drops_station_missing_at_window_end():
    """末端该站有值就保留（前面缺按 0）；末端缺测才剔除。"""
    values = _ones()
    values[0, 1] = np.nan
    assert _call(_dataset(FULL, values), require_complete=False).values.tolist() == [6.0, 5.0]

    values[WINDOW_HOURS - 1, 1] = np.nan
    dropped = _call(_dataset(FULL, values), require_complete=False)
    assert np.isfinite(dropped.values[0])
    assert np.isnan(dropped.values[1])


def test_complete_convention_drops_station_with_any_gap():
    """缺省口径：窗口内任一 NaN 就把该站置为缺测。"""
    values = _ones()
    values[0, 1] = np.nan

    result = _call(_dataset(FULL, values), require_complete=True)

    assert np.isfinite(result.values[0])
    assert np.isnan(result.values[1])


@pytest.mark.parametrize("require_complete", [True, False])
def test_missing_whole_hour_raises_in_both_conventions(require_complete):
    """整时次不在索引里 → 直接报错，且报文点明是哪个窗口。"""
    times = FULL.delete(1)

    with pytest.raises(ValueError, match="2026-08-20 06:00:00 的 6h 观测窗口不完整"):
        _call(_dataset(times, _ones(WINDOW_HOURS - 1)), require_complete=require_complete)


def test_aligned_run_emits_no_warning():
    """口径对齐的正常路径不产生任何 warning（含站级缺测的 reference 口径）。"""
    gaps = _ones()
    gaps[0:4, 1] = np.nan

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _call(_dataset(FULL, gaps), require_complete=False)
        _call(_dataset(FULL, _ones()), require_complete=False)
        _call(_dataset(FULL, gaps), require_complete=True)
        with pytest.raises(ValueError):
            _call(_dataset(FULL.delete(1), _ones(WINDOW_HOURS - 1)), require_complete=False)

    assert [str(item.message) for item in caught] == []
