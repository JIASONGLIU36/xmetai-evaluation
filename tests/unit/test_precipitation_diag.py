# -*- coding: utf-8 -*-
"""降水链的算法/协议层：观测窗口累积与观测时间窗推导。

对应台风套件的 ``test_typhoon_diag.py``（几何 + 链式诊断）+ ``test_typhoon_seed.py``
（协议纯函数）这两层，只是降水链的算法重头在**时间**而不是**空间**：

- ``transforms/temporal.py``：``window_sum_at`` 的两种缺测口径、``TimeWindowAccumulator``
  的步长/窗口校验与求和顺序；
- ``pipeline/protocols.py``：``station_observation_span``（协议层与执行计划层共用的
  观测窗口径）。

窗口累积是降水链"静默错"的重灾区：缺一个时次按 0 累计、求和顺序反过来，
分数照样出得来，只是与参考实现差着几个站点。所以这里每个结论都手算，
并把"0 是有效记录"和"求和必须从有效时刻往前"单独钉死。
"""
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.pipeline.protocols import station_observation_span
from xmetai_evaluation.transforms.temporal import TimeWindowAccumulator, window_sum_at

DAY = datetime(2025, 8, 19, 0, 0)


def _hourly(values, start=DAY):
    """逐时观测：第 i 个时次的值是 ``values[i]``（从 ``start`` 起，1 小时间隔）。"""
    times = [start + timedelta(hours=offset) for offset in range(len(values))]
    return xr.Dataset(
        {"precipitation": ("time", np.asarray(values, dtype="f8"), {"units": "mm"})},
        coords={"time": times},
    )


RAMP = np.arange(1, 49, dtype="f8")  # 8/19 00:00..8/20 23:00，值 = 绝对小时序号 + 1


# ---------------------------------------------------------------- 观测窗口累积

def test_window_sum_complete_window_matches_hand_calculation():
    # 8/20 05:00 收尾的 6h 窗口 = 8/20 00..05，值 25..30
    result = window_sum_at(_hourly(RAMP), "precipitation", datetime(2025, 8, 20, 5, 0), 6)
    assert float(result) == pytest.approx(25 + 26 + 27 + 28 + 29 + 30)


def test_window_sum_one_hour_window_is_the_value_itself():
    result = window_sum_at(_hourly(RAMP), "precipitation", datetime(2025, 8, 20, 5, 0), 1)
    assert float(result) == pytest.approx(30.0)


def test_window_sum_zero_is_a_valid_record_not_a_gap():
    """0 是有效记录：全 0 窗口出 0，不是 NaN。"""
    result = window_sum_at(_hourly(np.zeros(48)), "precipitation", datetime(2025, 8, 20, 5, 0), 6)
    assert float(result) == pytest.approx(0.0)
    assert np.isfinite(float(result))


def test_window_sum_missing_timestep_raises_regardless_of_fillna():
    gapped = _hourly(RAMP).isel(time=[i for i in range(RAMP.size) if i != 27])  # 去掉 8/20 03:00
    for require_complete in (True, False):
        with pytest.raises(ValueError, match="观测窗口不完整"):
            window_sum_at(
                gapped,
                "precipitation",
                datetime(2025, 8, 20, 5, 0),
                6,
                require_complete=require_complete,
            )


def test_window_sum_require_complete_masks_nan_value():
    values = RAMP.copy()
    values[27] = np.nan  # 8/20 03:00 缺测（时次在、值缺）
    result = window_sum_at(_hourly(values), "precipitation", datetime(2025, 8, 20, 5, 0), 6)
    assert np.isnan(float(result))


def test_window_sum_fillna_zero_treats_missing_value_as_zero():
    values = RAMP.copy()
    values[27] = np.nan  # 同上，但按参考实现 fillna(0) 口径
    with pytest.warns(UserWarning, match="缺测计 0"):
        result = window_sum_at(
            _hourly(values),
            "precipitation",
            datetime(2025, 8, 20, 5, 0),
            6,
            require_complete=False,
        )
    assert float(result) == pytest.approx(25 + 26 + 27 + 0 + 29 + 30)


def test_window_sum_fillna_zero_reverses_summation_order():
    """参考实现从有效时刻往前累加；正序会把 1.0 丢成 0.0，阈值站点就会翻转。"""
    values = [1.0, 1e16, -1e16, 0.0]
    assert float(np.sum(values[:3])) == 0.0  # 正序：1.0 被 1e16 吞掉

    with pytest.warns(UserWarning, match="缺测计 0"):
        result = window_sum_at(
            _hourly(values),
            "precipitation",
            datetime(2025, 8, 19, 2, 0),
            3,
            require_complete=False,
        )
    assert float(result) == pytest.approx(1.0)  # 反序：(-1e16 + 1e16) + 1.0


def test_window_sum_fillna_zero_drops_samples_with_missing_last_step():
    values = RAMP.copy()
    values[29] = np.nan  # 窗口末端 8/20 05:00 无记录
    with pytest.warns(UserWarning, match="缺测计 0"):
        result = window_sum_at(
            _hourly(values),
            "precipitation",
            datetime(2025, 8, 20, 5, 0),
            6,
            require_complete=False,
        )
    assert np.isnan(float(result))


# ---------------------------------------------------------------- 时效累积

def _lead_data(values, leads, attrs=None):
    return xr.DataArray(
        np.asarray(values, dtype="f8"),
        dims=("lead_time",),
        coords={"lead_time": np.asarray(leads, dtype="f8")},
        attrs=attrs or {"units": "mm"},
    )


def test_time_window_accumulator_slides_and_keeps_end_labels():
    accumulated = TimeWindowAccumulator(24.0).transform(
        _lead_data(np.arange(6), [0, 6, 12, 18, 24, 30])
    )

    # 4 步窗口：第 4 个时效（18h）才凑满，标签取窗口结束时效
    assert accumulated["lead_time"].values.tolist() == [18.0, 24.0, 30.0]
    assert accumulated.values.tolist() == [0 + 1 + 2 + 3, 1 + 2 + 3 + 4, 2 + 3 + 4 + 5]
    assert accumulated.attrs["accumulation_window_hours"] == 24.0
    assert accumulated.attrs["accumulation_method"] == "sliding_window"
    assert accumulated.attrs["units"] == "mm"  # 原有属性保留


def test_time_window_accumulator_masks_incomplete_window():
    values = np.arange(6, dtype="f8")
    values[0] = np.nan  # 只污染第一个窗口（lead 0..3），后面两个窗口仍完整
    accumulated = TimeWindowAccumulator(24.0).transform(_lead_data(values, [0, 6, 12, 18, 24, 30]))
    assert np.isnan(float(accumulated.isel(lead_time=0)))
    assert float(accumulated.isel(lead_time=1)) == pytest.approx(1 + 2 + 3 + 4)
    assert float(accumulated.isel(lead_time=2)) == pytest.approx(2 + 3 + 4 + 5)


def test_time_window_accumulator_rejects_bad_expectations():
    with pytest.raises(ValueError, match="lead_time"):
        TimeWindowAccumulator(24.0).transform(xr.DataArray(np.arange(6), dims=("time",)))
    with pytest.raises(ValueError, match="at least 2 time steps"):
        TimeWindowAccumulator(24.0).transform(_lead_data([1.0], [0.0]))
    with pytest.raises(ValueError, match="must be uniform"):
        TimeWindowAccumulator(24.0).transform(_lead_data(np.arange(4), [0, 6, 12, 20]))
    with pytest.raises(ValueError, match="smaller than step"):
        TimeWindowAccumulator(3.0).transform(_lead_data(np.arange(4), [0, 6, 12, 18]))
    with pytest.raises(ValueError, match="No complete windows"):
        TimeWindowAccumulator(24.0).transform(_lead_data(np.arange(3), [0, 6, 12]))


# ---------------------------------------------------------------- 观测时间窗

def test_station_observation_span_covers_earliest_window_and_latest_lead():
    start, end = station_observation_span(
        [datetime(2025, 8, 20, 0, 0), datetime(2025, 8, 20, 12, 0)],
        offset_hours=8.0,
        window_hours=24,
        lead_max_hours=72,
    )
    assert start == datetime(2025, 8, 19, 9, 0)  # 最早起报 + 8h 北京时 - 23h
    assert end == datetime(2025, 8, 23, 20, 0)  # 最晚起报 + 8h + 72h


def test_station_observation_span_one_hour_window_starts_at_first_valid_time():
    start, end = station_observation_span(
        [datetime(2025, 8, 20, 0, 0)],
        offset_hours=0.0,
        window_hours=1,
        lead_max_hours=6,
    )
    assert start == datetime(2025, 8, 20, 0, 0)
    assert end == datetime(2025, 8, 20, 6, 0)


def test_station_observation_span_uses_init_extremes_only():
    """窗口只由最早/最晚起报决定，中间起报日不影响。"""
    one = station_observation_span([datetime(2025, 8, 20, 0, 0)], 8.0, 24, 72)
    dense = station_observation_span(
        [
            datetime(2025, 8, 20, 0, 0),
            datetime(2025, 8, 20, 6, 0),
            datetime(2025, 8, 20, 12, 0),
        ],
        8.0,
        24,
        72,
    )
    same_extremes = station_observation_span(
        [datetime(2025, 8, 20, 0, 0), datetime(2025, 8, 20, 12, 0)], 8.0, 24, 72
    )
    assert dense == same_extremes
    assert dense != one  # 最晚起报往后推 12h，收尾时刻跟着推 12h
