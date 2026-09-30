# -*- coding: utf-8 -*-
"""pipeline/typhoon.py：报文解析 / 几何 / 链式诊断 / 配对（合成数据）。"""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from tests.runner import GRID_LAT, GRID_LON, KM_PER_DEG, gaussian_msl
from xmetai_evaluation.core.errors import DecodeError
from xmetai_evaluation.pipeline.typhoon import (
    TyphoonDataError,
    along_cross_track,
    diagnose_track,
    find_center_by_slp,
    great_circle_km,
    match_errors,
    read_babj_analyses,
)


# ---------------------------------------------------------------- 几何

def test_great_circle_meridian_known_distance():
    # 同经度两点：haversine == R·Δφ，可按公式手算
    for delta in (0.1, 0.25, 1.0):
        got = float(great_circle_km(20.0, 121.0, 20.0 + delta, 121.0))
        assert got == pytest.approx(delta * KM_PER_DEG, rel=1e-12)


def test_great_circle_zero_and_symmetry():
    assert float(great_circle_km(20.0, 121.0, 20.0, 121.0)) == pytest.approx(0.0, abs=1e-9)
    forward = float(great_circle_km(20.0, 121.0, 21.0, 122.0))
    backward = float(great_circle_km(21.0, 122.0, 20.0, 121.0))
    assert forward == pytest.approx(backward, rel=1e-12)


def test_great_circle_vectorized():
    values = great_circle_km(np.array([20.0, 21.0]), np.array([121.0, 122.0]), 20.0, 121.0)
    assert values.shape == (2,)
    assert float(values[0]) == pytest.approx(0.0, abs=1e-9)


def test_along_cross_track_aligned_north_offset():
    # 实况正北移动 0.5°，预报再往北多 0.25°：at=+0.25°、ct=0
    at, ct = along_cross_track(20.0, 121.0, 20.5, 121.0, 20.75, 121.0)
    assert at == pytest.approx(0.25 * KM_PER_DEG, rel=1e-9)
    assert ct == pytest.approx(0.0, abs=1e-9)


def test_along_cross_track_pure_cross_offset():
    # 实况段退化（前后同一点），预报正东 0.25°：ct == 预报到锚点的距离、at≈0
    at, ct = along_cross_track(20.0, 121.0, 20.0, 121.0, 20.0, 121.25)
    assert at == pytest.approx(0.0, abs=1e-9)
    expected = float(great_circle_km(20.0, 121.0, 20.0, 121.25))
    assert ct == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------- 报文解析

def test_read_babj_analyses_only_tau_zero(tmp_path, make_babj):
    path = tmp_path / "babj2501.dat"
    t0 = datetime(2025, 6, 10, 8, 0)
    make_babj(
        path,
        "WUTIP",
        "2501",
        [
            (t0, 0, 121.5, 20.3, 960, 25),                        # 分析实况
            (t0 + timedelta(hours=12), 12, 122.0, 20.8, 955, 28),  # 主观预报行：应被忽略
        ],
    )
    tcname, tcid, analyses = read_babj_analyses(path)
    assert (tcname, tcid) == ("WUTIP", "2501")
    assert list(analyses) == [t0]
    # 列序是 lon 在前、lat 在后（第 6、7 列）
    assert analyses[t0] == {"lon": 121.5, "lat": 20.3, "pmin_hpa": 960.0, "vmax_ms": 25.0}


def test_read_babj_analyses_requires_analysis_rows(tmp_path, make_babj):
    path = tmp_path / "babj9901.dat"
    make_babj(path, "SYNTH", "9901", [(datetime(2025, 6, 10, 8, 0), 12, 121.0, 20.0, 960, 25)])
    with pytest.raises(TyphoonDataError):
        read_babj_analyses(path)


def test_read_babj_analyses_missing_file(tmp_path):
    with pytest.raises(DecodeError):
        read_babj_analyses(tmp_path / "nope.dat")


# ---------------------------------------------------------------- 中心诊断

def test_find_center_exact_grid_minimum():
    msl = gaussian_msl(20.0, 121.0)
    lat, lon, pmin = find_center_by_slp(msl, GRID_LAT, GRID_LON, 20.5, 121.5, 3.0, 3.0)
    assert (lat, lon) == pytest.approx((20.0, 121.0), abs=1e-12)
    assert pmin == pytest.approx(95000.0, abs=1e-9)


def test_find_center_box_out_of_grid_raises():
    grid_lat = np.array([30.0, 31.0])
    grid_lon = np.array([0.0, 1.0])
    msl = gaussian_msl(20.0, 121.0, lat=grid_lat, lon=grid_lon)
    with pytest.raises(TyphoonDataError):
        find_center_by_slp(msl, grid_lat, grid_lon, 20.0, 121.0, 3.0, 3.0)


def test_find_center_across_lon_wrap():
    grid_lat = np.array([19.0, 20.0, 21.0])
    grid_lon = np.array([359.5, 0.0, 0.5])
    msl = np.array(
        [
            [90000.0, 91000.0, 92000.0],
            [91000.0, 89000.0, 93000.0],
            [92000.0, 93000.0, 94000.0],
        ]
    )
    lat, lon, pmin = find_center_by_slp(msl, grid_lat, grid_lon, 20.0, 359.9, 1.0, 1.0)
    assert (lat, lon, pmin) == pytest.approx((20.0, 0.0, 89000.0), abs=1e-9)


def test_diagnose_track_is_chained():
    # lead6 场中心 A 离起点 3°；lead12 场中心 B 离 A 2.5°、离起点 5.5°。
    # 链式：lead12 的搜索框以诊断出的 A 为中心 → 找到 B；
    # 若错误地以起点为中心，B 在 ±3° 之外，找不到。
    field = np.stack([gaussian_msl(23.0, 121.0), gaussian_msl(25.5, 121.0)])
    out = diagnose_track(
        field, GRID_LAT, GRID_LON, 20.0, 121.0, [6.0, 12.0],
        center_half=3.0, early_half=4.0, early_hours=12.0,
    )
    assert out["lat"] == pytest.approx([23.0, 25.5], abs=1e-9)
    assert out["lon"] == pytest.approx([121.0, 121.0], abs=1e-9)
    assert out["pmin_hpa"] == pytest.approx([950.0, 950.0], abs=1e-9)
    assert all(math.isnan(value) for value in out["vmax"])  # 无风场


def test_diagnose_track_early_box_and_wind():
    # lead6（< early_hours）用大框（±4°）：中心离起点 3.5° 也能找到。
    field = gaussian_msl(23.5, 121.0)[None, ...]
    wind = np.full_like(field, 30.0)
    out = diagnose_track(field, GRID_LAT, GRID_LON, 20.0, 121.0, [6.0], wind=wind)
    assert out["lat"] == pytest.approx([23.5], abs=1e-9)
    assert out["vmax"] == pytest.approx([30.0], abs=1e-9)
    assert out["pmin_hpa"] == pytest.approx([950.0], abs=1e-9)


# ---------------------------------------------------------------- 配对

def _track(lat, lon, pmin, vmax):
    return {"lat": lat, "lon": lon, "pmin_hpa": pmin, "vmax": vmax}


def test_match_errors_first_match_and_gap():
    init = datetime(2025, 6, 10, 0, 0)
    babj = {
        datetime(2025, 6, 10, 14, 0): {"lon": 121.0, "lat": 20.0, "pmin_hpa": 960.0, "vmax_ms": 25.0},
        # lead12（20:00）故意缺
        datetime(2025, 6, 11, 2, 0): {"lon": 121.0, "lat": 21.0, "pmin_hpa": 958.0, "vmax_ms": 28.0},
    }
    track = _track(
        [20.25, 20.75, 21.25],
        [121.0, 121.0, 121.0],
        [950.0, 949.0, 948.0],
        [30.0, float("nan"), 30.0],
    )
    rows = match_errors(track, [6.0, 12.0, 18.0], init, babj, tz_shift=8.0)
    r6, r12, r18 = rows
    # 首个配对时次的 at/ct 为空（没有前一个实况就无从定向）
    assert math.isnan(r6["at_km"]) and math.isnan(r6["ct_km"])
    assert r6["valid_bjt"] == "2025-06-10 14:00:00"
    assert r6["track_err_km"] == pytest.approx(0.25 * KM_PER_DEG, rel=1e-9)
    assert r6["wind_err_ms"] == pytest.approx(5.0)
    assert r6["pmin_err_hpa"] == pytest.approx(-10.0)
    # 缺口整行 NaN，且不携带陈旧方向（prev 只在配对成功时更新）
    assert math.isnan(r12["track_err_km"]) and math.isnan(r12["at_km"])
    # lead18 的 prev = 最后一条配对成功的 lead6 实况 → 仍按正北方向分解
    assert r18["track_err_km"] == pytest.approx(0.25 * KM_PER_DEG, rel=1e-9)
    assert r18["at_km"] == pytest.approx(0.25 * KM_PER_DEG, rel=1e-9)
    assert r18["ct_km"] == pytest.approx(0.0, abs=1e-9)
    assert r18["wind_err_ms"] == pytest.approx(2.0)


def test_match_errors_wind_nan_when_forecast_wind_missing():
    init = datetime(2025, 6, 10, 0, 0)
    babj = {
        datetime(2025, 6, 10, 14, 0): {"lon": 121.0, "lat": 20.0, "pmin_hpa": 960.0, "vmax_ms": 25.0}
    }
    track = _track([20.25], [121.0], [950.0], [float("nan")])
    row = match_errors(track, [6.0], init, babj)[0]
    assert math.isnan(row["wind_err_ms"])
    assert row["pmin_err_hpa"] == pytest.approx(-10.0)
