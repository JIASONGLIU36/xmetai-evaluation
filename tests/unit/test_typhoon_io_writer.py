# -*- coding: utf-8 -*-
"""io/babj_reader.py 与 typhoon_cases writer 的单元测试（合成数据）。"""
import json
import math
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from xmetai_evaluation.components import _seed_label, _writer_typhoon_cases
from xmetai_evaluation.core.contracts import DataRequest
from xmetai_evaluation.core.errors import DiscoveryError
from xmetai_evaluation.io.babj_reader import BabjCatalog, BabjReader
from xmetai_evaluation.metrics.track_error import CURVE_FIELDS, ENS_EXTRA_FIELDS


# ---------------------------------------------------------------- BABJ reader

def _rows(lat0=20.0, lon=121.0, count=3):
    t0 = datetime(2025, 6, 10, 8, 0)
    return [(t0 + timedelta(hours=6 * k), 0, lon, lat0 + 0.1 * k, 960, 25) for k in range(count)]


def test_babj_catalog_discovers_directory_and_file(tmp_path, make_babj):
    directory = tmp_path / "babj"
    make_babj(directory / "babj2501.dat", "WUTIP", "2501", _rows())
    make_babj(directory / "babj2502.dat", "MUN", "2502", _rows())
    request = DataRequest(source_id="babj", variables=["storm"])
    index = BabjCatalog(directory).discover(request)
    assert sorted(path.name for path in index.available) == ["babj2501.dat", "babj2502.dat"]
    assert index.ambiguous == []
    # 直接指到一个报文文件也可以
    single = BabjCatalog(directory / "babj2501.dat").discover(request)
    assert len(single.available) == 1


def test_babj_catalog_errors(tmp_path):
    request = DataRequest(source_id="babj", variables=["storm"])
    with pytest.raises(DiscoveryError):
        BabjCatalog(tmp_path / "missing").discover(request)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DiscoveryError):
        BabjCatalog(empty).discover(request)


def test_babj_reader_union_times_and_nan_fill(tmp_path, make_babj):
    directory = tmp_path / "babj"
    t0 = datetime(2025, 6, 10, 8, 0)
    make_babj(
        directory / "babj2501.dat", "杜苏芮", "2501",
        [
            (t0, 0, 121.0, 20.0, 960, 25),
            (t0 + timedelta(hours=6), 0, 121.1, 20.1, 958, 27),
        ],
    )
    make_babj(
        directory / "babj2502.dat", "MUN", "2502",
        [(t0 + timedelta(hours=12), 0, 125.0, 15.0, 970, 18)],
    )
    request = DataRequest(source_id="babj", variables=["storm"])
    bundle = BabjReader().read(request, BabjCatalog(directory).discover(request))
    payload = bundle.payload
    assert list(payload["storm"].values) == ["2501", "2502"]
    assert list(payload["tcname"].values) == ["杜苏芮", "MUN"]  # GBK 中文名
    assert payload.sizes["time"] == 3
    lat = payload["lat"].values
    assert np.isfinite(lat[0, 0]) and np.isfinite(lat[1, 0])
    assert math.isnan(lat[0, 1])  # 2502 在 08:00 没有实况 → 补 NaN
    assert lat[2, 1] == pytest.approx(15.0)
    assert payload["lat"].attrs["units"] == "degrees_north"
    assert bundle.semantic.timezone == "Asia/Shanghai"
    assert len(bundle.provenance.input_files) == 2


def test_babj_catalog_file_time(tmp_path, make_babj):
    directory = tmp_path / "babj"
    t0 = datetime(2025, 6, 10, 8, 0)
    path = directory / "babj2501.dat"
    make_babj(
        path, "WUTIP", "2501",
        [(t0 + timedelta(hours=12), 0, 121.0, 20.0, 960, 25), (t0, 0, 121.1, 20.1, 958, 27)],
    )
    assert BabjCatalog(directory).file_time(path) == t0


# ---------------------------------------------------------------- typhoon_cases writer

def _fake_curve(*, ens=False, seed_offset=6.0):
    curve = {
        "kind": "typhoon_track",
        "storm": "2501",
        "tcname": "WUTIP",
        "init_utc": "2025-06-10T00:00:00",
        "tz_shift": 8.0,
        "init_lat": 15.8,
        "init_lon": 114.5,
        "seed_offset_h": seed_offset,
        "search": {"center_half_deg": 3.0},
        "lead_h": [6.0, 12.0],
        "valid_bjt": ["2025-06-10 14:00:00", "2025-06-10 20:00:00"],
        "fcst_lat": [15.5, 15.75],
        "fcst_lon": [114.75, 114.5],
        "fcst_pmin_hpa": [997.84, 997.075],
        "fcst_vmax_ms": [float("nan"), float("nan")],
        "obs_lat": [15.8, 15.8],
        "obs_lon": [114.5, 114.2],
        "obs_pmin_hpa": [1000.0, 999.0],
        "obs_vmax_ms": [15.0, 15.0],
        "track_err_km": [42.7706, 32.58],
        "at_km": [float("nan"), -32.0981],
        "ct_km": [float("nan"), -5.55975],
        "wind_err_ms": [float("nan"), float("nan")],
        "pmin_err_hpa": [-2.16039, -1.92453],
        "n_leads": 2,
        "n_matched": 2,
    }
    if ens:
        curve.update(
            {
                "forecast_type": "ens",
                "members": ["0", "1"],
                "n_members": [2, 2],
                "n_valid_members": [2, 1],
                "track_err_km_a": [43.0, 33.0],
                "at_km_a": [float("nan"), -31.0],
                "ct_km_a": [float("nan"), -5.0],
            }
        )
    return curve


def test_typhoon_cases_writer_det(tmp_path):
    tables = SimpleNamespace(curves=[{"curve": _fake_curve()}])
    _writer_typhoon_cases(tables, None, tmp_path)
    case = tmp_path / "typhoon" / "tc2501_2025061000.csv"
    raw = case.read_bytes()
    assert b"\r" not in raw  # 行尾必须是 \n（逐字节对拍的坑）
    lines = raw.decode("utf-8").splitlines()
    assert lines[0] == ",".join(CURVE_FIELDS)
    assert len(lines) == 3
    first = lines[1].split(",")
    assert first[5] == "" and first[11] == "" and first[13] == ""  # vmax / at / wind 空
    assert first[10] == "42.7706"  # %.6g
    meta = json.loads((tmp_path / "typhoon" / "tc2501_2025061000_meta.json").read_text("utf-8"))
    assert meta["seed"] == "init+6h"
    assert meta["forecast_type"] == "det"
    assert meta["init_pos"] == [15.8, 114.5]  # 链的种子位置
    assert meta["n_leads"] == 2 and meta["n_matched"] == 2
    assert meta["has_wind"] is False
    combined = (tmp_path / "typhoon" / "typhoon.csv").read_text("utf-8").splitlines()
    assert combined[0] == "tcid,tcname,init_utc," + ",".join(CURVE_FIELDS)
    assert combined[1].startswith("2501,WUTIP,2025-06-10 00:00:00,")


def test_typhoon_cases_writer_ens(tmp_path):
    tables = SimpleNamespace(curves=[{"curve": _fake_curve(ens=True)}])
    _writer_typhoon_cases(tables, None, tmp_path)
    lines = (tmp_path / "typhoon" / "tc2501_2025061000.csv").read_text("utf-8").splitlines()
    assert lines[0] == ",".join(CURVE_FIELDS + ENS_EXTRA_FIELDS)
    assert len(lines[1].split(",")) == len(CURVE_FIELDS) + len(ENS_EXTRA_FIELDS)
    meta = json.loads((tmp_path / "typhoon" / "tc2501_2025061000_meta.json").read_text("utf-8"))
    assert meta["forecast_type"] == "ens"
    assert meta["n_members"] == 2 and meta["members"] == ["0", "1"]
    assert set(meta["aggregation"]) == {"A", "B"}
    # init_pos = 链的种子位置（当前口径；v1 ens 归档写 ob0 的差异已知、暂不追）
    assert meta["init_pos"] == [15.8, 114.5]


def test_seed_label():
    assert _seed_label(6.0) == "init+6h"
    assert _seed_label(12.0) == "init+12h"
    assert _seed_label(0.0) == "init"
    assert _seed_label(None) == "init"
