# -*- coding: utf-8 -*-
"""weather_typhoon_single_fuxi 全链集成（合成数据，默认跑）。

设计：网格 0.25°（中心恰落格点），高斯低压极小值 950 hPa；起报 2025-06-10
00Z；四个时效 6/12/18/24h；实况每 6h 北移 0.5°；预报中心 = 实况 +0.25°N。
预期路径误差 ≈ 0.25° 纬度距（KM_PER_DEG × 0.25），强度为固定差（±5 / −10）。
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from tests import runner
from tests.runner import KM_PER_DEG
from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.metrics.track_error import CURVE_FIELDS

pytest.importorskip("netCDF4")

INIT_UTC = datetime(2025, 6, 10, 0, 0)
INIT_ISO = "2025-06-10T00:00:00"
LEADS = [6.0, 12.0, 18.0, 24.0]
OBS_LAT = [20.5, 21.0, 21.5, 22.0]
OBS_LON = 121.0
FCST_LAT = [lat + 0.25 for lat in OBS_LAT]
EXPECTED_KM = 0.25 * KM_PER_DEG


def _babj_rows(*, with_init_obs=True, drop_lead=None, extra_offgrid=None):
    rows = []
    if with_init_obs:
        rows.append((datetime(2025, 6, 10, 8, 0), 0, OBS_LON, 20.0, 960, 25))
    for lead, lat in zip(LEADS, OBS_LAT):
        if lead == drop_lead:
            continue
        moment = datetime(2025, 6, 10, 8, 0) + timedelta(hours=lead)
        rows.append((moment, 0, OBS_LON, lat, 960, 25))
    rows.extend(extra_offgrid or [])
    rows.sort(key=lambda row: row[0])
    return rows


def _make_dataset(tmp_path, *, extra_init_day=False, babj_rows=None):
    forecast_root = tmp_path / "fuxi_single"
    runner.write_fuxi_single(forecast_root, INIT_UTC, [(lat, OBS_LON) for lat in FCST_LAT])
    if extra_init_day:
        runner.write_fuxi_single(
            forecast_root, datetime(2025, 6, 9, 0, 0), [(lat, OBS_LON) for lat in FCST_LAT]
        )
    babj_dir = tmp_path / "babj"
    runner.write_babj(
        babj_dir / "babj2501.dat", "WUTIP", "2501",
        babj_rows if babj_rows is not None else _babj_rows(),
    )
    return forecast_root, babj_dir


def _run(tmp_path, forecast_root, babj_dir, *, start="20250610", end="20250610",
         execution=None, options=None):
    return runner.run_config(
        "weather_typhoon_single_fuxi",
        forecast_root=forecast_root,
        babj_root=babj_dir,
        output_dir=tmp_path / "out",
        start=start,
        end=end,
        options={"storm_ids": ["2501"], **(options or {})},
        execution=execution,
    )


def test_det_chain_end_to_end(tmp_path):
    forecast_root, babj_dir = _make_dataset(tmp_path)
    out = _run(tmp_path, forecast_root, babj_dir)
    case = runner.read_case(out, "2501", INIT_ISO)
    assert list(case.columns) == list(CURVE_FIELDS)
    assert len(case) == 4
    assert case["track_err_km"].to_numpy() == pytest.approx([EXPECTED_KM] * 4, abs=1e-3)
    # 首时次 at/ct 为空（没有前一个实况）；其后按实况路径段分解
    assert math.isnan(case["at_km"].iloc[0]) and math.isnan(case["ct_km"].iloc[0])
    assert case["at_km"].to_numpy()[1:] == pytest.approx([EXPECTED_KM] * 3, abs=1e-3)
    assert case["ct_km"].to_numpy()[1:] == pytest.approx([0.0] * 3, abs=1e-3)
    assert case["wind_err_ms"].to_numpy() == pytest.approx([5.0] * 4, abs=1e-3)
    assert case["pmin_err_hpa"].to_numpy() == pytest.approx([-10.0] * 4, abs=1e-3)
    assert case["obs_lat"].to_numpy() == pytest.approx(OBS_LAT, abs=1e-6)
    meta = runner.read_meta(out, "2501", INIT_ISO)
    assert meta["seed"] == "init+6h"
    assert meta["init_pos"] == pytest.approx([OBS_LAT[0], OBS_LON])  # 链的种子位置
    assert meta["n_matched"] == 4 and meta["n_leads"] == 4
    assert meta["forecast_type"] == "det" and meta["has_wind"] is True
    scores = runner.read_scores(out)
    assert len(scores) == 5  # 五个 field 各一行
    assert set(scores["status"]) == {"success"}
    assert set(scores["storm"].astype(str)) == {"2501"}


def test_det_seed_gap_and_offgrid_record(tmp_path):
    # 删 14:00（lead6）、塞一条 17:00 非网格记录：种子跳到 20:00（init+12h）
    offgrid = [(datetime(2025, 6, 10, 17, 0), 0, 121.4, 20.4, 961, 24)]
    forecast_root, babj_dir = _make_dataset(
        tmp_path, babj_rows=_babj_rows(drop_lead=6.0, extra_offgrid=offgrid)
    )
    out = _run(tmp_path, forecast_root, babj_dir)
    case = runner.read_case(out, "2501", INIT_ISO)
    meta = runner.read_meta(out, "2501", INIT_ISO)
    assert meta["seed"] == "init+12h"
    assert meta["init_pos"] == pytest.approx([21.0, OBS_LON])  # 20:00 实况
    assert meta["n_matched"] == 3
    assert math.isnan(case["track_err_km"].iloc[0])  # lead6 配不上实况
    assert math.isnan(case["at_km"].iloc[1])         # 首个配对时次 at/ct 空
    assert case["at_km"].iloc[2] == pytest.approx(EXPECTED_KM, abs=1e-3)
    assert not case["valid_bjt"].str.contains("17:00").any()  # 非网格记录进不了曲线


def test_det_pre_genesis_init_day_filtered(tmp_path):
    # 多放一个生成前的起报日（06-09）：应被日期窗剔除，不产场次、不堆失败块
    forecast_root, babj_dir = _make_dataset(tmp_path, extra_init_day=True)
    out = _run(tmp_path, forecast_root, babj_dir, start="20250609", end="20250610")
    cases = sorted(path.name for path in runner.case_dir(out).glob("tc2501_*_meta.json"))
    assert cases == ["tc2501_2025061000_meta.json"]


def test_det_execution_guards(tmp_path):
    forecast_root, babj_dir = _make_dataset(tmp_path)
    # 观测加载角色必须是 slice（resident/window 会把北京时的报文时间轴切没）
    with pytest.raises(ConfigError, match="slice"):
        _run(tmp_path, forecast_root, babj_dir,
             execution={"loads": {"observation": "resident"}})
    # 链式诊断要求整段时效落在同一个工作块里：切了必须报错，不出假数
    with pytest.raises(ConfigError, match="lead_chunk_days"):
        _run(tmp_path, forecast_root, babj_dir, execution={"lead_chunk_days": 1})
