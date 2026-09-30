# -*- coding: utf-8 -*-
"""weather_typhoon_ens_fuxi 全链集成（合成数据，默认跑）。

设计：两个成员沿正北偏移 +0.25° / −0.75°（不对称，专门让方案 A≠B）：
- 集合平均位置在 obs−0.25° → B 口径误差 = 0.25°；
- 成员误差再平均 → A 口径误差 = (0.25+0.75)/2 = 0.5°；
- 首时次 at/ct 用起报时刻实况定向（有 ob0 非空、无 ob0 留空）。
"""
import math
from datetime import datetime, timedelta

import pytest

from tests import runner
from tests.runner import KM_PER_DEG
from xmetai_evaluation.metrics.track_error import CURVE_FIELDS, ENS_EXTRA_FIELDS

pytest.importorskip("netCDF4")

INIT_UTC = datetime(2025, 6, 10, 0, 0)
INIT_ISO = "2025-06-10T00:00:00"
LEADS = [6.0, 12.0, 18.0, 24.0]
OBS_LAT = [20.5, 21.0, 21.5, 22.0]
OBS_LON = 121.0
INIT_OBS = (20.0, 121.0)
MEMBER_OFFSETS = {"member_000": 0.25, "member_001": -0.75}
EXPECTED_B_KM = 0.25 * KM_PER_DEG
EXPECTED_A_KM = 0.5 * KM_PER_DEG
COLUMNS = list(CURVE_FIELDS) + list(ENS_EXTRA_FIELDS)


def _make_dataset(tmp_path, *, with_init_obs=True, corrupt_member=None):
    forecast_root = tmp_path / "fuxi_ens"
    member_centers = {
        member: [(lat + offset, OBS_LON) for lat in OBS_LAT]
        for member, offset in MEMBER_OFFSETS.items()
    }
    runner.write_fuxi_ens(forecast_root, INIT_UTC, member_centers)
    if corrupt_member:
        (forecast_root / "20250610" / corrupt_member / "001.nc").write_bytes(b"corrupted\n")
    babj_dir = tmp_path / "babj"
    rows = []
    if with_init_obs:
        rows.append((datetime(2025, 6, 10, 8, 0), 0, OBS_LON, INIT_OBS[0], 960, 25))
    for lead, lat in zip(LEADS, OBS_LAT):
        rows.append((datetime(2025, 6, 10, 8, 0) + timedelta(hours=lead), 0, OBS_LON, lat, 960, 25))
    runner.write_babj(babj_dir / "babj2501.dat", "WUTIP", "2501", rows)
    return forecast_root, babj_dir


def _run(tmp_path, forecast_root, babj_dir):
    return runner.run_config(
        "weather_typhoon_ens_fuxi",
        forecast_root=forecast_root,
        babj_root=babj_dir,
        output_dir=tmp_path / "out",
        start="20250610",
        end="20250610",
        options={"storm_ids": ["2501"]},
    )


def test_ens_chain_end_to_end(tmp_path):
    forecast_root, babj_dir = _make_dataset(tmp_path)
    out = _run(tmp_path, forecast_root, babj_dir)
    case = runner.read_case(out, "2501", INIT_ISO)
    assert list(case.columns) == COLUMNS
    assert len(case.columns) == 20
    assert len(case) == 4
    # 方案 B（主列）= 集合平均位置 vs 实况；方案 A（*_a）= 成员误差平均
    assert case["track_err_km"].to_numpy() == pytest.approx([EXPECTED_B_KM] * 4, abs=1e-3)
    assert case["track_err_km_a"].to_numpy() == pytest.approx([EXPECTED_A_KM] * 4, abs=1e-3)
    # 首时次 at/ct 用起报时刻实况定向（ob0 存在 → 非空）
    assert case["at_km"].to_numpy() == pytest.approx([-EXPECTED_B_KM] * 4, abs=1e-3)
    assert case["at_km_a"].to_numpy() == pytest.approx([-EXPECTED_B_KM] * 4, abs=1e-3)
    assert case["ct_km"].to_numpy() == pytest.approx([0.0] * 4, abs=1e-3)
    assert case["n_members"].tolist() == [2] * 4
    assert case["n_valid_members"].tolist() == [2] * 4
    assert case["wind_err_ms"].to_numpy() == pytest.approx([5.0] * 4, abs=1e-3)
    assert case["pmin_err_hpa"].to_numpy() == pytest.approx([-10.0] * 4, abs=1e-3)
    meta = runner.read_meta(out, "2501", INIT_ISO)
    assert meta["forecast_type"] == "ens"
    assert meta["n_members"] == 2 and meta["members"] == ["0", "1"]
    assert set(meta["aggregation"]) == {"A", "B"}
    assert meta["seed"] == "init+6h"
    assert meta["init_pos"] == pytest.approx([OBS_LAT[0], OBS_LON])  # 链的种子位置
    scores = runner.read_scores(out)
    assert len(scores) == 8  # 5 个确定性 field + 3 个 *_a
    assert set(scores["status"]) == {"success"}
    assert set(scores["product_kind"]) == {"ensemble"}


def test_ens_first_lead_anchor_requires_init_obs(tmp_path):
    forecast_root, babj_dir = _make_dataset(tmp_path, with_init_obs=False)
    out = _run(tmp_path, forecast_root, babj_dir)
    case = runner.read_case(out, "2501", INIT_ISO)
    assert math.isnan(case["at_km"].iloc[0]) and math.isnan(case["at_km_a"].iloc[0])
    assert math.isfinite(case["at_km"].iloc[1])  # 第二时效起用上一实况定向
    assert case["track_err_km"].to_numpy() == pytest.approx([EXPECTED_B_KM] * 4, abs=1e-3)


def test_ens_member_failure_is_skipped(tmp_path):
    # member_001 的文件坏掉：该成员被跳过，其余照常出结果
    forecast_root, babj_dir = _make_dataset(tmp_path, corrupt_member="member_001")
    out = _run(tmp_path, forecast_root, babj_dir)
    case = runner.read_case(out, "2501", INIT_ISO)
    meta = runner.read_meta(out, "2501", INIT_ISO)
    assert meta["n_members"] == 1 and meta["members"] == ["0"]
    assert case["n_valid_members"].tolist() == [1] * 4
    assert case["track_err_km"].to_numpy() == pytest.approx([EXPECTED_B_KM] * 4, abs=1e-3)
    # 只剩 +0.25° 的成员：at 翻符号（两成员时平均位置在 −0.25°）
    assert case["at_km"].to_numpy() == pytest.approx([EXPECTED_B_KM] * 4, abs=1e-3)
