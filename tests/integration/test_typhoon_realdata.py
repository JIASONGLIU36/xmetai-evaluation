# -*- coding: utf-8 -*-
"""真实数据档：BABJ 报文 / 确定性预报 / 集合预报 × 标准归档对拍。

路径全部经环境变量提供，未配置或路径不存在时自动 skip：

- ``XMETAI_BABJ_DIR``       BABJ 报文目录
- ``XMETAI_FUXI_DET_ROOT``  确定性 FuXi 预报根（``root/YYYYMMDD/001.nc…``）
- ``XMETAI_FUXI_ENS_ROOT``  集合 FuXi 预报根（``root/YYYYMMDD/member_*/001.nc…``）
- ``XMETAI_TC_DET_ARCHIVE`` 确定性标准归档（tc/fuxi）
- ``XMETAI_TC_ENS_ARCHIVE`` 集合标准归档（tc/ens/fuxi）

运行：``pytest -m realdata``；不选中时这些用例不会跑。
"""
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests import runner

pytestmark = pytest.mark.realdata
pytest.importorskip("netCDF4")

BABJ_DIR = os.environ.get("XMETAI_BABJ_DIR")
FUXI_DET_ROOT = os.environ.get("XMETAI_FUXI_DET_ROOT")
FUXI_ENS_ROOT = os.environ.get("XMETAI_FUXI_ENS_ROOT")
TC_DET_ARCHIVE = os.environ.get("XMETAI_TC_DET_ARCHIVE")
TC_ENS_ARCHIVE = os.environ.get("XMETAI_TC_ENS_ARCHIVE")

ENS_INIT_ISO = "2025-06-10T00:00:00"
DET_INIT_ISO = "2025-07-03T00:00:00"


def _require(value, name):
    if not value:
        pytest.skip(f"未配置环境变量 {name}")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{name}={value} 不存在")
    return path


def test_real_babj_seed_regression():
    """真实报文上的种子回归：2507@0722 非网格记录、2503@0703 首配。"""
    babj_dir = _require(BABJ_DIR, "XMETAI_BABJ_DIR")
    from xmetai_evaluation.core.contracts import DataRequest
    from xmetai_evaluation.io.babj_reader import BabjCatalog, BabjReader
    from xmetai_evaluation.pipeline.protocols import babj_seed_moment

    reader = BabjReader(source_id="babj")
    request = DataRequest(source_id="babj", variables=["storm"])
    payload = reader.read(request, BabjCatalog(babj_dir).discover(request)).payload
    names = [str(value) for value in payload["storm"].values]
    assert {"2501", "2503", "2507"} <= set(names)

    def moments(tcid):
        index = names.index(tcid)
        column = payload["lat"].values[:, index]
        return {
            payload["time"].values[k].astype("datetime64[us]").item()
            for k in range(column.size)
            if np.isfinite(column[k])
        }

    # 2507@0722：17:00 是非 6h 网格记录 → 种子必须取 20:00（归档回归案例）
    assert babj_seed_moment(moments("2507"), datetime(2025, 7, 22, 8, 0), 6.0, 6.0) == datetime(
        2025, 7, 22, 20, 0
    )
    # 2503@0703：第一条能配上时效的实况 = 14:00
    assert babj_seed_moment(moments("2503"), datetime(2025, 7, 3, 8, 0), 6.0, 6.0) == datetime(
        2025, 7, 3, 14, 0
    )


def test_real_det_parity_20250703(tmp_path):
    """真跑 2503@0703（确定性），与标准归档逐列对拍（本机实测逐位一致）。"""
    babj_dir = _require(BABJ_DIR, "XMETAI_BABJ_DIR")
    det_root = _require(FUXI_DET_ROOT, "XMETAI_FUXI_DET_ROOT")
    archive = _require(TC_DET_ARCHIVE, "XMETAI_TC_DET_ARCHIVE")
    out = runner.run_config(
        "weather_typhoon_single_fuxi",
        forecast_root=det_root,
        babj_root=babj_dir,
        output_dir=tmp_path / "out",
        start="20250703",
        end="20250703",
        options={"storm_ids": ["2503"]},
    )
    ours = runner.read_case(out, "2503", DET_INIT_ISO)
    reference = pd.read_csv(archive / "tc2503_2025070300.csv")
    assert list(ours.columns) == list(reference.columns)
    assert len(ours) == len(reference) == 60
    assert (ours["valid_bjt"] == reference["valid_bjt"]).all()
    for column in reference.columns:
        if column == "valid_bjt":
            continue
        actual = ours[column].to_numpy(dtype=float)
        expected = reference[column].to_numpy(dtype=float)
        assert np.allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True), column
    meta = runner.read_meta(out, "2503", DET_INIT_ISO)
    archive_meta = json.loads((archive / "tc2503_2025070300_meta.json").read_text("utf-8"))
    assert meta["init_pos"] == pytest.approx(archive_meta["init_pos"])  # 种子位置
    assert meta["n_matched"] == archive_meta["n_matched"] == 18
    assert meta["seed"] == "init+6h"


def test_real_ens_parity_20250610(tmp_path):
    """真跑 2501@0610（集合），用归档逐成员表重算 3 成员口径的期望值对拍。"""
    babj_dir = _require(BABJ_DIR, "XMETAI_BABJ_DIR")
    ens_root = _require(FUXI_ENS_ROOT, "XMETAI_FUXI_ENS_ROOT")
    archive = _require(TC_ENS_ARCHIVE, "XMETAI_TC_ENS_ARCHIVE")
    member_dirs = sorted((ens_root / "20250610").glob("member_*"))
    if not member_dirs:
        pytest.skip(f"{ens_root}/20250610 下没有成员目录")
    member_names = [f"member_{int(path.name.split('_')[1]):03d}" for path in member_dirs]

    out = runner.run_config(
        "weather_typhoon_ens_fuxi",
        forecast_root=ens_root,
        babj_root=babj_dir,
        output_dir=tmp_path / "out",
        start="20250610",
        end="20250610",
        options={"storm_ids": ["2501"]},
    )
    ours = runner.read_case(out, "2501", ENS_INIT_ISO).set_index("lead_h")
    meta = runner.read_meta(out, "2501", ENS_INIT_ISO)
    assert meta["n_members"] == len(member_names)
    assert meta["seed"] == "init+6h"

    reference = pd.read_csv(archive / "tc2501_2025061000_members.csv")
    reference = reference[reference["member"].isin(member_names)].copy()
    reference["lead_h"] = reference["lead_h"].astype(float)
    grouped = reference.groupby("lead_h", sort=True)
    leads = [float(value) for value in sorted(reference["lead_h"].unique())]
    obs_lat = grouped["obs_lat"].first().reindex(leads)
    obs_lon = grouped["obs_lon"].first().reindex(leads)
    mean_lat = grouped["fcst_lat"].mean().reindex(leads)
    mean_lon = grouped["fcst_lon"].mean().reindex(leads)

    # 成员独立列逐位一致；A 类 = 归档逐成员误差的均值
    assert (ours.loc[leads, "valid_bjt"] == grouped["valid_bjt"].first().reindex(leads)).all()
    assert np.allclose(ours.loc[leads, "obs_lat"], obs_lat, rtol=0, atol=1e-9, equal_nan=True)
    assert np.allclose(
        ours.loc[leads, "track_err_km_a"], grouped["track_err_km"].mean().reindex(leads),
        rtol=1e-6, atol=0.01, equal_nan=True,
    )
    assert np.allclose(
        ours.loc[leads, "at_km_a"], grouped["at_km"].mean().reindex(leads),
        rtol=1e-6, atol=0.01, equal_nan=True,
    )
    assert np.allclose(
        ours.loc[leads, "ct_km_a"], grouped["ct_km"].mean().reindex(leads),
        rtol=1e-6, atol=0.01, equal_nan=True,
    )

    # B 类：成员位置先平均、再与实况求误差
    from xmetai_evaluation.pipeline.typhoon import along_cross_track, great_circle_km

    obs_lat_arr = obs_lat.to_numpy(dtype=float)
    obs_lon_arr = obs_lon.to_numpy(dtype=float)
    mean_lat_arr = mean_lat.to_numpy(dtype=float)
    mean_lon_arr = mean_lon.to_numpy(dtype=float)
    obs_ok = np.isfinite(obs_lat_arr) & np.isfinite(mean_lat_arr)
    expected_b = np.full(len(leads), np.nan)
    expected_b[obs_ok] = great_circle_km(
        mean_lat_arr[obs_ok], mean_lon_arr[obs_ok], obs_lat_arr[obs_ok], obs_lon_arr[obs_ok]
    )
    assert np.allclose(
        ours.loc[leads, "track_err_km"], expected_b, rtol=1e-6, atol=0.01, equal_nan=True
    )

    # 首时次 anchor = 起报时刻实况（用真报文独立取，08:00 北京时）
    from xmetai_evaluation.pipeline.typhoon import read_babj_analyses

    _, _, analyses = read_babj_analyses(babj_dir / "babj2501.dat")
    anchor = analyses[datetime(2025, 6, 10, 8, 0)]
    prev = (anchor["lat"], anchor["lon"])
    expected_at = np.full(len(leads), np.nan)
    expected_ct = np.full(len(leads), np.nan)
    for k, _lead in enumerate(leads):
        if not obs_ok[k]:
            continue
        expected_at[k], expected_ct[k] = along_cross_track(
            prev[0], prev[1], obs_lat_arr[k], obs_lon_arr[k], mean_lat_arr[k], mean_lon_arr[k]
        )
        prev = (obs_lat_arr[k], obs_lon_arr[k])
    assert np.allclose(
        ours.loc[leads, "at_km"], expected_at, rtol=1e-6, atol=0.01, equal_nan=True
    )
    assert np.allclose(
        ours.loc[leads, "ct_km"], expected_ct, rtol=1e-6, atol=0.01, equal_nan=True
    )

    # 计数：n_valid_members 只对配上的时次有值；n_matched 与实况覆盖一致
    expected_valid = (
        grouped["fcst_lat"].apply(lambda column: int(np.isfinite(column).sum())).reindex(leads)
    )
    assert np.array_equal(
        ours.loc[leads, "n_valid_members"].to_numpy()[obs_ok],
        expected_valid.to_numpy()[obs_ok],
    )
    assert meta["n_matched"] == int(obs_ok.sum())
    assert math.isfinite(float(ours.loc[leads[0], "at_km"]))  # 首时次有 anchor

    # init_pos = 链的种子位置（= lead6 的实况位置）
    seed_lat = float(reference.loc[reference["lead_h"] == 6.0, "obs_lat"].iloc[0])
    seed_lon = float(reference.loc[reference["lead_h"] == 6.0, "obs_lon"].iloc[0])
    assert meta["init_pos"] == pytest.approx([seed_lat, seed_lon])
