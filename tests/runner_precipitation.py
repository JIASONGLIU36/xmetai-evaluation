# -*- coding: utf-8 -*-
"""降水/站点产品线的合成数据工厂：Diamond 逐时站点降水文件。

与 ``runner.py``（台风 BABJ 报文的 diamond 7）并列的一层：本模块**只造输入**，
不含任何评测逻辑——评测全在主仓库。

文件格式（GBK，两行文件头 + 空行 + 数据行）::

    diamond  3 2025年8月20日0时1小时降水(逐时)
    25 8 20 0 1000 0 0 0 0 1 77017

    50005    112.0000     21.0000    5.0    0.0
    站号 经度 纬度 海拔(m) 降水(mm)

命名支持读取端认的两种布局：分层 ``root/YYYYMMDD/HH.000``（``station_file``）
与扁平 ``root/YYYYMMDDHH.000``（``flat_station_file``）。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

#: 数据行列宽（与参考实现的 diamond 3 站点列一致）
_ROW_FORMAT = "%5d %11.4f %11.4f %7.1f %7.1f"

#: 一行数据 = (站号, 经度, 纬度, 海拔 m, 降水 mm)
StationRow = Tuple[int, float, float, float, float]


def station_file(root, moment: datetime) -> Path:
    """分层布局路径：``root/YYYYMMDD/HH.000``。"""
    return Path(root) / moment.strftime("%Y%m%d") / (moment.strftime("%H") + ".000")


def flat_station_file(root, moment: datetime) -> Path:
    """扁平布局路径：``root/YYYYMMDDHH.000``。"""
    return Path(root) / (moment.strftime("%Y%m%d%H") + ".000")


def write_diamond(path, moment: datetime, rows: Sequence[StationRow]) -> None:
    """写一个时刻的 Diamond 3 站点降水文件（GBK）。

    Args:
        path: 目标文件路径，通常由 ``station_file`` / ``flat_station_file`` 给出。
        moment: 该文件的观测时刻。
        rows: ``(站号, 经度, 纬度, 海拔 m, 降水 mm)``；写入顺序即文件顺序
            （读取端会按站号重排，用来测\"文件顺序与站号顺序不一致\"）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "diamond  3 %d年%d月%d日%d时1小时降水(逐时)"
        % (moment.year, moment.month, moment.day, moment.hour),
        "%02d %d %d %d 1000 0 0 0 0 1 77017"
        % (moment.year % 100, moment.month, moment.day, moment.hour),
        "",
    ]
    lines.extend(_ROW_FORMAT % row for row in rows)
    path.write_bytes(("\n".join(lines) + "\n").encode("gbk"))


def write_fuxi_tp(
    root, init_utc: datetime, *, lat, lon, values: np.ndarray
) -> Path:
    """写四个 FuXi tp 预报文件（时效 6/12/18/24，``root/YYYYMMDD/001.nc…``）。

    四个文件用同一张 ``values`` 场（``(lat.size, lon.size)``，mm），于是 24h 累积
    = 4 × 单时效值——测试把站点摆在格点上，就能手算插值后的 24h 总量。
    """
    root = Path(root)
    date_dir = root / init_utc.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 5):
        dataset = xr.Dataset(
            {"tp": (("lat", "lon"), np.asarray(values, dtype="f8"))},
            coords={"lat": np.asarray(lat, dtype="f8"), "lon": np.asarray(lon, dtype="f8")},
        )
        dataset["tp"].attrs["units"] = "mm"
        dataset.to_netcdf(date_dir / f"{index:03d}.nc")
    return root


def run_synthetic_ts_config(*, forecast_root, obs_root, output_dir) -> Path:
    """按真实 ``weather_ts_single_fgvp`` 配置跑一条链，只覆盖数据/时段/阈值/并发。"""
    from xmetai_evaluation.cli import run_evaluation
    from xmetai_evaluation.configs.base import load_config

    cfg = load_config("weather_ts_single_fgvp")
    cfg.forecast_reader = {
        **cfg.forecast_reader,
        "root_dir": str(forecast_root),
        "lead_times": [6, 12, 18, 24],
    }
    cfg.observation_reader = {
        **cfg.observation_reader,
        "root_dir": str(obs_root),
        "station_list": None,  # 用文件里的全部站点，不读白名单
    }
    cfg.start_date = "20250101"
    cfg.end_date = "20250101"
    cfg.periods = None          # 生产配置已改走多时段 periods；这里只跑单时段，必须清掉
    cfg.output_dir = str(output_dir)
    cfg.limit = None
    # 只保留全球行：丢掉生产配置的分纬度带，对齐口径保持 UTC（offset 0）。
    cfg.options = {"local_utc_offset_hours": 0}
    cfg.metric_options = {"ts_score": {"thresholds": [0.1, 10.0]}}
    cfg.execution = {"mode": "serial", "n_workers": 1, "resume": False}
    run_evaluation(cfg)
    return Path(output_dir)


def read_scores(output_dir) -> pd.DataFrame:
    """读统一长表 ``scores.csv``。"""
    return pd.read_csv(Path(output_dir) / "scores.csv")


def read_categorical_wide(output_dir) -> pd.DataFrame:
    """读 ``diagnostics/categorical_wide.csv``（TS 宽表，含整数列联表）。"""
    return pd.read_csv(Path(output_dir) / "diagnostics" / "categorical_wide.csv")


def read_probability_wide(output_dir) -> pd.DataFrame:
    """读 ``diagnostics/probability_wide.csv``（集合概率宽表 AROC/BS/BSS）。"""
    return pd.read_csv(Path(output_dir) / "diagnostics" / "probability_wide.csv")


def write_fuxi_ens_tp(root, init_utc: datetime, *, lat, lon, member_values) -> Path:
    """写集合 FuXi tp 预报：``root/YYYYMMDD/member_XXX/001..004.nc``。

    ``member_values``：``{"member_000": [[...]], "member_001": [[...]]}``，每个成员的
    tp 场 ``(lat.size, lon.size)``；四个时效共用同一张场，24h 累积 = 4 × 单时效值。
    """
    root = Path(root)
    date_dir = root / init_utc.strftime("%Y%m%d")
    for member, values in member_values.items():
        member_dir = date_dir / member
        member_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, 5):
            dataset = xr.Dataset(
                {"tp": (("lat", "lon"), np.asarray(values, dtype="f8"))},
                coords={
                    "lat": np.asarray(lat, dtype="f8"),
                    "lon": np.asarray(lon, dtype="f8"),
                },
            )
            dataset["tp"].attrs["units"] = "mm"
            dataset.to_netcdf(member_dir / f"{index:03d}.nc")
    return root


def write_ref_probability(root, when: datetime, station_rows) -> Path:
    """写 ``ref/MMDDHH.000``：每行 ``站号 p≥0.1 p≥4 p≥13 p≥25``（4 列概率）。"""
    path = Path(root) / f"{when.strftime('%m%d%H')}.000"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "%d %.3f %.3f %.3f %.3f" % (sid, p0, p4, p13, p25)
        for (sid, p0, p4, p13, p25) in station_rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_synthetic_ts_ens_config(
    *, forecast_root, obs_root, ref_root, output_dir
) -> Path:
    """按真实 ``weather_ts_ens_fuxi`` 配置跑一条链，覆盖数据/时段/阈值/并发。"""
    from xmetai_evaluation.cli import run_evaluation
    from xmetai_evaluation.configs.base import load_config

    cfg = load_config("weather_ts_ens_fuxi")
    cfg.forecast_reader = {
        **cfg.forecast_reader,
        "root_dir": str(forecast_root),
        "lead_times": [6, 12, 18, 24],
    }
    cfg.observation_reader = {
        **cfg.observation_reader,
        "root_dir": str(obs_root),
        "station_list": None,
    }
    cfg.reference_reader = {**cfg.reference_reader, "root_dir": str(ref_root)}
    cfg.start_date = "20250101"
    cfg.end_date = "20250101"
    cfg.output_dir = str(output_dir)
    cfg.limit = None
    # 只保留全球行（丢生产配置的分纬度带），对齐口径保持北京时（offset 8）。
    cfg.options = {"local_utc_offset_hours": 8}
    cfg.metric_options = {
        "ts_score": {"thresholds": [10.0]},
        "ensemble_probability": {"thresholds": [0.1]},
    }
    cfg.execution = {"mode": "serial", "n_workers": 1, "resume": False}
    run_evaluation(cfg)
    return Path(output_dir)
