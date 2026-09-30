# -*- coding: utf-8 -*-
"""测试共享工具：合成台风数据工厂 + 评测运行器 + 产物读取。

放在 ``tests/`` 外层，供 unit / integration 共用；这里**不放任何评测逻辑**
——评测全在主仓库。本模块只负责"造合成输入、按真实配置跑一条链、把产物读回来"。

坐标约定与主仓库一致：经纬度对一律 ``(lat, lon)``；init 用 UTC，
BABJ 报文用北京时（init 的北京时 = init + 8h）。
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

#: 同经度上 1° 纬度的大圆距离（r=6371 km），用来手算预期路径误差
KM_PER_DEG = math.pi * 6371.0 / 180.0

#: 合成网格：0.25°，所有设计位置都落在格点上
GRID_LAT = np.round(np.arange(15.0, 27.0 + 1e-9, 0.25), 6)
GRID_LON = np.round(np.arange(115.0, 127.0 + 1e-9, 0.25), 6)

#: 合成场参数：中心 95000 Pa = 950 hPa；风场恒定（vmax 就是它）
MSL_BASE_PA = 101000.0
MSL_MIN_PA = 95000.0
MSL_SIGMA_DEG = 2.0
WIND_MS = 30.0

#: 测试一律串行：不并发、不复用 resume 状态
SERIAL_EXECUTION = {"mode": "serial", "n_workers": 1, "resume": False}


# --------------------------------------------------------------------------
# 合成输入工厂
# --------------------------------------------------------------------------

def gaussian_msl(
    center_lat: float,
    center_lon: float,
    *,
    lat: np.ndarray = GRID_LAT,
    lon: np.ndarray = GRID_LON,
    base_pa: float = MSL_BASE_PA,
    min_pa: float = MSL_MIN_PA,
    sigma_deg: float = MSL_SIGMA_DEG,
) -> np.ndarray:
    """以 ``(center_lat, center_lon)`` 为极小值的高斯海平面气压场（Pa）。"""
    d2 = (lat[:, None] - center_lat) ** 2 + (lon[None, :] - center_lon) ** 2
    return base_pa - (base_pa - min_pa) * np.exp(-d2 / (2.0 * sigma_deg ** 2))


def _write_field_file(path: Path, lat, lon, center, *, u_ms: float, v_ms: float) -> None:
    dataset = xr.Dataset(
        {
            "msl": (("lat", "lon"), gaussian_msl(center[0], center[1], lat=lat, lon=lon)),
            "u10m": (("lat", "lon"), np.full((lat.size, lon.size), float(u_ms))),
            "v10m": (("lat", "lon"), np.full((lat.size, lon.size), float(v_ms))),
        },
        coords={"lat": lat, "lon": lon},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)


def write_fuxi_single(
    root,
    init_utc: datetime,
    centers: Sequence[Tuple[float, float]],
    *,
    lat=GRID_LAT,
    lon=GRID_LON,
    u_ms: float = WIND_MS,
    v_ms: float = 0.0,
) -> None:
    """写确定性 FuXi 目录：``root/YYYYMMDD/001.nc…``。

    ``centers`` 按 lead 升序给每个时效的台风中心；第 k 个文件（001.nc）对应
    时效 6k h（布局 ``lead_from="index"``，文件顺序即时效顺序）。
    """
    date_dir = Path(root) / init_utc.strftime("%Y%m%d")
    for index, center in enumerate(centers, start=1):
        _write_field_file(date_dir / f"{index:03d}.nc", lat, lon, center, u_ms=u_ms, v_ms=v_ms)


def write_fuxi_ens(
    root,
    init_utc: datetime,
    member_centers: Mapping[str, Sequence[Tuple[float, float]]],
    *,
    lat=GRID_LAT,
    lon=GRID_LON,
    u_ms: float = WIND_MS,
    v_ms: float = 0.0,
) -> None:
    """写集合 FuXi 目录：``root/YYYYMMDD/member_XXX/001.nc…``。

    ``member_centers`` 形如 ``{"member_000": [(lat, lon), ...], "member_001": [...]}``。
    """
    date_dir = Path(root) / init_utc.strftime("%Y%m%d")
    for member, centers in member_centers.items():
        for index, center in enumerate(centers, start=1):
            _write_field_file(
                date_dir / member / f"{index:03d}.nc", lat, lon, center, u_ms=u_ms, v_ms=v_ms
            )


def write_babj(
    path,
    tcname: str,
    tcid: str,
    rows: Sequence[Tuple[datetime, float, float, float, float, float]],
) -> None:
    """写 BABJ diamond7 报文（GBK）。

    ``rows``: ``(北京时 datetime, 时效 h, lon, lat, pmin_hpa, vmax_ms)``；
    时效 000 的行才是分析实况（读取端只认这些行，其余行用来测过滤）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(("diamond 7 %s台风路径\n" % tcname).encode("gbk"))
        stream.write(
            ("%s      %s   20            %d\n" % (tcname, tcid, len(rows))).encode("gbk")
        )
        stream.write(b"\n")
        for moment, tau, lon, lat, pmin, vmax in rows:
            stream.write(
                (
                    "%4d %3d %3d %3d %3d %11.1f %11.1f %6d %5d"
                    "     0.0       0.0       0.0       0.0\n"
                    % (
                        moment.year,
                        moment.month,
                        moment.day,
                        moment.hour,
                        int(tau),
                        lon,
                        lat,
                        int(pmin),
                        int(vmax),
                    )
                ).encode("ascii")
            )


# --------------------------------------------------------------------------
# 运行器与产物读取
# --------------------------------------------------------------------------

def compact_init(init_utc: str) -> str:
    """``"2025-06-10T00:00:00"`` -> ``"2025061000"``（逐场次文件名口径）。"""
    digits = "".join(ch for ch in str(init_utc) if ch.isdigit())
    return digits[:10]


def run_config(
    config_name: str,
    *,
    forecast_root,
    babj_root,
    output_dir,
    start: str,
    end: str,
    options: Optional[Dict[str, Any]] = None,
    execution: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> Path:
    """按**真实配置**跑一条链，数据路径 / 时段 / 并发覆盖成测试值。

    走 ``cli.run_evaluation``（与命令行同一入口）；返回产物目录。
    """
    from xmetai_evaluation.cli import run_evaluation
    from xmetai_evaluation.configs.base import load_config

    cfg = load_config(config_name)
    cfg.forecast_reader = {**cfg.forecast_reader, "root_dir": str(forecast_root)}
    cfg.observation_reader = {**cfg.observation_reader, "root_dir": str(babj_root)}
    cfg.output_dir = str(output_dir)
    cfg.start_date = str(start)
    cfg.end_date = str(end)
    cfg.limit = limit
    if options:
        cfg.options = {**cfg.options, **options}
    cfg.execution = {**cfg.execution, **SERIAL_EXECUTION, **(execution or {})}
    run_evaluation(cfg)
    return Path(output_dir)


def case_dir(output_dir) -> Path:
    """逐场次产物目录（``typhoon/``）。"""
    return Path(output_dir) / "typhoon"


def read_case(output_dir, tcid: str, init_utc: str) -> pd.DataFrame:
    """读 ``typhoon/tc<编号>_<起报>.csv``（逐时效曲线）。"""
    return pd.read_csv(case_dir(output_dir) / f"tc{tcid}_{compact_init(init_utc)}.csv")


def read_meta(output_dir, tcid: str, init_utc: str) -> Dict[str, Any]:
    """读 ``typhoon/tc<编号>_<起报>_meta.json``。"""
    path = case_dir(output_dir) / f"tc{tcid}_{compact_init(init_utc)}_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_scores(output_dir) -> pd.DataFrame:
    """读统一长表 ``scores.csv``。"""
    return pd.read_csv(Path(output_dir) / "scores.csv")
