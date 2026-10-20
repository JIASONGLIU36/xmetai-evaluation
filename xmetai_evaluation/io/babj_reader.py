# -*- coding: utf-8 -*-
"""BABJ 台风路径报文 Reader。

一个 ``babj<编号>.dat`` 文件 = 一号台风的**整条路径**（diamond7、GBK 编码、
空白分隔）。报文格式：

  第 1 行 diamond 7 头；第 2 行 台风名 编号 …；第 3 行空；
  数据行  yy mm dd hh 时效 lon lat 中心气压 最大风速(m/s) …
  **时效列 = 000 的行是分析实况**，其余为 BABJ 主观预报（不用）。
  时间为**北京时**（UTC 预报需 +8h 对时）。

与其它观测源的关键差别，也是本 Reader 不按时间筛文件的原因：

- **一条报文横跨多天**，不是「一个文件一个时刻」。``station_reader`` 那种
  按 ``[start, end]`` 挑文件的选法在这里无意义；
- 台风评测要的是「某个编号在某起报时刻的实况位置及其整条路径」，报文数量
  以十计、体量以 KB 计，所以 ``discover`` 一把全给、``read`` 一次全读，
  由协议按编号取用。

输出（``dims=(time, storm)``）：

- ``lat`` / ``lon`` / ``pmin`` / ``vmax``：分析实况（只取报文里时效 000 的行）；
- 坐标 ``storm`` = 台风编号字符串、``tcname`` = 中文名；
- 各台风的实况时刻并不对齐，``time`` 取**并集**、缺的位置为 NaN——
  取用时按 ``.sel(storm=..., time=...)`` 取值，NaN 即「该时刻无实况」。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import (
    DataBundle,
    DataIndex,
    DataRequest,
    DataKind,
    Provenance,
    SemanticMetadata,
)
from xmetai_evaluation.core.errors import DecodeError, DiscoveryError, TyphoonDataError
from xmetai_evaluation.core.variables import TemporalKind
from xmetai_evaluation.io.base import DataCatalog, Reader

log = logging.getLogger(__name__)


def read_babj_analyses(path) -> Tuple[str, str, Dict[datetime, Dict[str, float]]]:
    """读 babj diamond7，只取时效 000 的实况记录。

    返回 ``(tcname, tcid, {北京时datetime: dict(lon, lat, pmin_hpa, vmax_ms)})``。

    列序是 **lon 在前、lat 在后**（第 6、7 列），别按经纬顺序读反。
    """
    try:
        with open(path, "rb") as f:
            f.readline()
            head = f.readline().decode("gbk", "ignore").split()
        tcname, tcid = head[0], head[1]
        rows = np.loadtxt(path, skiprows=3, encoding="gbk", ndmin=2)
    except (OSError, ValueError, IndexError) as e:
        raise DecodeError("解析 babj 报文 %s 失败: %s" % (path, e)) from e
    analyses: Dict[datetime, Dict[str, float]] = {}
    for r in rows:
        if int(round(r[4])) != 0:  # 只要实况行
            continue
        t = datetime(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        analyses[t] = {
            "lon": float(r[5]),
            "lat": float(r[6]),
            "pmin_hpa": float(r[7]),
            "vmax_ms": float(r[8]),
        }
    if not analyses:
        raise TyphoonDataError("%s 中没有时效 000 的实况记录" % path)
    return tcname, tcid, analyses


class BabjCatalog(DataCatalog):
    """BABJ 报文目录：``babj*.dat``（也接受直接指到一个报文文件）。"""

    def __init__(self, babj_dir: Path):
        """
        Args:
            babj_dir: 报文目录，或单个 ``babj<编号>.dat`` 文件
        """
        self.babj_dir = Path(babj_dir)

    def discover(self, request: DataRequest) -> DataIndex:
        """列出目录下全部报文。**不做时间筛选**（理由见模块 docstring）。"""
        if not self.babj_dir.exists():
            raise DiscoveryError(
                f"BABJ 路径不存在: {self.babj_dir}", source_id=request.source_id
            )
        if self.babj_dir.is_file():
            return DataIndex(source_id=request.source_id, available=[self.babj_dir], ambiguous=[])
        files = sorted(self.babj_dir.glob("babj*.dat"))
        if not files:
            raise DiscoveryError(
                f"{self.babj_dir} 下没有 babj*.dat", source_id=request.source_id
            )
        return DataIndex(source_id=request.source_id, available=files, ambiguous=[])

    def file_time(self, path) -> datetime:
        """文件里最早的实况时刻（供按时间选文件的调用方兜底）。"""
        _, _, analyses = read_babj_analyses(path)
        return min(analyses)


class BabjReader(Reader):
    """读全部报文，拼成 ``(time, storm)`` 的分析实况表。"""

    def __init__(self, source_id: str = "babj", version: str = "1.0.0"):
        super().__init__(source_id=source_id, version=version)

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        storms: List[Dict[str, Any]] = []
        for path in index.available:
            tcname, tcid, analyses = read_babj_analyses(path)
            storms.append({"tcid": str(tcid), "tcname": tcname, "analyses": analyses})

        if not storms:
            raise DiscoveryError("没有可读的 BABJ 报文", source_id=request.source_id)

        times = sorted({t for storm in storms for t in storm["analyses"]})
        time_index = np.array(times, dtype="datetime64[ns]")
        tcid_index = np.array([storm["tcid"] for storm in storms], dtype=object)
        tcname_index = np.array([storm["tcname"] for storm in storms], dtype=object)

        shape = (len(times), len(storms))
        lat = np.full(shape, np.nan, dtype="f8")
        lon = np.full(shape, np.nan, dtype="f8")
        pmin = np.full(shape, np.nan, dtype="f8")
        vmax = np.full(shape, np.nan, dtype="f8")
        slot = {t: i for i, t in enumerate(times)}
        for j, storm in enumerate(storms):
            for moment, record in storm["analyses"].items():
                i = slot[moment]  # 该时刻必有槽位（times 是全并集）
                lat[i, j] = record["lat"]
                lon[i, j] = record["lon"]
                pmin[i, j] = record["pmin_hpa"]
                vmax[i, j] = record["vmax_ms"]

        ds = xr.Dataset(
            {
                "lat": (("time", "storm"), lat),
                "lon": (("time", "storm"), lon),
                "pmin": (("time", "storm"), pmin),
                "vmax": (("time", "storm"), vmax),
            },
            coords={
                "time": time_index,
                "storm": tcid_index,
                "tcname": ("storm", tcname_index),
            },
        )
        ds["lat"].attrs.update(units="degrees_north", long_name="Storm centre latitude")
        ds["lon"].attrs.update(units="degrees_east", long_name="Storm centre longitude")
        ds["pmin"].attrs.update(units="hPa", long_name="Minimum sea level pressure")
        ds["vmax"].attrs.update(units="m/s", long_name="Maximum sustained wind")

        log.info(
            "BABJ 报文读取完成：共 %d 个台风编号、%d 个实况时刻（%s 到 %s）；"
            "这是**整个报文目录**的规模，不代表本次评测评了这么多——"
            "评哪些由时段和 storm_ids 决定",
            len(storms),
            len(times),
            times[0] if times else "—",
            times[-1] if times else "—",
        )

        return DataBundle(
            payload=ds,
            kind=DataKind.STATION_OBSERVATION,
            source_id=request.source_id,
            standard_vars={name: name for name in ("lat", "lon", "pmin", "vmax")},
            semantic=SemanticMetadata(
                units={"lat": "degrees_north", "lon": "degrees_east", "pmin": "hPa", "vmax": "m/s"},
                temporal_kind=TemporalKind.INSTANTANEOUS,
                grid_type="storm_track",
                # 报文时刻是**北京时**，与格点预报的 UTC 差 8h——
                # 协议按 tz_shift 对齐，不在这里改轴。
                timezone="Asia/Shanghai",
            ),
            provenance=Provenance(
                input_files=[str(path) for path in index.available],
                reader_id=self.source_id,
                reader_version=self.version,
            ),
        )
