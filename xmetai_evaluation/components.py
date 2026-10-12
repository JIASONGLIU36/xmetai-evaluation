# -*- coding: utf-8 -*-
"""内置组件注册。

所有内置 Reader / Transform / Metric / Writer 的工厂在这里登记一次，
cli 与各流程只按注册名查表构造组件，不再按类型分支判断。

扩展方式：
    新增数据源或算法 -> 在这里补一个注册项（或在自己的插件模块里注册）
    新增评测         -> 只加配置，代码不动
"""

from __future__ import annotations

import csv
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.core.registry import (
    get_registry,
    register_metric,
    register_protocol,
    register_reader,
    register_transform,
    register_writer,
)


@dataclass
class SourceHandle:
    """一个数据源的读写句柄：Reader + Catalog + 原始配置。"""

    reader: Any
    catalog: Any
    config: Dict[str, Any]

    @property
    def source_id(self) -> str:
        return self.config.get("source_id") or getattr(self.reader, "source_id", "")

    def option(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


def _require_root(params: Dict[str, Any], reader_name: str) -> Path:
    root = params.get("root_dir") or params.get("root")
    if not root:
        raise ConfigError(f"reader '{reader_name}' 需要 root_dir")
    return Path(root)


def _gridded_source(
    layout, params: Dict[str, Any], default_source_id: str, step_hours=None
) -> SourceHandle:
    """通用格点数据源：具名布局 + 通用 Reader/Catalog。

    数据源之间的差别只有布局声明（见 ``io/layouts.py``），
    所以这里不需要为每个模型写一个 Reader 类。
    """
    from xmetai_evaluation.io.gridded import GriddedCatalog, GriddedReader

    selected = layout.with_step_hours(step_hours) if step_hours is not None else layout
    source_id = params.get("source_id", default_source_id)
    root = _require_root(params, selected.name)
    return SourceHandle(
        reader=GriddedReader(source_id=source_id, layout=selected),
        catalog=GriddedCatalog(root, selected),
        config=params,
    )


# --------------------------------------------------------------------------
# Reader：只负责"文件怎么变成标准 Dataset"
# --------------------------------------------------------------------------


def _fuxi_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.layouts import FUXI_LAYOUT

    return _gridded_source(
        FUXI_LAYOUT, params, "fuxi", float(params.get("step_hours", 6.0))
    )


def _fuxi_ensemble_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.layouts import FUXI_ENS_LAYOUT

    return _gridded_source(
        FUXI_ENS_LAYOUT, params, "fuxi_ens", float(params.get("step_hours", 6.0))
    )


def _station_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.station_reader import (
        DiamondStationCatalog,
        DiamondStationReader,
        load_station_whitelist,
    )

    source_id = params.get("source_id", "diamond_station")
    root = _require_root(params, "station")
    whitelist = load_station_whitelist(
        params.get("station_whitelist") or params.get("station_list")
    )
    return SourceHandle(
        reader=DiamondStationReader(source_id=source_id, station_whitelist=whitelist),
        catalog=DiamondStationCatalog(root, station_whitelist=whitelist),
        config=params,
    )


def _babj_source(**params) -> SourceHandle:
    """BABJ 台风路径报文（diamond7）。一个文件 = 一号台风的整条路径。"""
    from xmetai_evaluation.io.babj_reader import BabjCatalog, BabjReader

    source_id = params.get("source_id", "babj")
    root = _require_root(params, "babj")
    return SourceHandle(
        reader=BabjReader(source_id=source_id),
        catalog=BabjCatalog(root),
        config=params,
    )


def _fengqing_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.layouts import FENGQING_LAYOUT

    return _gridded_source(FENGQING_LAYOUT, params, "fengqing")


def _phys_source(layout_name: str, default_source_id: str):
    """按 xu 报告口径（z500 = m²/s²、q = g/kg）读预报的通用工厂。

    与 ``_fuxi_source`` 等的差别只在那份布局声明，所以共用 ``_gridded_source``；
    ``step_hours`` 缺省时用布局自己声明的值。
    """

    def factory(**params) -> SourceHandle:
        from xmetai_evaluation.io.layouts import LAYOUTS

        override = params.get("step_hours")
        return _gridded_source(
            LAYOUTS[layout_name],
            params,
            default_source_id,
            float(override) if override is not None else None,
        )

    return factory


def _cra_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.layouts import CRA_LAYOUT

    return _gridded_source(CRA_LAYOUT, params, "cra")


def _era5_zarr_source(**params) -> SourceHandle:
    """ERA5 再分析实况（zarr）。

    与通用格点源的区别是数据形态：一个 store 装全部时次，不靠文件名发现，
    所以有自己的 Catalog；变量/单位/分组仍走 ``ERA5_ZARR_LAYOUT`` 声明。
    一个 run 可以给两个 store（``{"pl": ..., "sfc": ...}``）。
    """
    from xmetai_evaluation.io.era5_zarr_reader import Era5ZarrCatalog, Era5ZarrReader

    source_id = params.get("source_id", "era5_zarr")
    stores = params.get("stores") or {}
    if not stores:
        raise ConfigError("reader 'era5_zarr' 需要 stores（如 {'pl': ..., 'sfc': ...}）")
    return SourceHandle(
        reader=Era5ZarrReader(source_id=source_id, stores=stores),
        catalog=Era5ZarrCatalog(stores),
        config=params,
    )


def _climatology_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.climatology_reader import (
        ClimatologyCatalog,
        ClimatologyReader,
    )

    source_id = params.get("source_id", "climatology")
    root = _require_root(params, "climatology")
    engine = str(params.get("engine", "cfgrib"))
    return SourceHandle(
        reader=ClimatologyReader(source_id=source_id, engine=engine),
        catalog=ClimatologyCatalog(root),
        config=params,
    )


def _daily_climatology_source(**params) -> SourceHandle:
    """单文件日序气候态（与按 MMDDHH 找文件的 climatology 是两种索引方式）。"""
    from xmetai_evaluation.io.daily_climatology_reader import (
        DEFAULT_SMOOTH_DAYS,
        DailyClimatologyCatalog,
        DailyClimatologyReader,
    )

    source_id = params.get("source_id", "daily_climatology")
    root = _require_root(params, "daily_climatology")
    return SourceHandle(
        reader=DailyClimatologyReader(
            source_id=source_id,
            smooth_days=int(params.get("smooth_days", DEFAULT_SMOOTH_DAYS)),
            scales=params.get("scales"),
            units=params.get("units"),
        ),
        catalog=DailyClimatologyCatalog(root),
        config=params,
    )


def _ref_probability_source(**params) -> SourceHandle:
    from xmetai_evaluation.io.ref_probability_reader import RefProbabilityReader

    source_id = params.get("source_id", "ref_probability")
    root = _require_root(params, "ref_probability")
    return SourceHandle(
        reader=RefProbabilityReader(root_dir=root, source_id=source_id),
        catalog=None,
        config=params,
    )


# --------------------------------------------------------------------------
# Transform：只负责"怎么把两个数据集变成可比的"
# --------------------------------------------------------------------------


def _transform_grid_to_station(method: str = "bilinear", **params):
    from xmetai_evaluation.transforms.interpolation import GridToStationInterpolator

    return GridToStationInterpolator(method)


def _transform_time_window(window_hours: float, time_dim: str = "lead_time", **params):
    from xmetai_evaluation.transforms.temporal import TimeWindowAccumulator

    return TimeWindowAccumulator(window_hours=float(window_hours), time_dim=time_dim)


def _transform_ensemble_mean(member_dim: str = "member", **params):
    from xmetai_evaluation.transforms.regrid import EnsembleMeanTransform

    return EnsembleMeanTransform(member_dim=member_dim)


# --------------------------------------------------------------------------
# Metric：只负责"算什么差异"
# --------------------------------------------------------------------------


def _metric_rmse(**params):
    from xmetai_evaluation.metrics.rmse import RMSE

    return RMSE(params)


def _metric_bias(**params):
    from xmetai_evaluation.metrics.bias import Bias

    return Bias(params)


def _metric_track_error(**params):
    from xmetai_evaluation.metrics.track_error import TrackError

    return TrackError(params)


def _metric_track_error_ens(**params):
    from xmetai_evaluation.metrics.track_error import TrackErrorEns

    return TrackErrorEns(params)


def _metric_acc(climatology_path: Optional[str] = None, **params):
    from xmetai_evaluation.metrics.acc import ACC

    return ACC({"climatology_path": climatology_path, **params})


def _metric_acc_uncentered(climatology_path: Optional[str] = None, **params):
    from xmetai_evaluation.metrics.acc import ACC

    return ACC({"climatology_path": climatology_path, "centered": False, **params})


def _normalize_thresholds(thresholds: Optional[List[Any]]) -> List[Any]:
    """阈值统一成 ``[(等级名, 阈值), ...]``；纯数字自动命名 ``≥<值>``。"""
    normalized: List[Any] = []
    for item in thresholds or []:
        if isinstance(item, (tuple, list)):
            normalized.append((str(item[0]), float(item[1])))
        else:
            normalized.append((f"≥{float(item):g}", float(item)))
    return normalized


def _metric_ts_score(thresholds: Optional[List[Any]] = None, **params):
    from xmetai_evaluation.metrics.categorical import TSScore

    return TSScore(thresholds=_normalize_thresholds(thresholds), params=params)


def _metric_ensemble_probability(thresholds: Optional[List[Any]] = None, **params):
    from xmetai_evaluation.metrics.probabilistic import EnsembleProbabilityScore

    return EnsembleProbabilityScore(
        thresholds=_normalize_thresholds(thresholds), params=params
    )


def _metric_crps(**params):
    from xmetai_evaluation.metrics.ensemble import CRPS

    return CRPS(params)


def _metric_spread_error(ddof: int = 1, **params):
    from xmetai_evaluation.metrics.ensemble import SpreadError

    return SpreadError(ddof=int(ddof), params=params)


def _metric_fss(thresholds: Optional[List[Any]] = None, windows=None, **params):
    from xmetai_evaluation.metrics.spatial import DEFAULT_WINDOWS, FractionsSkillScore

    return FractionsSkillScore(
        thresholds=_normalize_thresholds(thresholds),
        windows=windows or DEFAULT_WINDOWS,
        params=params,
    )


def _metric_activity(**params):
    from xmetai_evaluation.metrics.specialized import ActivityRatio

    return ActivityRatio(params)


def _metric_spectrum(max_wavenumber: int = 30, lat_order=None, **params):
    from xmetai_evaluation.metrics.specialized import PowerSpectrum

    return PowerSpectrum(
        max_wavenumber=int(max_wavenumber), lat_order=lat_order, params=params
    )


def _metric_zonal_spectrum(max_wavenumber: int = 30, **params):
    from xmetai_evaluation.metrics.specialized import ZonalSpectrum

    return ZonalSpectrum(max_wavenumber=int(max_wavenumber), params=params)


def _metric_spherical_bands(bands=None, block: int = 16, **params):
    from xmetai_evaluation.metrics.specialized import SphericalBands

    return SphericalBands(bands=bands, block=int(block), params=params)


# --------------------------------------------------------------------------
# Writer：只负责"长表之外的派生视图"
# --------------------------------------------------------------------------


def _writer_csv_long(tables, context, output_dir: Path) -> Path:
    """统一长表本身由 ResultStore 无条件写出，这里只为让配置能显式声明它。"""
    return Path(output_dir) / "scores.csv"


def _writer_coverage(tables, context, output_dir: Path) -> Path:
    path = Path(output_dir) / "coverage.csv"
    tables.coverage.to_csv(path, index=False, encoding="utf-8")
    return path


def _writer_details(tables, context, output_dir: Path) -> Path:
    """列联表计数等诊断量（长表形态）。"""
    path = Path(output_dir) / "diagnostics" / "scores_detail.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    tables.details.to_csv(path, index=False, encoding="utf-8")
    return path


def _writer_json(tables, context, output_dir: Path) -> Path:
    path = Path(output_dir) / "scores.json"
    records = json.loads(tables.score_table.to_json(orient="records", force_ascii=False))
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "run_id": context.run_id,
                "model_id": context.model_id,
                "dataset_id": context.dataset_id,
                "protocol_id": context.protocol_id,
                "scores": records,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
    return path


def _writer_categorical_wide(tables, context, output_dir: Path) -> Path:
    from xmetai_evaluation.output.table import categorical_wide

    path = Path(output_dir) / "diagnostics" / "categorical_wide.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    categorical_wide(tables).to_csv(path, index=False, encoding="utf-8")
    return path


def _writer_probability_wide(tables, context, output_dir: Path) -> Path:
    from xmetai_evaluation.output.table import probability_wide

    path = Path(output_dir) / "diagnostics" / "probability_wide.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    probability_wide(tables).to_csv(path, index=False, encoding="utf-8")
    return path


#: 赤道周长（km）。波长 = 赤道周长 / 纬向波数，与参考实现同一个常数。
_EARTH_CIRCUMFERENCE_KM = 40075.0


def _spectrum_number(value) -> str:
    """按参考实现的 ``%.6g`` 输出，便于跟已有归档逐格对比。"""
    return f"{float(value):.6g}"


def _spectrum_curve_mean(entries: List[Any]):
    """按样本数加权平均若干条功率谱曲线，返回 ``(pred, obs)`` 两条一维数组。

    每条曲线本来就是「该结果自身样本的平均谱」，权重取该结果的 ``n_requested``
    才等价于把所有样本一起平均（各结果的样本数不一定相同）。
    """
    import numpy as np

    size = min(
        min(
            np.asarray(curve["power_forecast"]).size,
            np.asarray(curve["power_observation"]).size,
        )
        for _, curve in entries
    )
    pred_total = np.zeros(size, dtype="f8")
    observation_total = np.zeros(size, dtype="f8")
    weight_total = 0
    for weight, curve in entries:
        keep = weight if weight > 0 else 1
        pred_total += np.asarray(curve["power_forecast"], dtype="f8")[:size] * keep
        observation_total += (
            np.asarray(curve["power_observation"], dtype="f8")[:size] * keep
        )
        weight_total += keep
    if weight_total == 0:
        return None, None
    return pred_total / weight_total, observation_total / weight_total


def _writer_spectrum(tables, context, output_dir: Path) -> Path:
    """逐波数功率谱：全体样本均值曲线 + 逐起报曲线。

    对标参考实现的 ``det_summary_spectrum_{var}.csv`` / ``spectrum_{init}_{var}.csv``：
    只出 k=1..720（k=0 去纬向均值后恒为 0，去掉），``wavelength_km = 40075 / k``。
    每个起报一条曲线，是其 60 个时效的平均。

    曲线不走 ``value``（那样每个波数都会展开成明细行），而是从 ``tables.curves``
    直接取原始数组。
    """
    grouped: Dict[Any, List[Any]] = {}
    for entry in tables.curves:
        curve = entry.get("curve") or {}
        if curve.get("kind") != "wavenumber_spectrum":
            continue
        base = entry.get("base") or {}
        key = (str(base.get("variable", "")), str(base.get("init_time", "")))
        grouped.setdefault(key, []).append((int(base.get("n_requested") or 0), curve))

    diagnostics = Path(output_dir) / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)

    per_variable: Dict[str, List[Any]] = {}
    by_init_rows: List[List[Any]] = []
    for (variable, init_time), entries in sorted(grouped.items()):
        per_variable.setdefault(variable, []).extend(entries)
        pred, obs = _spectrum_curve_mean(entries)
        if pred is None:
            continue
        for k in range(1, pred.size):
            by_init_rows.append(
                [
                    variable,
                    init_time,
                    k,
                    _spectrum_number(_EARTH_CIRCUMFERENCE_KM / k),
                    _spectrum_number(pred[k]),
                    _spectrum_number(obs[k]),
                ]
            )

    by_init_path = diagnostics / "spectrum_by_init.csv"
    with by_init_path.open("w", encoding="utf-8", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(
            ["variable", "init_time", "wavenumber", "wavelength_km", "pred", "obs"]
        )
        out.writerows(by_init_rows)

    summary_path: Optional[Path] = None
    for variable, entries in sorted(per_variable.items()):
        pred, obs = _spectrum_curve_mean(entries)
        if pred is None:
            continue
        path = diagnostics / f"spectrum_{variable}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            out = csv.writer(stream)
            out.writerow(["wavenumber", "wavelength_km", "pred_mean", "obs_mean"])
            for k in range(1, pred.size):
                out.writerow(
                    [
                        k,
                        _spectrum_number(_EARTH_CIRCUMFERENCE_KM / k),
                        _spectrum_number(pred[k]),
                        _spectrum_number(obs[k]),
                    ]
                )
        if summary_path is None:
            summary_path = path
    return summary_path or by_init_path


def _typhoon_number(value: Any) -> str:
    """逐场次 csv 的数值格式：6 位有效数字，非有限值写空。

    对齐旧归档 ``tc<编号>_<起报>.csv`` 的写法（``%.6g``）——那是给人看的表，
    6 位有效数字足够读；``str(float)`` 的 17 位尾数在这里只是噪音。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        # 非数值列（``valid_bjt`` 那种时刻字符串）原样透传：这里返回空会把
        # 整列时刻写没，而列还在，看着像"这条路径没有有效时刻"。
        return "" if value is None else str(value)
    if not math.isfinite(number):
        return ""
    return "%.6g" % number


def _typhoon_repr(value: Any) -> str:
    """拼接表 ``typhoon.csv`` 的数值格式：``str(float)`` 原样，非有限值写空。

    与逐场次表刻意不同：这张表是给下游脚本/渲染器做聚合的，取整到 6 位会
    让"同一场次在两份产物里对不上"，所以保留完整尾数。
    """
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return str(number)


def _writer_typhoon_cases(tables, context, output_dir: Path) -> Path:
    """台风逐场次路径表 + 拼接总表。

    对标旧归档 ``weather_typhoon_*/`` 的三个产物：

    - ``typhoon/tc<编号>_<起报>.csv``：一个场次一行一个时效，15 列，
      数值 6 位有效数字；对不上实况的时效留空；
    - ``typhoon/tc<编号>_<起报>_meta.json``：该场次的起报点、搜索参数、配对
      时效数等口径信息；
    - ``typhoon/typhoon.csv``：全部场次竖着拼起来，前面加
      ``tcid,tcname,init_utc`` 三列，数值取完整尾数。

    曲线从 ``tables.curves`` 取（``kind == "typhoon_track"``），不走 ``value``：
    15 字段 × 60 时效 × 全年场次会把长表撑爆，而曲线本来就有专门的出口。
    """
    from xmetai_evaluation.metrics.track_error import CURVE_FIELDS, ENS_EXTRA_FIELDS

    cases = [
        entry
        for entry in tables.curves
        if (entry.get("curve") or {}).get("kind") == "typhoon_track"
    ]
    cases_dir = Path(output_dir) / "typhoon"
    cases_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: List[List[Any]] = []
    combined_fields: List[str] = list(CURVE_FIELDS)
    written: List[Path] = []
    for entry in cases:
        curve = entry["curve"]
        # 集合曲线比确定性多 5 列（方案 A 的三项 + 逐时效成员数），**确定性那条
        # 路径一列不加**：加列会让已有归档的对拍整个错位。
        is_ens = str(curve.get("forecast_type") or "") == "ens"
        fields = list(CURVE_FIELDS) + (list(ENS_EXTRA_FIELDS) if is_ens else [])
        combined_fields = fields
        tcid = str(curve.get("storm") or "")
        tcname = str(curve.get("tcname") or "")
        # 曲线里的 init_utc 是 isoformat（带 T），旧归档写的是空格分隔
        init_utc = _iso_space(curve.get("init_utc"))
        leads = list(curve.get("lead_h") or [])
        # 文件名里的起报时刻用紧凑写法（2025061000），与旧归档一致
        stamp = _compact_init(init_utc)
        stem = f"tc{tcid}_{stamp}"

        path = cases_dir / f"{stem}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            # lineterminator 必须显式给：csv 模块默认 '\r\n'，旧归档是 '\n'。
            # 不写这一行，逐字节对拍时会看到"整个文件每一行都不同"。
            out = csv.writer(stream, lineterminator="\n")
            out.writerow(fields)
            for row_index in range(len(leads)):
                out.writerow(
                    [
                        _typhoon_number(curve.get(field, [None] * len(leads))[row_index])
                        for field in fields
                    ]
                )
        written.append(path)

        matched = [
            k for k in range(len(leads)) if _is_finite(_at(curve, "track_err_km", k))
        ]
        # init_pos 取**链的种子位置**（首个能配上时效的实况，通常为首个 +6h；
        # 种子取的时刻由下面的 ``seed`` 字段标注）。首时次 at/ct 的定向基准另有
        # 一条"起报时刻实况"（ob0，见 ``TrackErrorEns``）——旧 ens 归档把
        # init_pos 也写成 ob0，是已知的记录级差异，经确认暂不追平（2026-09-30）。
        meta = {
            "tcid": tcid,
            "tcname": tcname,
            "init_utc": init_utc,
            "init_bjt": _shift_iso(init_utc, curve.get("tz_shift", 8.0)),
            "init_pos": [curve.get("init_lat"), curve.get("init_lon")],
            # 起始中心（链的种子）取自起报后哪一条实况："init+6h" 是常态（第一个
            # 时效的步长）；更大说明那条分析场缺了，比如台风当天的第一条定位在
            # 20:00，就会是 "init+12h"。和别家归档对不上 init_pos 的场次先看这个
            # 字段——曲线可以完全一样，种子取的时刻不同。
            "seed": _seed_label(curve.get("seed_offset_h")),
            "forecast_type": "ens" if is_ens else "det",
            "lead_step": _lead_step(leads),
            "tz_shift_h": curve.get("tz_shift", 8.0),
            "search": curve.get("search") or {},
            "has_wind": any(
                _is_finite(value) for value in (curve.get("fcst_vmax_ms") or [])
            ),
            "n_leads": len(leads),
            "n_matched": len(matched),
        }
        if is_ens:
            # 集合口径：曲线里的 fcst_* 与 track_err_km / at_km / ct_km 是**方案 B**
            # （集合平均位置 vs 实况），*_a 三列是**方案 A**（成员误差平均）。
            # 强度类两项两方案代数恒等，只有一列。
            meta.update(
                {
                    "n_members": len(curve.get("members") or []),
                    "members": list(curve.get("members") or []),
                    "aggregation": {
                        "B": "集合平均位置 vs 实况"
                        "（fcst_* / track_err_km / at_km / ct_km）",
                        "A": "成员误差平均（track_err_km_a / at_km_a / ct_km_a）",
                    },
                }
            )
        with (cases_dir / f"{stem}_meta.json").open("w", encoding="utf-8") as stream:
            json.dump(meta, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        for row_index in range(len(leads)):
            combined_rows.append(
                [tcid, tcname, init_utc]
                + [
                    _typhoon_repr(curve.get(field, [None] * len(leads))[row_index])
                    for field in fields
                ]
            )

    combined = cases_dir / "typhoon.csv"
    with combined.open("w", encoding="utf-8", newline="") as stream:
        out = csv.writer(stream, lineterminator="\n")
        out.writerow(["tcid", "tcname", "init_utc", *combined_fields])
        out.writerows(combined_rows)
    return combined


def _seed_label(offset: Any) -> str:
    """台风 meta 的 ``seed`` 标签：起始中心取自起报后几小时的实况（"init+6h" 是常态）。"""
    try:
        hours = float(offset or 0.0)
    except (TypeError, ValueError):
        hours = 0.0
    return "init" if not hours else "init+%gh" % hours


def _at(curve: Dict[str, Any], field: str, index: int) -> Any:
    values = curve.get(field)
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return None
    return values[index]


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _iso_space(value: Any) -> str:
    """``2025-06-10T00:00:00`` -> ``2025-06-10 00:00:00``（旧归档的写法）。"""
    return str(value or "").replace("T", " ")


def _compact_init(init_utc: str) -> str:
    """``2025-06-10T00:00:00`` -> ``2025061000``（旧归档的文件名口径）。"""
    text = str(init_utc).replace("T", " ").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:10] if len(digits) >= 10 else text


def _shift_iso(init_utc: str, hours: float) -> str:
    from datetime import datetime, timedelta

    try:
        moment = datetime.fromisoformat(str(init_utc))
    except ValueError:
        return ""
    return str(moment + timedelta(hours=float(hours)))


def _lead_step(leads: List[Any]) -> Any:
    """相邻时效的间隔（单时次返回 None）。"""
    if len(leads) < 2:
        return None
    try:
        return float(leads[1]) - float(leads[0])
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Protocol：只负责"样本空间怎么遍历、怎么配对"
# --------------------------------------------------------------------------


def _protocol_station_valid_time(spec):
    from xmetai_evaluation.pipeline.protocols import StationValidTimeProtocol

    return StationValidTimeProtocol(spec)


def _protocol_grid_valid_time(spec):
    from xmetai_evaluation.pipeline.protocols import GridValidTimeProtocol

    return GridValidTimeProtocol(spec)


def _protocol_typhoon_track(spec):
    from xmetai_evaluation.pipeline.protocols import TyphoonTrackProtocol

    return TyphoonTrackProtocol(spec)


def _protocol_typhoon_track_ens(spec):
    from xmetai_evaluation.pipeline.protocols import TyphoonTrackEnsProtocol

    return TyphoonTrackEnsProtocol(spec)


_REGISTERED = False
_REGISTER_LOCK = threading.Lock()


def register_builtin_components() -> None:
    """注册全部内置组件（幂等、线程安全）。"""
    global _REGISTERED
    # 快路径无锁；未注册过的线程再进锁里双检。标记在注册**完成之后**才置位，
    # 保证别的线程看到 True 时全部组件都已可用。
    if _REGISTERED:
        return
    with _REGISTER_LOCK:
        if _REGISTERED:
            return
        _register_all()
        _REGISTERED = True


def _register_all() -> None:
    """一次性注册全部内置组件（只被 register_builtin_components 持锁调用）。"""
    register_reader("fuxi", "1.0.0", _fuxi_source, "FuXi 网格预报：root/YYYYMMDD/NNN.nc")
    register_reader(
        "fuxi_ens",
        "1.0.0",
        _fuxi_ensemble_source,
        "FuXi 集合预报：root/YYYYMMDD/member_*/NNN.nc",
    )
    register_reader("station", "1.0.0", _station_source, "Diamond 格式站点观测")
    register_reader(
        "diamond_station", "1.0.0", _station_source, "Diamond 格式站点观测（别名）"
    )
    register_reader(
        "babj", "1.0.0", _babj_source, "BABJ 台风路径报文（diamond7，一个文件一条路径）"
    )
    register_reader("fengqing", "1.0.0", _fengqing_source, "Fengqing 集合预报")
    register_reader(
        "fuxi_phys",
        "1.0.0",
        _phys_source("fuxi_phys", "fuxi"),
        "FuXi 网格预报（xu 报告口径：z500 = m²/s²、q = g/kg）",
    )
    register_reader(
        "fuxi_ens_phys",
        "1.0.0",
        _phys_source("fuxi_ens_phys", "fuxi_ens"),
        "FuXi 集合预报（xu 报告口径：z500 = m²/s²、q = g/kg）",
    )
    register_reader(
        "fengqing_phys",
        "1.0.0",
        _phys_source("fengqing_phys", "fengqing"),
        "风清单卡预报（xu 报告口径：z500 = m²/s²、q = g/kg）",
    )
    register_reader("cra", "1.0.0", _cra_source, "CRA40 再分析实况")
    register_reader(
        "era5_zarr",
        "1.0.0",
        _era5_zarr_source,
        "ERA5 再分析实况（zarr store，气压层/地面分 store）",
    )
    register_reader(
        "climatology",
        "1.0.0",
        _climatology_source,
        "CRA CLI_6HOUR 气候态参考场（按 月日+时次 索引）",
    )
    register_reader(
        "daily_climatology",
        "1.0.0",
        _daily_climatology_source,
        "日序气候态参考场（单文件，按年内日序索引，含环形平滑）",
    )
    register_reader(
        "ref_probability",
        "1.0.0",
        _ref_probability_source,
        "BSS 逐 6h 气候概率参考（ref/MMDDHH.000，站号+4阈值概率）",
    )

    register_transform(
        "grid_to_station", "1.0.0", _transform_grid_to_station, "网格场插值到站点"
    )
    register_transform(
        "time_window_accumulator",
        "1.0.0",
        _transform_time_window,
        "固定窗口时间累积（如 24h 降水）",
    )
    register_transform("ensemble_mean", "1.0.0", _transform_ensemble_mean, "集合均值（DataBundle -> DataBundle）")

    register_metric("rmse", "1.0.0", _metric_rmse, "均方根误差")
    register_metric("bias", "1.0.0", _metric_bias, "平均误差")
    register_metric(
        "track_error",
        "1.0.0",
        _metric_track_error,
        "台风路径误差：大圆距离 + 沿/横分解 + 强度偏差（配合 typhoon_track 协议）",
    )
    register_metric(
        "track_error_ens",
        "1.0.0",
        _metric_track_error_ens,
        "集合台风的路径误差：方案 B（平均位置）与方案 A（成员误差平均）两种口径"
        "（配合 typhoon_track_ens 协议）",
    )
    register_metric("acc", "1.0.0", _metric_acc, "距平相关系数")
    register_metric(
        "acc_uncentered",
        "1.0.0",
        _metric_acc_uncentered,
        "距平相关系数（uncentered，FDP/WeatherBench2 口径）",
    )
    register_metric("ts_score", "1.0.0", _metric_ts_score, "分类检验 TS/POD/FAR/BIAS")
    register_metric(
        "ensemble_probability",
        "1.0.0",
        _metric_ensemble_probability,
        "集合概率评分 AROC/BS/BSS（需要原始成员）",
    )
    register_metric("crps", "1.0.0", _metric_crps, "集合 CRPS（纬度加权闭式解）")
    register_metric(
        "spread_error", "1.0.0", _metric_spread_error,
        "集合离散度-误差比（含 Spread 与 RMSE；ddof=0 vfc 口径 / 1 FDP 口径）",
    )
    register_metric(
        "fss", "1.0.0", _metric_fss, "邻域分数技巧评分（空间检验，需网格场）"
    )
    register_metric(
        "activity", "1.0.0", _metric_activity, "活跃度比（距平加权标准差之比，需气候态）"
    )
    register_metric(
        "spectrum", "1.0.0", _metric_spectrum, "功率谱（二维 FFT，对原始场，0-max_k 波）"
    )
    register_metric(
        "zonal_spectrum",
        "1.0.0",
        _metric_zonal_spectrum,
        "纬向功率谱（去纬向均值 rfft，cos 纬度加权）",
    )
    register_metric(
        "spherical_bands",
        "1.0.0",
        _metric_spherical_bands,
        "球谐带功率（总波数分带，仅全球含极网格；bands 配带边界）",
    )

    register_protocol(
        "station_valid_time",
        "1.0.0",
        _protocol_station_valid_time,
        "格点预报插值到站点，按有效时刻配对；样本键为 lead_h",
    )
    register_protocol(
        "grid_valid_time",
        "1.0.0",
        _protocol_grid_valid_time,
        "格点预报插值到实况网格，按 valid_time 配对；样本键为 valid_time",
    )
    register_protocol(
        "typhoon_track",
        "1.0.0",
        _protocol_typhoon_track,
        "台风路径：从实况位置链式诊断预报中心，对 BABJ 报文配对；样本键为 storm",
    )
    register_protocol(
        "typhoon_track_ens",
        "1.0.0",
        _protocol_typhoon_track_ens,
        "集合台风路径：逐成员链式诊断（成员串行读，峰值内存与确定性链同级），"
        "批次带成员轴；样本键为 storm",
    )

    register_writer("csv_long", "1.0.0", lambda **kw: _writer_csv_long, "统一长表 scores.csv")
    register_writer(
        "coverage", "1.0.0", lambda **kw: _writer_coverage, "请求/有效样本覆盖（可由长表派生）"
    )
    register_writer(
        "details",
        "1.0.0",
        lambda **kw: _writer_details,
        "列联表计数等诊断明细（可由宽表替代）",
    )
    register_writer("json", "1.0.0", lambda **kw: _writer_json, "评分 JSON 快照")
    register_writer(
        "categorical_wide", "1.0.0", lambda **kw: _writer_categorical_wide, "分类检验宽表"
    )
    register_writer(
        "probability_wide",
        "1.0.0",
        lambda **kw: _writer_probability_wide,
        "集合概率评分宽表（AROC/BS/BSS）",
    )
    register_writer(
        "spectrum",
        "1.0.0",
        lambda **kw: _writer_spectrum,
        "逐波数功率谱：全体均值曲线 + 逐起报曲线",
    )
    register_writer(
        "typhoon_cases",
        "1.0.0",
        lambda **kw: _writer_typhoon_cases,
        "台风逐场次路径表 + 拼接总表（tc<编号>_<起报>.csv / typhoon.csv）",
    )


def available_components() -> Dict[str, List[str]]:
    """列出各类已注册组件的短名称（用于 --list 与错误提示）。"""
    from xmetai_evaluation.core.registry import ComponentType

    register_builtin_components()
    return {kind.value: get_registry().names(kind) for kind in ComponentType}
