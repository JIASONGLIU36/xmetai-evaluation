# -*- coding: utf-8 -*-
"""台风路径/强度误差指标。

输入是 ``typhoon_track`` 协议配对好的两条路径（同为 ``(lead_time, storm)``）：

- ``forecast``：诊断出来的预报中心（lat/lon/pmin/vmax）；
- ``observation``：配对上的实况中心（同上四项；对不上的时效为 NaN）。

**协议只管配对，误差在这里算**：大圆距离、沿/横路径分解、强度偏差。逐时效
走标量循环（与本模块的 ``match_errors`` 同一套调用），不向量化——归档对拍
要的是逐位一致，不是"数学上等价"。

沿/横路径误差（``at_km`` / ``ct_km``）的定向用**前一个配对上的实况位置**，
不是预报-实况连线，也不是起报点；所以确定性链里**第一个配对时效**的 at/ct
为空——没有前一个实况就无从定向。（集合链例外：首时次用**起报时刻实况**
定向，那是旧集合链路的归档口径，见 ``TrackErrorEns``。）实况有缺口时缺的是
那个时效，不是固定 6h。

除标量外，本指标把整条逐时效表放进 ``MetricResult.curve``：一个场次一行一个
时效，15 个字段，直接喂给 ``typhoon_cases`` writer 出 ``tc<编号>_<起报>.csv``。
不走 ``value`` 是因为 value 里每个键都会展开成长表明细行，15 字段 × 60 时效
× 全年场次会把长表撑成几十万行，而曲线本来就有专门的出口。
"""

import datetime as _dt
import math
from typing import Any, Dict, List, Sequence

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, MetricResult, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.base import (
    Metric,
    MetricRequirements,
    MetricState,
    ProductType,
)

#: 大圆距离用的地球半径（km），与旧链路一致。
R_EARTH_KM = 6371.0


def great_circle_km(lat1, lon1, lat2, lon2):
    """球面大圆距离（km，r=6371）。向量化；近距用 haversine 保精度。"""
    lat1, lon1, lat2, lon2 = (
        np.deg2rad(np.asarray(x, dtype="f8")) for x in (lat1, lon1, lat2, lon2)
    )
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def along_cross_track(lat0, lon0, lat_b, lon_b, lat_p, lon_p):
    """沿路径(at)/横路径(ct)误差分解（参考脚本公式，km）。

    ``(lat0,lon0)`` 是**前一个配对实况**位置，``(lat_b,lon_b)`` 是当前实况、
    ``(lat_p,lon_p)`` 是当前预报——方向由实况路径段定，不是预报-实况连线。
    """
    def _alpha(la0, lo0, la, lo):
        dlat = (la - la0) / 180.0 * math.pi
        dlon = (lo - lo0) / 180.0 * math.pi
        return math.atan2(dlon * math.cos((la + la0) / 2 / 180 * math.pi), dlat)

    dalpha = _alpha(lat0, lon0, lat_p, lon_p) - _alpha(lat0, lon0, lat_b, lon_b)
    len_b = float(great_circle_km(lat0, lon0, lat_b, lon_b))
    len_p = float(great_circle_km(lat0, lon0, lat_p, lon_p))
    ct = len_p * math.sin(dalpha)
    at = len_p * math.cos(dalpha) - len_b
    return at, ct


def match_errors(
    track: Dict[str, List[float]],
    leads_hours: Sequence[float],
    init_utc: _dt.datetime,
    babj: Dict[_dt.datetime, Dict[str, float]],
    tz_shift: float = 8.0,
) -> List[Dict[str, float]]:
    """诊断路径对 babj 实况逐时效配对并计算误差（旧链路兼容入口）。

    ``track`` 是 ``diagnose_track`` 的输出；``babj`` 是
    ``read_babj_analyses`` 的字典（北京时）。验证时刻(北京时) =
    ``init_utc + 时效 + tz_shift``。无实况记录的时效各项为 NaN（路径/强度
    误差保持 NaN，不外推）。

    首个**配对成功**的时效 ``at_km``/``ct_km`` 为空——没有前一个实况位置就
    无从定向；实况有缺口时，缺的是那个时效，不是固定 ``lead_h=6``。

    生产链路由协议先做 ``pair_observations``、指标再从批次算误差；这个函数
    保留给旧归档对拍与回归测试，调用顺序与旧实现一致。
    """
    prev_babj = None
    rows: List[Dict[str, float]] = []
    for k, tau in enumerate(leads_hours):
        v_bjt = init_utc + _dt.timedelta(hours=float(tau) + tz_shift)
        ob = babj.get(v_bjt)
        f_lat, f_lon = track["lat"][k], track["lon"][k]
        f_p, f_v = track["pmin_hpa"][k], track["vmax"][k]
        if ob is None:
            rows.append(
                {
                    "lead_h": float(tau),
                    "valid_bjt": str(v_bjt),
                    "fcst_lat": f_lat,
                    "fcst_lon": f_lon,
                    "fcst_pmin_hpa": f_p,
                    "fcst_vmax_ms": f_v,
                    "obs_lat": np.nan,
                    "obs_lon": np.nan,
                    "obs_pmin_hpa": np.nan,
                    "obs_vmax_ms": np.nan,
                    "track_err_km": np.nan,
                    "at_km": np.nan,
                    "ct_km": np.nan,
                    "wind_err_ms": np.nan,
                    "pmin_err_hpa": np.nan,
                }
            )
            continue
        o_lat, o_lon = ob["lat"], ob["lon"]
        err = float(great_circle_km(f_lat, f_lon, o_lat, o_lon))
        if prev_babj is None:  # 首个配对时次
            at = ct = float("nan")
        else:
            at, ct = along_cross_track(prev_babj[0], prev_babj[1], o_lat, o_lon, f_lat, f_lon)
        rows.append(
            {
                "lead_h": float(tau),
                "valid_bjt": str(v_bjt),
                "fcst_lat": f_lat,
                "fcst_lon": f_lon,
                "fcst_pmin_hpa": f_p,
                "fcst_vmax_ms": f_v,
                "obs_lat": o_lat,
                "obs_lon": o_lon,
                "obs_pmin_hpa": ob["pmin_hpa"],
                "obs_vmax_ms": ob["vmax_ms"],
                "track_err_km": err,
                "at_km": at,
                "ct_km": ct,
                "wind_err_ms": f_v - ob["vmax_ms"] if np.isfinite(f_v) else float("nan"),
                "pmin_err_hpa": f_p - ob["pmin_hpa"],
            }
        )
        prev_babj = (o_lat, o_lon)
    return rows


#: 逐时效表里的字段（顺序即写盘顺序，与旧归档的 csv 列一致）
CURVE_FIELDS = (
    "lead_h",
    "valid_bjt",
    "fcst_lat",
    "fcst_lon",
    "fcst_pmin_hpa",
    "fcst_vmax_ms",
    "obs_lat",
    "obs_lon",
    "obs_pmin_hpa",
    "obs_vmax_ms",
    "track_err_km",
    "at_km",
    "ct_km",
    "wind_err_ms",
    "pmin_err_hpa",
)

#: 落进长表的场次级标量（"value 键 -> 中文说明"，键名同时是 curve 里的字段名）
SUMMARY_FIELDS = ("track_err_km", "at_km", "ct_km", "wind_err_ms", "pmin_err_hpa")

#: 集合曲线在确定性 15 列之外多出来的列（顺序即写盘顺序）
ENS_EXTRA_FIELDS = (
    "n_members",
    "n_valid_members",
    "track_err_km_a",
    "at_km_a",
    "ct_km_a",
)

#: 集合口径落进长表的场次级标量。强度类（wind/pmin）两方案代数恒等，只出一行；
#: 路径类三项方案 A / B 各出一行，键名里的 ``_a`` 就是口径标记。
ENS_SUMMARY_FIELDS = (
    "track_err_km",
    "at_km",
    "ct_km",
    "wind_err_ms",
    "pmin_err_hpa",
    "track_err_km_a",
    "at_km_a",
    "ct_km_a",
)


def _nanmean(values: np.ndarray) -> float:
    """全 NaN 时返回 NaN，不报警告（全 NaN 是"这条路径一个时效都没配上"）。"""
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.mean(values[finite]))


class TrackError(Metric):
    """路径误差（大圆距离）+ 沿/横路径分解 + 强度偏差。"""

    #: ``merge`` 时要跟着 lead 一起重排的列。集合那条链的曲线比确定性多几列
    #: （方案 A 与逐时效成员数），不一起重排的话那几列会留在第一个状态的顺序上
    #: ——看着有值，其实已经错位了。
    MERGE_FIELDS = CURVE_FIELDS

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="track_error", version="1.0.0", params=params or {})

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
        )

    def validate(self, batch: EvaluationBatch) -> None:
        super().validate(batch)
        storm = self._storm_axis(batch)
        if storm.size != 1:
            raise MetricError(
                f"track_error 的一个批次只承载一号台风的一条路径，"
                f"这一批有 {storm.size} 号（{list(storm)}）——一个场次一个样本是 "
                f"typhoon_track 协议的约定，多条路径塞进一批会让逐场次曲线丢掉。",
                variable=self.name,
            )

    # ------------------------------------------------------------ 计算

    @staticmethod
    def _storm_axis(batch: EvaluationBatch):
        data = batch.forecast.payload if hasattr(batch.forecast, "payload") else batch.forecast
        return np.asarray(data["storm"].values)

    @staticmethod
    def _side(batch: EvaluationBatch, which: str) -> xr.Dataset:
        data = getattr(batch, which)
        return data.payload if hasattr(data, "payload") else data

    def _session_context(self, batch: EvaluationBatch) -> Dict[str, Any]:
        """从批次里取这次场次的标识与对时参数（写 curve 用）。"""
        key = dict(batch.sample_keys[0]) if batch.sample_keys else {}
        alignment = dict(batch.alignment or {})
        origin = alignment.get("init_position") or [None, None]
        return {
            "storm": str(self._storm_axis(batch)[0]),
            "tcname": str(alignment.get("tcname") or ""),
            "init_utc": str(key.get("init") or ""),
            "tz_shift": float(alignment.get("tz_shift_hours", 8.0)),
            # 起报点实况位置与搜索参数带着走：typhoon_cases writer 要把它们写进
            # 逐场次的 meta（起报点是画路径图的起点，搜索参数是复现结果的口径）。
            "init_lat": origin[0],
            "init_lon": origin[1],
            # 起始中心相对起报时刻的偏移小时（正常 6，见 babj_seed_moment）。
            # 由协议给的，只写进 meta，不参与计算。
            "seed_offset_h": float(alignment.get("seed_offset_hours", 0.0)),
            "search": dict(alignment.get("search") or {}),
        }

    def _compute(self, batch: EvaluationBatch) -> Dict[str, Any]:
        """逐时效算误差，返回 curve 形态的整张表。"""
        forecast = self._side(batch, "forecast")
        observation = self._side(batch, "observation")

        leads = np.asarray(forecast["lead_time"].values, dtype="f8")
        f_lat = np.asarray(forecast["lat"].values, dtype="f8")[:, 0]
        f_lon = np.asarray(forecast["lon"].values, dtype="f8")[:, 0]
        f_pmin = np.asarray(forecast["pmin"].values, dtype="f8")[:, 0]
        f_vmax = np.asarray(forecast["vmax"].values, dtype="f8")[:, 0]
        o_lat = np.asarray(observation["lat"].values, dtype="f8")[:, 0]
        o_lon = np.asarray(observation["lon"].values, dtype="f8")[:, 0]
        o_pmin = np.asarray(observation["pmin"].values, dtype="f8")[:, 0]
        o_vmax = np.asarray(observation["vmax"].values, dtype="f8")[:, 0]

        nan = float("nan")
        err = np.full(leads.size, nan)
        at = np.full(leads.size, nan)
        ct = np.full(leads.size, nan)
        wind = np.full(leads.size, nan)
        pmin = np.full(leads.size, nan)

        # 逐时效标量推进：与旧链路的 match_errors 同一套调用与顺序，
        # prev 只在**配对成功**的时效上更新，实况缺口不携带陈旧方向。
        prev = None
        for k in range(leads.size):
            if not np.isfinite(o_lat[k]):
                continue
            err[k] = float(great_circle_km(f_lat[k], f_lon[k], o_lat[k], o_lon[k]))
            if prev is not None:
                at[k], ct[k] = along_cross_track(
                    prev[0], prev[1], o_lat[k], o_lon[k], f_lat[k], f_lon[k]
                )
            if np.isfinite(f_vmax[k]):
                wind[k] = float(f_vmax[k] - o_vmax[k])
            pmin[k] = float(f_pmin[k] - o_pmin[k])
            prev = (o_lat[k], o_lon[k])

        context = self._session_context(batch)
        init = context["init_utc"]
        base = _parse_iso(init)
        valid = [
            str(base + _hours(float(lead) + context["tz_shift"])) if base else ""
            for lead in leads
        ]
        return {
            "kind": "typhoon_track",
            **context,
            "lead_h": [float(value) for value in leads],
            "valid_bjt": valid,
            "fcst_lat": f_lat.tolist(),
            "fcst_lon": f_lon.tolist(),
            "fcst_pmin_hpa": f_pmin.tolist(),
            "fcst_vmax_ms": f_vmax.tolist(),
            "obs_lat": o_lat.tolist(),
            "obs_lon": o_lon.tolist(),
            "obs_pmin_hpa": o_pmin.tolist(),
            "obs_vmax_ms": o_vmax.tolist(),
            "track_err_km": err.tolist(),
            "at_km": at.tolist(),
            "ct_km": ct.tolist(),
            "wind_err_ms": wind.tolist(),
            "pmin_err_hpa": pmin.tolist(),
            "n_leads": int(leads.size),
            "n_matched": int(np.isfinite(err).sum()),
        }

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        self.validate(batch)
        curve = self._compute(batch)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={"curve": curve, "n_valid": curve["n_matched"]},
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        """把同一场次的状态并成一条曲线。

        协议保证一个场次落在一个工作块里，所以正常情况这里只收到一个状态；
        真并起来时按 lead 拼接——**重复 lead 直接报错**，那是同一条曲线被算了
        两遍，拼出来的数看着正常但已经错了。
        """
        if not states:
            raise MetricError("Cannot merge empty states list")
        first = states[0]
        for state in states[1:]:
            if state.metric_name != first.metric_name or state.metric_version != first.metric_version:
                raise MetricError(
                    f"Cannot merge states from different metrics/versions: "
                    f"{state.metric_name}@{state.metric_version} vs "
                    f"{first.metric_name}@{first.metric_version}"
                )
        if len(states) == 1:
            return first

        curves = [state.data["curve"] for state in states]
        leads: List[float] = []
        for curve in curves:
            leads.extend(curve["lead_h"])
        if len(set(leads)) != len(leads):
            raise MetricError(
                f"台风场次 {curves[0].get('storm')}@{curves[0].get('init_utc')} 的曲线"
                f"在合并时出现重复时效——同一条路径被算了不止一遍，结果不可用。"
            )
        order = sorted(range(len(leads)), key=lambda index: leads[index])
        merged = dict(curves[0])
        for field in self.MERGE_FIELDS:
            if field in ("lead_h", "valid_bjt"):
                continue
            if field not in merged:
                continue
            values: List[Any] = []
            for curve in curves:
                values.extend(curve[field])
            merged[field] = [values[index] for index in order]
        merged["lead_h"] = [leads[index] for index in order]
        merged["valid_bjt"] = [
            curve["valid_bjt"][k] for curve in curves for k in range(len(curve["lead_h"]))
        ]
        merged["valid_bjt"] = [merged["valid_bjt"][index] for index in order]
        merged["n_leads"] = len(merged["lead_h"])
        merged["n_matched"] = int(np.isfinite(np.asarray(merged["track_err_km"], dtype="f8")).sum())
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={"curve": merged, "n_valid": merged["n_matched"]},
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        curve = state.data["curve"]
        n_leads = int(curve["n_leads"])
        n_matched = int(curve["n_matched"])
        values = {
            field: _nanmean(np.asarray(curve[field], dtype="f8")) for field in SUMMARY_FIELDS
        }
        status = ResultStatus.SUCCESS if n_matched else ResultStatus.NO_VALID_DATA
        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=values,
            status=status,
            n_requested=n_leads,
            n_valid=n_matched,
            unit="km",
            aggregation="mean_over_leads",
            product_kind=self.PRODUCT_KIND,
            curve=curve,
            warnings=[] if n_matched else ["这条路径没有任何时效配上实况"],
        )


def _row_nanmean(values: np.ndarray) -> np.ndarray:
    """按行取均值（跳过 NaN）；整行全 NaN 时给 NaN，且不触发 RuntimeWarning。

    集合平均位置对"这个时效能诊断出中心的成员"求平均，与旧链路
    （``g["fcst_lat"].mean()``）同口径：成员的链在个别时效搜不到中心是正常的，
    那片 NaN 不该把整个时效的集合平均也变成 NaN。
    """
    values = np.asarray(values, dtype="f8")
    if values.ndim == 1:
        values = values[:, None]
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    totals = np.where(finite, values, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = totals / counts
    return np.where(counts > 0, means, np.nan)


class TrackErrorEns(TrackError):
    """集合台风的路径与强度误差：方案 A / B 两种口径一次算清。

    与 ``TrackError`` 有两处差别：

    - **批次带成员轴时怎么算**：曲线的 15 个确定性列里，``fcst_*`` 给的是
      **集合平均位置**，所以 ``track_err_km`` / ``at_km`` / ``ct_km`` 就是
      **方案 B**；方案 A 的三项另开 ``*_a`` 列（逐成员先算误差、再对成员平均）；
      强度类两项两方案代数恒等，只出一列——
      ``mean_m(pmin_m − obs) ≡ mean_m(pmin_m) − obs``，实况对成员是常数。
    - **首时次 at/ct 的定向基准**：集合链按旧链路（``run_tc`` 的集合分支 /
      ``recompute_ens_track_error.py``）的口径，用**起报时刻实况**当"前一个
      位置"（协议放在 alignment 的 ``init_obs_position``）；起报时刻没有实况时
      首时次才留空。确定性链没有这个基准、首时次一贯为空——两条链各自对齐自己
      的归档，别按其中一条的直觉改另一条。

    逐时效仍走标量循环；方案 A 只是在成员维上多一层循环，**每个成员算误差的调用
    顺序与确定性链完全一致**——对拍要的是逐位一致，不是"数学上等价"。

    集合平均位置用朴素算术平均（不处理环绕）——与旧链路
    ``recompute_ens_track_error.py`` 一致；一条路径横跨换日线的情况不存在。
    """

    MERGE_FIELDS = CURVE_FIELDS + ENS_EXTRA_FIELDS

    #: 结果产品类型（长表 product_kind 列）：与确定性链分开。两条链的指标名不同
    #: （track_error / track_error_ens），产品身份同样是集合的——两份归档拼在
    #: 一起时别让下游按确定性口径取用。
    PRODUCT_KIND = "ensemble"

    def requirements(self) -> MetricRequirements:
        """集合输入契约：成员轴是这条链的正常输入，声明 ``ENSEMBLE_SAMPLES``。

        ``TrackError`` 的 ``DETERMINISTIC_FIELD`` 会被基类校验拦下（"要求确定性
        场，但输入含 member 维"）——那道检查拦的是"确定性指标收到集合输入"，
        正是这条链的对立面。成员级诊断场就在 forecast 的
        ``(lead_time, storm, member)`` 轴上，不另设 ``members`` 槽位（基类校验
        已同时认这两种入口，见 ``metrics/base.py`` 的同款注释）。
        """
        return MetricRequirements(
            product_type=ProductType.ENSEMBLE_SAMPLES,
            variables=["*"],
        )

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        # TrackError.__init__ 把名字定成了 "track_error"，这里改回集合的这个
        self.name = "track_error_ens"

    def validate(self, batch: EvaluationBatch) -> None:
        super().validate(batch)
        data = self._side(batch, "forecast")
        if "member" not in data.dims:
            raise MetricError(
                f"track_error_ens 要的是带成员轴的批次（预报 (lead_time, storm, "
                f"member)），这一批的维是 {tuple(data.dims)}。确定性链路请用 "
                f"track_error。",
                variable=self.name,
            )

    def _compute(self, batch: EvaluationBatch) -> Dict[str, Any]:
        forecast = self._side(batch, "forecast")
        observation = self._side(batch, "observation")

        leads = np.asarray(forecast["lead_time"].values, dtype="f8")
        members = [str(value) for value in np.asarray(forecast["member"].values)]
        # (lead, member)：预报侧保留着成员轴，这里整条取出来
        f_lat = np.asarray(forecast["lat"].values, dtype="f8")[:, 0, :]
        f_lon = np.asarray(forecast["lon"].values, dtype="f8")[:, 0, :]
        f_pmin = np.asarray(forecast["pmin"].values, dtype="f8")[:, 0, :]
        f_vmax = np.asarray(forecast["vmax"].values, dtype="f8")[:, 0, :]
        # (lead,)：实况对成员是同一份
        o_lat = np.asarray(observation["lat"].values, dtype="f8")[:, 0]
        o_lon = np.asarray(observation["lon"].values, dtype="f8")[:, 0]
        o_pmin = np.asarray(observation["pmin"].values, dtype="f8")[:, 0]
        o_vmax = np.asarray(observation["vmax"].values, dtype="f8")[:, 0]

        m_lat = _row_nanmean(f_lat)
        m_lon = _row_nanmean(f_lon)
        m_pmin = _row_nanmean(f_pmin)
        m_vmax = _row_nanmean(f_vmax)

        n_lead = leads.size
        n_member = len(members)
        nan = float("nan")
        err_b = np.full(n_lead, nan)
        at_b = np.full(n_lead, nan)
        ct_b = np.full(n_lead, nan)
        err_a = np.full(n_lead, nan)
        at_a = np.full(n_lead, nan)
        ct_a = np.full(n_lead, nan)
        wind = np.full(n_lead, nan)
        pmin = np.full(n_lead, nan)
        n_valid = np.zeros(n_lead, dtype=int)

        # 首时次 at/ct 的定向基准：**起报时刻实况**（旧集合链路口径，协议放在
        # alignment 的 init_obs_position）。它和种子不是同一条记录：种子是
        # "起报后第一条能配上时效的实况"，通常晚 6h 以上；起报时刻没有实况时
        # 基准为 None，首时次 at/ct 留空——与旧归档（run_tc 集合分支 /
        # recompute_ens_track_error.py）一致。
        alignment = dict(batch.alignment or {})
        init_obs = alignment.get("init_obs_position")
        initial = (float(init_obs[0]), float(init_obs[1])) if init_obs else None

        # 逐时效标量推进：prev 只在**配对成功**的时效上更新；首时次用起报时刻
        # 实况当"前一个位置"定向，其后与确定性链同款。
        prev = None
        for k in range(n_lead):
            if not np.isfinite(o_lat[k]):
                continue
            anchor = prev if prev is not None else initial
            # 方案 B：集合平均位置 vs 实况
            err_b[k] = float(great_circle_km(m_lat[k], m_lon[k], o_lat[k], o_lon[k]))
            if anchor is not None:
                at_b[k], ct_b[k] = along_cross_track(
                    anchor[0], anchor[1], o_lat[k], o_lon[k], m_lat[k], m_lon[k]
                )
            # 方案 A：逐成员算误差，再对"能诊断出中心"的成员平均
            member_err: List[float] = []
            member_at: List[float] = []
            member_ct: List[float] = []
            for j in range(n_member):
                if not np.isfinite(f_lat[k, j]):
                    continue
                member_err.append(
                    float(great_circle_km(f_lat[k, j], f_lon[k, j], o_lat[k], o_lon[k]))
                )
                if anchor is not None:
                    along, cross = along_cross_track(
                        anchor[0], anchor[1], o_lat[k], o_lon[k], f_lat[k, j], f_lon[k, j]
                    )
                    member_at.append(float(along))
                    member_ct.append(float(cross))
            if member_err:
                err_a[k] = float(np.mean(member_err))
            if member_at:
                at_a[k] = float(np.mean(member_at))
            if member_ct:
                ct_a[k] = float(np.mean(member_ct))
            n_valid[k] = len(member_err)
            # 强度类：两方案代数恒等（实况对成员是常数），但按**"成员误差再平均"**
            # 算，而不是"平均后再求差"——两者数学上相等，浮点上差最后几位，而旧
            # 链路是对逐成员的 pmin_err_hpa 取 .mean()（前者）。对拍要的是逐位一致。
            member_wind = [
                f_vmax[k, j] - o_vmax[k]
                for j in range(n_member)
                if np.isfinite(f_vmax[k, j])
            ]
            if member_wind:
                wind[k] = float(np.mean(member_wind))
            member_pmin = [
                f_pmin[k, j] - o_pmin[k]
                for j in range(n_member)
                if np.isfinite(f_pmin[k, j])
            ]
            if member_pmin:
                pmin[k] = float(np.mean(member_pmin))
            prev = (o_lat[k], o_lon[k])

        context = self._session_context(batch)
        base = _parse_iso(context["init_utc"])
        valid = [
            str(base + _hours(float(lead) + context["tz_shift"])) if base else ""
            for lead in leads
        ]
        return {
            "kind": "typhoon_track",
            **context,
            # 曲线级标记：writer 据它决定要不要多写集合那几列
            "forecast_type": "ens",
            "members": list(members),
            "lead_h": [float(value) for value in leads],
            "valid_bjt": valid,
            "fcst_lat": m_lat.tolist(),
            "fcst_lon": m_lon.tolist(),
            "fcst_pmin_hpa": m_pmin.tolist(),
            "fcst_vmax_ms": m_vmax.tolist(),
            "obs_lat": o_lat.tolist(),
            "obs_lon": o_lon.tolist(),
            "obs_pmin_hpa": o_pmin.tolist(),
            "obs_vmax_ms": o_vmax.tolist(),
            "track_err_km": err_b.tolist(),
            "at_km": at_b.tolist(),
            "ct_km": ct_b.tolist(),
            "wind_err_ms": wind.tolist(),
            "pmin_err_hpa": pmin.tolist(),
            # 逐时效的成员数：``n_members`` 是参与了这批的成员总数（每个时效都
            # 一样），``n_valid_members`` 是真的诊断出中心的那几个。旧链路的
            # ``_ensemble.csv`` 就是这么两列。
            "n_members": [n_member] * n_lead,
            "n_valid_members": [int(value) for value in n_valid],
            "track_err_km_a": err_a.tolist(),
            "at_km_a": at_a.tolist(),
            "ct_km_a": ct_a.tolist(),
            "n_leads": int(n_lead),
            "n_matched": int(np.isfinite(err_b).sum()),
        }

    def finalize(self, state: MetricState) -> MetricResult:
        curve = state.data["curve"]
        n_leads = int(curve["n_leads"])
        n_matched = int(curve["n_matched"])
        values = {
            field: _nanmean(np.asarray(curve[field], dtype="f8"))
            for field in ENS_SUMMARY_FIELDS
        }
        status = ResultStatus.SUCCESS if n_matched else ResultStatus.NO_VALID_DATA
        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=values,
            status=status,
            n_requested=n_leads,
            n_valid=n_matched,
            unit="km",
            aggregation="mean_over_leads",
            product_kind=self.PRODUCT_KIND,
            curve=curve,
            warnings=[] if n_matched else ["这条路径没有任何时效配上实况"],
        )


def _parse_iso(value: str):
    from datetime import datetime

    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _hours(value: float):
    from datetime import timedelta

    return timedelta(hours=float(value))
