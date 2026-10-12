#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性多模型评估报告：加载评估产物 → 统一口径统计 → 渲染 Markdown。

**输入是「评估产物目录」**，一个模型一个，形状由 ``output/store.py`` 的写入契约
决定（见 ``skills/xmetai-evaluation/SKILL.md`` 的「输出契约」）::

    <model_root>/
      scores.csv                   # 长表，列见 output/table.py 的 SCORE_COLUMNS
      manifest.json                # 运行记录：statuses / valid_times / regions /
                                   #   resolved_config.pipeline / execution
      diagnostics/
        spectrum_<变量>.csv        # 全体样本加权平均谱：wavenumber, pred_mean, obs_mean
        spectrum_by_init.csv       # 逐起报谱：variable, init_time, wavenumber, pred, obs

**长表怎么变回报告要的表**：``analyze()`` 以上全部沿用旧口径，只把取数换成从
``scores.csv`` 透视，所以下面这些等价关系要记住：

    老批次归档                          新产物
    summary.csv 的 rmse_<v>_mean        metric=rmse 的行按起报日对全时效平均
    <date>/rmse_<date>_det.csv          metric=rmse 的行（index=lead_h, columns=变量）
    <date>/acc_<date>_det.csv           metric=acc 的行
    <date>/fa_<date>_det.csv 的 ratio   metric=activity_ratio 的行
    <date>/spectrum_<date>_<变量>.csv   diagnostics/spectrum_by_init.csv
    batch_meta.json                     manifest.json（见 :func:`_meta_from_scores`）

「逐日文件」没有了，等价物是 ``init_date`` 这一列——由 ``init_time`` 前 10 位得来
（``2025-01-02T00:00:00.000000`` → ``20250102``）。

**全球口径**：同一 (起报, 时效, 指标) 下产物会同时写全球行和三块区域行
（``config_options.regions``），只有 ``region`` 为空的那行是全球。报告正文用全球行，
所有取数一律先过 :func:`_global`。

**确定性口径**：长表里 ``metric="rmse"`` 有两套——确定性 RMSE
（``product_kind=deterministic``）和集合 ``spread_error`` 展开出来的那个 rmse
（``product_kind=ensemble``）。名字一样，取数一律先过 :func:`_deterministic`。

**统一口径**：正文与附录图都用 N 个模型共同的起报日（原报告的正文 169 天 /
附录 349 天双口径不保留）。

**派生指标口径**（原报告没有生成脚本，这几个定义是反推的，写在这里备查）::

    相对 RMSE       逐变量 100 × v_m / geomean_模型(v)，再对共享变量等权平均
                    （几何均值基线 = 100；原报告 89.258/96.354/99.004/117.55 反推验证过）
    ACC (%)         共同日期 × 全 lead 的 ACC 变量平均
    FA 偏差 (pp)    100 × mean(|ratio − 1|)，对全部 FA 变量、全 lead 平均
    频谱对数 RMS×100 100 × mean_v sqrt(mean_k (log10(pred_k / obs_k))²)
    平均排名分      上述四个指标上名次的平均

**显著性**：逐日期配对 Wilcoxon 符号秩检验 + Holm 多重校正。
Holm 是手写的五行（``_holm``），不为它引入 statsmodels。
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from xmetai_evaluation.visualization.det_plots import DetPlotter, radar_scores

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except ImportError:  # 推迟到真正做检验时报错，让"只出图"也能跑
    _scipy_wilcoxon = None


#: 几何均值基线：相对 RMSE 的 100 就是这个意思
BASELINE = 100.0

#: ACC / 谱图优先用的变量，缺了就退到字典序第一个
PREFERRED_VARIABLE = "z500"

#: ACC 在首时效高于这个百分数就算"饱和"，写进 6.3 的风险提示
SATURATED_ACC = 99.0

#: 分时效段的边界（左开右闭，单位 h）；空段不输出，段名按段内实际 lead 现写
LEAD_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 120.0),
    (120.0, 240.0),
    (240.0, 360.0),
)

_CN_NUM = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}


# ====================================================================== 加载


#: 评估产物的固定文件名。新契约下**只认这一种**形状，不再有
#: ``summary.csv`` / ``batch_meta.json`` / ``<YYYYMMDD>/`` 那套批次归档。
SCORES_NAME = "scores.csv"
MANIFEST_NAME = "manifest.json"
DIAGNOSTICS_DIR = "diagnostics"

#: 逐起报谱表。``spectrum_<变量>.csv`` 是全体样本平均，只有这一张能给出
#: **每次起报各自**的谱，报告里按日期画/检验的谱曲线都取自它。
_SPECTRUM_BY_INIT_STEM = "spectrum_by_init"
SPECTRUM_BY_INIT_NAME = f"{_SPECTRUM_BY_INIT_STEM}.csv"

#: ``init_time`` 是 ISO 串（``2025-01-02T00:00:00.000000``），前 10 位即起报日。
_INIT_DATE_LEN = 10

#: 同一个产物目录在一个进程里可能被反复取数，长表又不小，所以缓存住。
_SCORES_CACHE: Dict[str, pd.DataFrame] = {}
_MANIFEST_CACHE: Dict[str, Dict[str, object]] = {}


class Archive(NamedTuple):
    """一个模型的评估产物。

    ``summary`` 与 ``meta`` 是为兼容渲染层保留的两个**派生视图**，新产物不落盘：

    ``summary``
        老 ``summary.csv`` 的形状——``init_date`` 一列，加每变量一列
        ``rmse_<v>_mean``，另带 ``variables`` / ``n_leads``。由 ``scores.csv``
        里 ``metric=rmse`` 的行按起报日、对全部时效平均现算。
    ``meta``
        老 ``batch_meta.json`` 的形状——``n_dates`` / ``n_ok`` / ``failures``。
        新产物没有「请求过的起报日数」这个概念，所以 ``n_dates`` 取**实际出了数的
        起报日数**，``n_ok`` 是其中全部行都 ``status == "success"`` 的那些。
    """

    name: str
    root: Path
    summary: pd.DataFrame
    meta: Dict[str, object]
    dates: List[str]  # scores.csv 里出现过的起报日（升序）


def _to_init_date(series: pd.Series) -> pd.Series:
    """``init_time`` / ``valid_time`` 的 ISO 串 → ``YYYYMMDD``。"""
    return series.astype(str).str.slice(0, _INIT_DATE_LEN).str.replace("-", "", regex=False)


def load_scores(root: Path) -> pd.DataFrame:
    """读 ``scores.csv`` 长表，附一列 ``init_date``。同一个目录只读一次。

    Raises:
        FileNotFoundError: 没有 ``scores.csv``——那不是一份评估产物目录。
        ValueError: 缺报告要用的列。
    """
    root = Path(root)
    key = str(root)
    cached = _SCORES_CACHE.get(key)
    if cached is not None:
        return cached
    path = root / SCORES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"缺 {SCORES_NAME}（{path}）：这不是一份评估产物目录")
    # low_memory=False：``unit`` / ``level`` / ``region`` 这些列是「字符串或空」，
    # 分块推断会在不同块里得出不同的 dtype（一串 DtypeWarning），读全再推断才一致。
    frame = pd.read_csv(path, low_memory=False)
    missing = [
        column for column in ("variable", "metric", "product_kind", "lead_h", "init_time", "value")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{path} 缺列 {missing}；现有列: {list(frame.columns)}")
    frame["lead_h"] = pd.to_numeric(frame["lead_h"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["init_date"] = _to_init_date(frame["init_time"])
    _SCORES_CACHE[key] = frame
    return frame


def load_manifest(root: Path) -> Dict[str, object]:
    """读 ``manifest.json``（运行记录），同一个目录只读一次。"""
    root = Path(root)
    key = str(root)
    cached = _MANIFEST_CACHE.get(key)
    if cached is not None:
        return cached
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"缺 {MANIFEST_NAME}（{path}）")
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    _MANIFEST_CACHE[key] = meta
    return meta


def _region_rows(scores: pd.DataFrame, region: Optional[str]) -> pd.DataFrame:
    """按纬度带取行：``region=None`` 只要全球行，给了带名只要**那一个带**的行。

    产物对同一 (起报, 时效, 指标) 会同时写全球行和 ``config_options.regions``
    里的区域行（``tropics`` 等），全球行的 ``region`` 是空串，读进来是 NaN。

    **两种口径不能混着平均**：全球平均和纬度带平均是两个量，把带行和全球行一起
    丢进 ``pivot_table`` 会让 ``aggfunc="mean"`` 悄悄把带间差异抹平，出来的数既不是
    全球的也不是任何一个带的，且没人看得出来。所以取数一律显式二选一。

    长表没有 ``region`` 列（这份产物不分区）时：要全球行就是全部行，
    要某个带就是空——**不静默退化成全球**。
    """
    if "region" not in scores.columns:
        return scores if region is None else scores.iloc[0:0]
    values = scores["region"]
    empty = values.isna() | (values.astype(str).str.strip() == "")
    if region is None:
        return scores[empty]
    return scores[~empty & (values.astype(str).str.strip() == str(region))]


def _global(scores: pd.DataFrame) -> pd.DataFrame:
    """只留全球口径的行（``region`` 为空）。见 :func:`_region_rows`。"""
    return _region_rows(scores, None)


#: ``manifest["regions"]`` 的条目形状：``"tropics[-20, 20]"``。
_REGION_ENTRY = re.compile(r"^(?P<name>.+?)\s*\[(?P<bounds>.+)\]$")


def region_names(root: Path) -> List[str]:
    """这份产物**真的落了数**的纬度带名（升序），取自 ``scores.csv`` 的 ``region`` 列。

    以数据为准而不是 manifest：manifest 里写着一堆 ``regions``、长表里却没有对应行
    （配置改了没重跑、或老产物）时，按数据来才不会端出一张空表。
    """
    scores = load_scores(root)
    if "region" not in scores.columns:
        return []
    values = scores["region"]
    filled = values[values.notna() & (values.astype(str).str.strip() != "")]
    return sorted({str(v).strip() for v in filled})


def region_labels(root: Path) -> Dict[str, str]:
    """``{带名: 边界串}``，边界取自 ``manifest["regions"]``（形如 ``tropics[-20, 20]``）。

    带名和边界都**照搬配置**——报告不翻译、也不写死任何带名。配置里叫
    ``mid_latitudes`` 就显示 ``mid_latitudes``；要换成「热带/中纬/高纬」那套切法，
    改配置的 ``options["regions"]`` 重跑即可，报告这边一个字都不用动。
    """
    labels: Dict[str, str] = {}
    for entry in load_manifest(root).get("regions") or []:
        match = _REGION_ENTRY.match(str(entry).strip())
        if match:
            labels[match.group("name").strip()] = match.group("bounds").strip()
    return labels


def region_display(name: str, labels: Mapping[str, str]) -> str:
    """带名的展示写法：``tropics [-20, 20]``；没有边界信息就只写带名。"""
    bounds = labels.get(name)
    return f"{name} [{bounds}]" if bounds else str(name)


def _deterministic(scores: pd.DataFrame) -> pd.DataFrame:
    """只留确定性指标的行。

    ``spread_error`` 展开出来的 ``rmse`` 与确定性 ``rmse`` **在长表里同名**，
    只能靠 ``product_kind``（``ensemble`` / ``deterministic``）区分。
    """
    if "product_kind" not in scores.columns:
        return scores
    return scores[scores["product_kind"].astype(str) == "deterministic"]


def metric_rows(
    root: Path,
    metric: str,
    *,
    product_kind: Optional[str] = None,
    region: Optional[str] = None,
) -> pd.DataFrame:
    """长表里某个指标的行；``product_kind`` 给了就再按它过滤。

    三个报告模块都从这里取数：确定性指标传 ``product_kind="deterministic"``，
    集合特有的（``crps`` / ``spread`` / ``spread_error_ratio``）传 ``"ensemble"``，
    ``activity`` 的四个展开量是 ``"specialized"``（不传就是不过滤）。

    ``region`` 省略即**全球口径**（只要 ``region`` 为空的行）；给了带名就只要那个带。
    两者互斥，没有「都要」的选项——见 :func:`_region_rows`。
    """
    rows = _region_rows(load_scores(root), region)
    rows = rows[rows["metric"] == str(metric)]
    if product_kind is not None and "product_kind" in rows.columns:
        rows = rows[rows["product_kind"].astype(str) == str(product_kind)]
    return rows


def by_date_frames(rows: pd.DataFrame, dates: Sequence[str], label: str = "") -> Dict[str, pd.DataFrame]:
    """长表行 → ``{起报日: lead × 变量 表}``。

    配对检验要**逐日起报**的样本（先按日平均再检验等于把样本量从 N 天压成 1 个点），
    所以这里保留逐日，不像 :func:`load_by_lead` 那样直接跨日平均。

    Raises:
        FileNotFoundError: 任何一个请求的起报日没有结果——按 SKILL.md 的硬性规则
            不静默跳过。
    """
    wanted = [str(date) for date in dates]
    have = set(rows["init_date"].astype(str))
    missing = [date for date in wanted if date not in have]
    if missing:
        raise FileNotFoundError(
            f"{label or '产物'}: {len(missing)} 个起报日没有结果（如 {missing[0]}）"
        )
    subset = rows[rows["init_date"].astype(str).isin(set(wanted))]
    frames: Dict[str, pd.DataFrame] = {}
    for date in wanted:
        own = subset[subset["init_date"].astype(str) == date]
        frames[date] = own.pivot_table(
            index="lead_h", columns="variable", values="value", aggfunc="mean"
        ).sort_index()
    return frames


def _summary_from_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """把长表透视成老 ``summary.csv`` 的形状。

    Raises:
        ValueError: 长表里没有 ``metric=rmse`` 的确定性行，分变量结果无从谈起。
    """
    rows = _deterministic(_global(scores))
    rows = rows[rows["metric"] == "rmse"]
    if rows.empty:
        raise ValueError("scores.csv 里没有确定性 rmse 的行，算不出分变量 RMSE")
    table = rows.pivot_table(index="init_date", columns="variable", values="value", aggfunc="mean")
    table = table.sort_index()
    table.columns = [f"rmse_{v}_mean" for v in table.columns]
    table = table.reset_index()
    # 老 summary 的 variables 列是「该模型算过的变量清单」，第 6.2 节拿它跟
    # --declared 对照。这里取长表里出现过的全部变量，不只 rmse 覆盖的那些。
    table["variables"] = ",".join(sorted({str(v) for v in scores["variable"].dropna().unique()}))
    table["n_leads"] = int(scores["lead_h"].dropna().nunique())
    return table


def _meta_from_scores(scores: pd.DataFrame) -> Dict[str, object]:
    """老 ``batch_meta.json`` 的等价物，喂给第 2 节的完整性检查。"""
    dates = sorted({str(d) for d in scores["init_date"].dropna().unique()})
    if "status" in scores.columns:
        ok = scores["status"].astype(str) == "success"
    else:
        ok = pd.Series(True, index=scores.index)
    failed = sorted({str(d) for d in scores.loc[~ok, "init_date"].dropna().unique()})
    return {
        "n_dates": len(dates),
        "n_ok": len(dates) - len(failed),
        "failures": failed,
        # 新产物没有「请求过的日期」这一说，留空让渲染层显式判空
        "dates": [],
    }


def load_archive(name: str, root: Path) -> Archive:
    """读一份评估产物目录。

    Raises:
        FileNotFoundError: 目录、``scores.csv`` 或 ``manifest.json`` 不存在。
        ValueError: ``scores.csv`` 没有数据行或缺列。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"产物目录不存在: {root}")
    scores = load_scores(root)
    if scores.empty:
        raise ValueError(f"{name}: {SCORES_NAME} 没有数据行")
    load_manifest(root)  # 缺 manifest 提前报，别拖到渲染层才炸
    dates = sorted({str(d) for d in scores["init_date"].dropna().unique()})
    return Archive(
        name=str(name), root=root, summary=_summary_from_scores(scores),
        meta=_meta_from_scores(scores), dates=dates,
    )


def common_dates(archives: Sequence[Archive]) -> List[str]:
    """所有模型**和它们的 summary.csv** 都有的日期（升序）。"""
    if not archives:
        raise ValueError("没有模型")
    shared: Optional[set] = None
    for archive in archives:
        own = set(archive.summary["init_date"])
        shared = own if shared is None else (shared & own)
    return sorted(shared or set())


def summary_variables(archive: Archive) -> List[str]:
    """``summary.csv`` 的 ``variables`` 列里声明过的变量（并集，字典序）。"""
    names: set = set()
    if "variables" in archive.summary.columns:
        for value in archive.summary["variables"].dropna():
            for item in str(value).split(","):
                item = item.strip()
                if item:
                    names.add(item)
    if not names:  # 没有 variables 列就退回从 rmse_<v>_mean 列名反推
        names = set(rmse_mean_columns(archive.summary))
    return sorted(names)


def rmse_mean_columns(summary: pd.DataFrame) -> List[str]:
    """``rmse_<v>_mean`` 列名里的变量（字典序）。"""
    found = []
    for column in summary.columns:
        match = re.fullmatch(r"rmse_(.+)_mean", str(column))
        if match:
            found.append(match.group(1))
    return sorted(found)


def _metric_by_lead(
    root: Path,
    metric: str,
    dates: Sequence[str],
    name: str = "",
    *,
    deterministic: bool = True,
    region: Optional[str] = None,
) -> pd.DataFrame:
    """长表里 ``metric`` 的行 → lead × 变量 表（跨起报日平均）。``region`` 给了就只算那个带。

    Raises:
        FileNotFoundError: 任何一个请求的起报日没有这个指标的结果——按 SKILL.md
            的硬性规则不静默跳过（跳过会让表和图的样本量对不上，且没人会发现）。
    """
    rows = metric_rows(
        root, metric, product_kind="deterministic" if deterministic else None, region=region,
    )
    frames = by_date_frames(rows, dates, f"{name or root}: {metric}")
    if not frames:
        raise ValueError(f"{name or root}: 没有可读的 {metric} 结果")
    stacked = pd.concat(frames.values(), axis=0)
    return stacked.groupby(level=0).mean().sort_index()


def probe_dates(root: Path, dates: Iterable[str], stem: str) -> List[str]:
    """这些起报日里，**没有** ``stem`` 结果的（升序）。用于第 2 节完整性检查。"""
    scores = _deterministic(_global(load_scores(root)))
    have = set(scores.loc[scores["metric"] == str(stem), "init_date"].astype(str))
    return sorted(str(date) for date in dates if str(date) not in have)


def load_by_lead(
    root: Path, stem: str, dates: Sequence[str], name: str = "", *, region: Optional[str] = None,
) -> pd.DataFrame:
    """``stem`` 指标按 lead 平均成一张 lead × 变量表；``region`` 给了就只算那个带。"""
    return _metric_by_lead(root, stem, dates, name, region=region)


def region_rmse_by_date(root: Path, region: str, dates: Sequence[str]) -> pd.DataFrame:
    """某个纬度带上 ``metric=rmse`` 的 **日期 × 变量** 表（对该带的全部 lead 平均）。

    全球口径的同名物是 :func:`_summary_from_scores` 里那张透视表；这里只是多一个
    ``region`` 过滤，口径与它完全一致（同样只取确定性 rmse、同样跨 lead 平均）。

    该带在这份产物里没有 rmse 行时返回**空表**而不是报错——分带块整体可以「本批未出」，
    但一台模型缺一个带不该把整份报告打掉，缺哪个带由调用方按空表判。
    """
    rows = metric_rows(root, "rmse", product_kind="deterministic", region=region)
    if rows.empty:
        return pd.DataFrame(index=[str(d) for d in dates], dtype=float)
    table = rows.pivot_table(index="init_date", columns="variable", values="value", aggfunc="mean")
    return table.reindex(index=[str(d) for d in dates])


def load_fa_ratio(root: Path, dates: Sequence[str], name: str = "") -> pd.DataFrame:
    """``activity_ratio`` → lead × 变量 的 FA 比值表。

    ``activity`` 指标在长表里展开成 ``activity_ratio`` / ``_bias`` /
    ``_forecast`` / ``_observation`` 四行，报告要的 ratio 只是其中一行。
    """
    return _metric_by_lead(root, "activity_ratio", dates, name, deterministic=False)


def spectrum_variables(root: Path) -> List[str]:
    """产物里出了纬向谱的变量（字典序）。

    以 ``diagnostics/spectrum_by_init.csv`` 的 ``variable`` 列为准——它有全部逐起报
    曲线；没有这张表就退回 ``diagnostics/spectrum_<变量>.csv`` 的文件名。
    """
    directory = Path(root) / DIAGNOSTICS_DIR
    by_init = directory / SPECTRUM_BY_INIT_NAME
    if by_init.is_file():
        raw = pd.read_csv(by_init, usecols=["variable"])
        return sorted({str(v) for v in raw["variable"].dropna().unique()})
    return sorted({
        path.stem[len("spectrum_"):]
        for path in directory.glob("spectrum_*.csv")
        if path.stem != _SPECTRUM_BY_INIT_STEM
    })


def load_spectrum(
    root: Path, dates: Sequence[str], variable: str, name: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``diagnostics/spectrum_by_init.csv`` → ``(pred, obs)`` 两张 波数 × 起报日 表。

    这张表是**逐起报**的：每个 ``(变量, 起报时刻)`` 一条曲线，是该起报各时效曲线的
    按样本数加权平均——正是老归档 ``spectrum_<date>_<变量>.csv`` 的口径。

    Raises:
        FileNotFoundError: 产物没有 ``spectrum_by_init.csv``（配置里没挂 spectrum writer）。
        ValueError: 表里没有该变量，或请求的日期区间内没有它的谱。
    """
    path = Path(root) / DIAGNOSTICS_DIR / SPECTRUM_BY_INIT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{name or root}: 缺 {DIAGNOSTICS_DIR}/{SPECTRUM_BY_INIT_NAME}"
            f"（配置的 writers 里要有 spectrum 才会出这张表）"
        )
    raw = pd.read_csv(path)
    for column in ("variable", "init_time", "wavenumber", "pred", "obs"):
        if column not in raw.columns:
            raise ValueError(f"{path} 缺列 {column!r}；现有列: {list(raw.columns)}")
    raw = raw[raw["variable"].astype(str) == str(variable)]
    if raw.empty:
        available = sorted({str(v) for v in pd.read_csv(path, usecols=["variable"])["variable"].dropna()})
        raise ValueError(f"{path}: 没有变量 {variable} 的谱；现有变量: {available}")
    raw = raw.assign(init_date=_to_init_date(raw["init_time"]))
    raw = raw[raw["init_date"].isin({str(d) for d in dates})]
    if raw.empty:
        raise ValueError(f"{path}: 请求的起报日区间内没有 {variable} 的谱")
    pred = raw.pivot_table(
        index="wavenumber", columns="init_date", values="pred", aggfunc="mean"
    ).sort_index()
    obs = raw.pivot_table(
        index="wavenumber", columns="init_date", values="obs", aggfunc="mean"
    ).sort_index()
    return pred, obs


# ================================================================== 统计检验


def _wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    """配对 Wilcoxon 符号秩检验的 p 值；样本不足或全部相等时返回 1.0。"""
    if _scipy_wilcoxon is None:
        raise ImportError(
            "做配对检验需要 scipy（scipy.stats.wilcoxon）；"
            "当前环境没有装，装上再跑，或不要调用报告生成"
        )
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if left.size < 2:
        return float("nan")
    if np.allclose(left, right, equal_nan=True):
        return 1.0
    try:
        return float(_scipy_wilcoxon(left, right).pvalue)
    except ValueError:
        return 1.0


def _holm(pvalues: Sequence[float]) -> List[float]:
    """Holm 逐步下降法校正；p 里带 nan 的原样返回 nan。"""
    index = [i for i, p in enumerate(pvalues) if math.isfinite(p)]
    ordered = sorted(index, key=lambda i: pvalues[i])
    adjusted = [float("nan")] * len(pvalues)
    running = 0.0
    total = len(ordered)
    for rank, i in enumerate(ordered):
        running = max(running, (total - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


class PairTest(NamedTuple):
    """一对模型的检验结论。"""

    a: str
    b: str
    mean_a: float
    mean_b: float
    delta_pct: float  # (A − B) / B × 100
    p_raw: float
    p_holm: float
    significant: bool
    better: str  # 显著优者；不显著时是 "—"


def paired_tests(
    frame: pd.DataFrame,
    *,
    higher_is_better: bool = False,
) -> List[PairTest]:
    """``frame`` 是 日期 × 模型，两两配对做 Wilcoxon + Holm。

    Args:
        frame: index=配对样本（日期），columns=模型名。
        higher_is_better: ACC 这类"越大越好"的指标置 True。
    """
    names = [str(c) for c in frame.columns]
    comparisons = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    if not comparisons:
        return []

    raw, means = [], {}
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce")
        means[name] = float(values.mean()) if values.notna().any() else float("nan")
    for a, b in comparisons:
        raw.append(_wilcoxon_p(frame[a], frame[b]))
    holm = _holm(raw)

    tests = []
    for (a, b), p_raw, p_holm in zip(comparisons, raw, holm):
        significant = math.isfinite(p_holm) and p_holm < 0.05
        if significant:
            a_wins = means[a] > means[b] if higher_is_better else means[a] < means[b]
            better = a if a_wins else b
        else:
            better = "—"
        base = means[b]
        delta = BASELINE * (means[a] - means[b]) / base if base else float("nan")
        tests.append(PairTest(a, b, means[a], means[b], delta, p_raw, p_holm, significant, better))
    return tests


def variable_win_counts(
    by_variable: Mapping[str, pd.DataFrame],
    names: Sequence[str],
) -> pd.DataFrame:
    """逐变量配对检验 → 每个模型对的「A 显著更优 / B 显著更优 / 无差异」计数。

    Args:
        by_variable: ``{变量: 日期 × 模型 表}``（RMSE 这类越小越好）。
    """
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wins_a = wins_b = ties = 0
            for frame in by_variable.values():
                if a not in frame.columns or b not in frame.columns:
                    continue
                p = _wilcoxon_p(frame[a], frame[b])
                if not (math.isfinite(p) and p < 0.05):
                    ties += 1
                elif float(frame[a].mean()) < float(frame[b].mean()):
                    wins_a += 1
                else:
                    wins_b += 1
            rows.append({
                "模型对": (a, b),
                "A 显著更优": wins_a,
                "B 显著更优": wins_b,
                "无显著差异": ties,
                "总变量数": wins_a + wins_b + ties,
            })
    return pd.DataFrame(rows)


# ================================================================== 派生指标


def _relative(table: pd.DataFrame) -> pd.DataFrame:
    """逐行（变量）做 100 × 值 / 几何均值，columns=模型、index=变量。"""
    geometric = np.exp(np.log(table.where(table > 0)).mean(axis=1))
    return BASELINE * table.div(geometric, axis=0)


def _composite(relative: pd.DataFrame) -> pd.Series:
    """对共享变量等权平均成每个模型一个数。"""
    return relative.mean(axis=0, skipna=True)


def _rank(series: pd.Series, *, higher_is_better: bool = False) -> pd.Series:
    """名次（1 最好）；nan 排最后。"""
    ordered = series.astype(float)
    return ordered.rank(ascending=not higher_is_better, method="min")


# ================================================================== 汇总分析


class Bundle(NamedTuple):
    """算完的一切。渲染只读它，不再碰文件。"""

    names: List[str]
    archives: Dict[str, Archive]
    dates: List[str]  # 共同日期（配对检验与 §2「有效共同日期」那一行用）
    #: 模型 -> 该模型**本次统计实际用的**起报日。默认等于 ``dates``（共同日期）；
    #: ``own_dates=True`` 时是各模型自己的全部起报日，长度可以互不相同。
    #: 凡是要给**单个模型**算平均的地方都读这里，别读 ``dates``。
    dates_by_model: Dict[str, List[str]]
    common_variables: List[str]
    acc_variable: str
    fa_variables: List[str]
    spectrum_variable: str
    spectrum_variables: List[str]
    rmse_by_date: Dict[str, pd.DataFrame]  # 模型 -> 日期 × 变量
    acc_by_lead: Dict[str, pd.DataFrame]
    fa_ratio_by_lead: Dict[str, pd.DataFrame]
    rmse_by_lead: Dict[str, pd.DataFrame]
    spectrum: Dict[str, pd.DataFrame]  # 模型 -> 波数 × {pred, obs}
    relative: pd.DataFrame  # 变量 × 模型
    per_date_relative: pd.DataFrame  # 日期 × 模型（综合相对 RMSE）
    relative_composite: pd.Series
    acc_mean: pd.Series
    fa_bias: pd.Series
    spectrum_rms: pd.Series
    rank_score: pd.Series
    per_date_acc: pd.DataFrame
    per_variable_rmse: Dict[str, pd.DataFrame]
    overall_tests: List[PairTest]
    acc_tests: List[PairTest]
    win_counts: pd.DataFrame
    anomaly_variable: str
    anomaly_model: str
    anomaly_ratio: float
    lead_relative: pd.DataFrame  # lead × 模型（逐 lead 综合相对 RMSE）
    # --- 分纬度带（可选块「附 L」）---
    region_names: List[str]  # 各模型都有的带名（升序），空 = 本批出不了这一块
    region_labels: Dict[str, str]  # 带名 -> 边界串（来自 manifest）
    region_relative_composite: pd.DataFrame  # 带名 × 模型（带内综合相对 RMSE）
    region_lead_relative: Dict[str, pd.DataFrame]  # 带名 -> lead × 模型
    region_rmse_by_lead: Dict[str, Dict[str, pd.DataFrame]]  # 带名 -> 模型 -> lead × 变量
    region_note: str  # 出不了这一块时的原因，渲染进「本批未出」那句


def _lead_relative(rmse_by_lead: Mapping[str, pd.DataFrame], names: Sequence[str],
                   variables: Sequence[str]) -> pd.DataFrame:
    """逐 lead 的综合相对 RMSE（lead × 模型）。"""
    columns = {}
    for name in names:
        frame = rmse_by_lead[name].reindex(columns=variables)
        columns[name] = frame
    stacked = pd.concat(columns, axis=1, keys=names)  # 列 = (模型, 变量)
    per_variable = {v: stacked.xs(v, level=1, axis=1) for v in variables}
    relative = [_relative(per_variable[v]) for v in variables]
    return sum(relative) / len(relative)


def analyze(
    archives: Sequence[Archive],
    *,
    declared: Optional[Mapping[str, Sequence[str]]] = None,
    own_dates: bool = False,
) -> Bundle:
    """把 N 份产物算成一份 :class:`Bundle`。

    Args:
        declared: 调用方声明的共同变量，``None`` 表示由产物自己定。
        own_dates: ``False``（默认）时每个模型的每一项统计都只在**共同日期**上算
            ——各模型覆盖的时段不同时，比的是同一批日子。
            ``True`` 时每个模型的统计与曲线用**它自己的全部起报日**（各自的平均值、
            各自的天数），只有**配对检验**仍退回共同日期：日期不对齐的样本做不了
            Wilcoxon。这条路径用于各模型覆盖时段差很多、又要直接对比的情形——
            此时基线（跨模型的几何均值）是拿不同时段的平均在比，**不公平但看得见**，
            报告里必须点名（见 :mod:`ens_report` 的 §1/§2/§7）。

    Raises:
        ValueError: 共同日期太少（< 2）或没有共享的 RMSE 变量。
        FileNotFoundError: 用到的日期上缺逐日文件——不静默跳过。
    """
    if len(archives) < 2:
        raise ValueError("多模型报告至少要两个模型")
    names = [archive.name for archive in archives]
    by_name = {archive.name: archive for archive in archives}
    dates = common_dates(archives)
    if len(dates) < 2:
        raise ValueError(f"共同日期只有 {len(dates)} 天，做不了配对检验: {dates}")
    # 每个模型「本次统计实际用哪几天」。own_dates=False 时所有模型都是共同日期，
    # 逐字保持既有行为；True 时各用各的，长度可以不一样。
    dates_by_model = {
        archive.name: (list(archive.dates) if own_dates else dates) for archive in archives
    }
    if own_dates:
        # 配对检验要有足够的共同日期才有意义；共同日期本身在 :func:`common_dates`
        # 那里已经保证 ≥2，这里只负责让「谁的天数太少」这件事在报告里说得出来。
        empty = [name for name, own in dates_by_model.items() if not own]
        if empty:
            raise ValueError(f"这些模型自己的起报日为空，算不出任何统计: {empty}")

    # --- 变量集：所有模型 summary 里都有、且真的算出了数的 RMSE 变量 ---
    shared = set(rmse_mean_columns(archives[0].summary))
    for archive in archives[1:]:
        shared &= set(rmse_mean_columns(archive.summary))
    variables = sorted(
        v for v in shared
        if all(archive.summary[f"rmse_{v}_mean"].notna().any() for archive in archives)
    )
    if not variables:
        raise ValueError("这些模型没有共享的 rmse_<变量>_mean 列，算不出分变量结果")

    rmse_by_date: Dict[str, pd.DataFrame] = {}
    for archive in archives:
        indexed = archive.summary.set_index("init_date")
        columns = {v: pd.to_numeric(indexed[f"rmse_{v}_mean"], errors="coerce") for v in variables}
        rmse_by_date[archive.name] = pd.DataFrame(columns).reindex(
            index=dates_by_model[archive.name]
        )

    # --- 逐日文件（缺就报错，不静默跳过）---
    # acc_<date>_det.csv 里存的是相关系数本身（-1..1），报告统一按百分数呈现，
    # 所以在入口乘 100，下游（表、图、检验）拿到的都是 %。
    acc_by_lead = {
        a.name: load_by_lead(a.root, "acc", dates_by_model[a.name], a.name) * BASELINE
        for a in archives
    }
    fa_ratio_by_lead = {
        a.name: load_fa_ratio(a.root, dates_by_model[a.name], a.name) for a in archives
    }
    rmse_by_lead = {
        a.name: load_by_lead(a.root, "rmse", dates_by_model[a.name], a.name) for a in archives
    }

    acc_columns = set(acc_by_lead[names[0]].columns)
    for name in names[1:]:
        acc_columns &= set(acc_by_lead[name].columns)
    acc_variable = _pick_variable(acc_columns, variables)

    fa_columns = set(fa_ratio_by_lead[names[0]].columns)
    for name in names[1:]:
        fa_columns &= set(fa_ratio_by_lead[name].columns)
    fa_variables = sorted(fa_columns)
    if not fa_variables:
        raise ValueError("这些模型没有共享的 <变量>_ratio 列，算不出 FA 偏差")

    spectrum_vars = None
    for archive in archives:
        own = set(spectrum_variables(archive.root))
        spectrum_vars = own if spectrum_vars is None else (spectrum_vars & own)
    spectrum_vars = sorted(spectrum_vars or set())
    if not spectrum_vars:
        raise ValueError(
            "这些模型的产物里都没有纬向谱（diagnostics/"
            f"{SPECTRUM_BY_INIT_NAME}），算不出频谱指标"
        )
    spectrum_variable = _pick_variable(set(spectrum_vars), variables)

    spectrum: Dict[str, pd.DataFrame] = {}
    spectrum_rms_values: Dict[str, float] = {}
    for archive in archives:
        pred, obs = load_spectrum(
            archive.root, dates_by_model[archive.name], spectrum_variable, archive.name
        )
        spectrum[archive.name] = pd.DataFrame({"pred": pred.mean(axis=1), "obs": obs.mean(axis=1)})
        per_date = []
        for date in pred.columns:
            p = pred[date].where(lambda s: s > 0)
            o = obs[date].where(lambda s: s > 0)
            ratio = (p / o).dropna()
            ratio = ratio[ratio.index > 0]
            if not ratio.empty:
                per_date.append(float(np.sqrt(np.mean(np.log10(ratio.values) ** 2))))
        spectrum_rms_values[archive.name] = BASELINE * float(np.mean(per_date)) if per_date else float("nan")

    # --- 派生指标 ---
    # 跨模型的**几何均值基线**（``_relative`` 的分母）逐行算，所以每一行（变量／日期）
    # 必须所有模型都在场，否则这一行的基线是拿残缺的模型集算的。own_dates=True 时
    # 各模型天数不同，**只有配对检验那些帧**需要按下面的 ``_paired`` 裁齐，
    # 按变量取平均的 ``mean_rmse`` 不受影响：每个模型各自的均值都在。
    mean_rmse = pd.DataFrame({name: rmse_by_date[name].mean(axis=0) for name in names}).reindex(variables)
    relative = _relative(mean_rmse)
    relative_composite = _composite(relative)

    #: 逐日配对帧统一走这个：裁到共同日期；``own_dates`` 时还要把任何模型缺数的
    #: 那一天整行丢掉——日期不齐会让「谁跟谁配对」这件事说不清。默认路径
    #: **不丢**，逐字保持既有行为。
    def _pair(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.reindex(index=[str(d) for d in dates])
        return frame.dropna(how="any") if own_dates else frame

    per_variable_relative = {}
    for variable in variables:
        columns = {name: rmse_by_date[name][variable] for name in names}
        per_variable_relative[variable] = _relative(_pair(pd.DataFrame(columns))).reindex(index=dates)
    per_date_relative = sum(per_variable_relative.values()) / len(variables)
    # ``per_variable_rmse`` 进 Bundle，§4.1 拿它的**均值**填表——own_dates 时那就是
    # 各模型自己的全部日期；「显著优胜」那一列另由共同日期上的检验给，
    # 两者口径不同，报告里写明。
    per_variable_rmse = {
        variable: pd.DataFrame({name: rmse_by_date[name][variable] for name in names})
        for variable in variables
    }

    # 逐日 ACC 直接读逐日文件，而不是从"先按 lead 平均、再按日期平均"的表里取——
    # 两者数值上等价，但配对检验要的是逐日样本，不能只有均值。
    per_date_acc = pd.DataFrame({
        name: _acc_by_date(by_name[name].root, dates_by_model[name], acc_variable, name)
        for name in names
    })
    acc_mean = per_date_acc.mean(axis=0)

    deviation = pd.DataFrame({
        name: (fa_ratio_by_lead[name][fa_variables] - 1.0).abs().mean(axis=1) for name in names
    })
    fa_bias = BASELINE * deviation.mean(axis=0)
    spectrum_rms = pd.Series(spectrum_rms_values)

    rank_score = (
        _rank(relative_composite)
        + _rank(acc_mean, higher_is_better=True)
        + _rank(fa_bias)
        + _rank(spectrum_rms)
    ) / 4.0

    # 检验一律在共同日期上做（见上方 ``_pair``）：§6 表里印的均值因此是**共同日期**
    # 上的均值，与 §4 各模型全量日期的均值**不是同一个数**，报告里两处都写明。
    overall_tests = paired_tests(per_date_relative)
    acc_tests = paired_tests(_pair(per_date_acc), higher_is_better=True)
    win_counts = variable_win_counts(
        {v: _pair(frame) for v, frame in per_variable_rmse.items()}, names
    )

    anomaly_variable, anomaly_model, anomaly_ratio = _detect_anomaly(mean_rmse, names)
    lead_relative = _lead_relative(rmse_by_lead, names, variables)

    # --- 分纬度带（可选块「附 L」）：各模型都有的带才出，一个都没有就整块「本批未出」---
    shared_regions: Optional[set] = None
    for archive in archives:
        own = set(region_names(archive.root))
        shared_regions = own if shared_regions is None else (shared_regions & own)
    region_list = sorted(shared_regions or set())
    region_label_map = region_labels(archives[0].root)
    region_rmse_by_lead: Dict[str, Dict[str, pd.DataFrame]] = {}
    region_lead_relative: Dict[str, pd.DataFrame] = {}
    per_region_composite: Dict[str, pd.Series] = {}
    region_note = ""
    for region in region_list:
        own_models = {
            archive.name: region_rmse_by_date(
                archive.root, region, dates_by_model[archive.name]
            )
            .reindex(columns=variables).mean(axis=0)
            for archive in archives
        }
        # 带内**自己的**几何均值基线：跨带比绝对值没有意义（热带与极区差一个量级），
        # 除成本带基线之后，「哪个模型在这个带上更稳」才是可比的。
        per_region_composite[region] = _composite(_relative(pd.DataFrame(own_models).reindex(index=variables)))
        region_rmse_by_lead[region] = {
            archive.name: _metric_by_lead(
                archive.root, "rmse", dates_by_model[archive.name], archive.name, region=region
            )
            for archive in archives
        }
        region_lead_relative[region] = _lead_relative(region_rmse_by_lead[region], names, variables)

    if not region_list:
        region_note = (
            "这批产物的 scores.csv 里没有共同的 region 行——"
            "评测配置的 config_options.regions 为空，或者各模型的分带对不上。"
        )
    region_relative_composite = pd.DataFrame(per_region_composite).T

    return Bundle(
        names=names, archives=by_name, dates=dates, dates_by_model=dates_by_model,
        common_variables=variables,
        acc_variable=acc_variable, fa_variables=fa_variables,
        spectrum_variable=spectrum_variable, spectrum_variables=spectrum_vars,
        rmse_by_date=rmse_by_date, acc_by_lead=acc_by_lead, fa_ratio_by_lead=fa_ratio_by_lead,
        rmse_by_lead=rmse_by_lead, spectrum=spectrum,
        relative=relative, per_date_relative=per_date_relative,
        relative_composite=relative_composite, acc_mean=acc_mean, fa_bias=fa_bias,
        spectrum_rms=spectrum_rms, rank_score=rank_score, per_date_acc=per_date_acc,
        per_variable_rmse=per_variable_rmse,
        overall_tests=overall_tests, acc_tests=acc_tests, win_counts=win_counts,
        anomaly_variable=anomaly_variable, anomaly_model=anomaly_model,
        anomaly_ratio=anomaly_ratio, lead_relative=lead_relative,
        region_names=region_list, region_labels=region_label_map,
        region_relative_composite=region_relative_composite,
        region_lead_relative=region_lead_relative,
        region_rmse_by_lead=region_rmse_by_lead, region_note=region_note,
    )


def _acc_by_date(root: Path, dates: Sequence[str], variable: str, name: str) -> pd.Series:
    """逐起报日的 ACC 变量在全部 lead 上的平均（百分数），index=起报日。"""
    acc = metric_rows(root, "acc", product_kind="deterministic")
    rows = acc[acc["variable"].astype(str) == str(variable)]
    if rows.empty:
        available = sorted({str(v) for v in acc["variable"].dropna()})
        raise ValueError(f"{name}: 产物里没有变量 {variable} 的 acc 结果；现有变量: {available}")
    per_date = rows.groupby(rows["init_date"].astype(str))["value"].mean()
    wanted = [str(date) for date in dates]
    missing = [date for date in wanted if date not in per_date.index]
    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} 个起报日没有 {variable} 的 acc 结果（如 {missing[0]}）"
        )
    return BASELINE * per_date.reindex(wanted)


def _pick_variable(candidates: Iterable[str], fallback: Sequence[str]) -> str:
    """优先 z500，其次候选里字典序最小的，最后退回 fallback 的第一个。"""
    options = sorted(str(c) for c in candidates)
    if PREFERRED_VARIABLE in options:
        return PREFERRED_VARIABLE
    if options:
        return options[0]
    return str(fallback[0])


def _detect_anomaly(mean_rmse: pd.DataFrame, names: Sequence[str]) -> Tuple[str, str, float]:
    """挑「跨模型量级最离谱」的变量：max / median 比值最大的那个。"""
    ratios = {}
    for variable in mean_rmse.index:
        row = mean_rmse.loc[variable].dropna()
        median = float(row.median())
        ratios[variable] = float(row.max()) / median if median > 0 else float("nan")
    finite = {v: r for v, r in ratios.items() if math.isfinite(r)}
    if not finite:
        variable = str(mean_rmse.index[0])
    else:
        variable = max(finite, key=lambda v: finite[v])
    row = mean_rmse.loc[variable].dropna()
    model = str(row.idxmax())
    return variable, model, float(row.max()) / float(row.median()) if float(row.median()) > 0 else float("nan")


# ==================================================================== 渲染


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value, digits: int = 3) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _signed(value, digits: int = 2) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:+.{digits}f}"


def _model_count_word(count: int) -> str:
    return _CN_NUM.get(count, str(count))


def _date_span(dates: Sequence[str]) -> str:
    """日期跨度写成「集合」形式：``20250102，20251215``。

    分隔符是**全角逗号**而不是连接号——起点终点是一组日期的两个端点，
    读作集合比读作区间更贴合「这批共同日期」的含义。调用方大多已经套了括号，
    所以这里**不带括号**，套两层会变成 ``（（20250102，20251215））``。
    """
    if not dates:
        return "—"
    return f"{dates[0]}，{dates[-1]}" if len(dates) > 1 else dates[0]


def _lead_span(leads: Sequence[float]) -> Tuple[str, str, int]:
    finite = [float(x) for x in leads if math.isfinite(float(x))]
    if not finite:
        return "—", "—", 0
    return f"{min(finite):g}", f"{max(finite):g}", len(finite)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _bullets(items: Sequence[str], empty: str = "• 无。") -> List[str]:
    return [f"• {item}" for item in items] if items else [empty]


# ------------------------------------------------------------------ 各章节


def _section_samples(bundle: Bundle, declared: Optional[Mapping[str, Sequence[str]]]) -> List[str]:
    lines = ["# 2. 样本与完整性", "", "**样本与完整性**", ""]
    rows = []
    for name in bundle.names:
        archive = bundle.archives[name]
        own = set(archive.summary["init_date"])
        common = own & set(bundle.dates)
        rows.append([
            name, str(len(own)), _date_span(sorted(own)),
            str(len(common)), str(len(own) - len(common)),
        ])
    lines += _table(["**模型**", "**有效日期数**", "**日期范围**", "**共同日期**", "**非共同日期**"], rows)
    lines += ["", "**完整性检查**", ""]

    check_rows = []
    for name in bundle.names:
        archive = bundle.archives[name]
        meta = archive.meta
        # scores.csv 里出现过、却没有确定性 rmse 结果的起报日。正常产物是空的；
        # 非空说明这次只算了一部分指标，那些日期也就进不了 summary 和共同日期。
        missing = probe_dates(archive.root, archive.dates, "rmse")
        nan_variables = [
            v for v in rmse_mean_columns(archive.summary)
            if archive.summary[f"rmse_{v}_mean"].isna().all()
        ]
        n_dates = meta.get("n_dates")
        n_ok = meta.get("n_ok")
        check_rows.append([
            name,
            f"{n_ok}/{n_dates}" if n_ok is not None and n_dates is not None else "—",
            str(len(archive.dates)),
            str(len(missing)) if missing else "无",
            "无" if not nan_variables else "、".join(nan_variables[:4]),
        ])
    lines += _table(
        ["**模型**", "**n_ok/n_dates**", "**起报日数**", "**缺 RMSE 的日期**", "**NaN RMSE**"],
        check_rows,
    )
    lines.append("")

    failures = {name: bundle.archives[name].meta.get("failures") or [] for name in bundle.names}
    broken = {name: items for name, items in failures.items() if items}
    if broken:
        detail = "；".join(f"{name} 有 {len(items)} 个起报点失败" for name, items in broken.items())
    else:
        detail = "所有模型的产物里每一行 status 都是 success，n_ok = n_dates，没有失败起报点"
    lines.append(
        f"{detail}。正文与附录统一使用 {len(bundle.dates)} 个共同日期"
        f"（{_date_span(bundle.dates)}），非共同日期不参与任何统计与曲线，"
        f"因此各表的样本量一致、可直接横向比较。"
    )
    lines.append("")

    declared = declared or {}
    unknown = [name for name in bundle.names if name not in declared]
    if not unknown:
        lines.append("各模型的声明变量清单由 `--declared` 提供，与产物实际变量逐一对照见第 6.2 节。")
    elif len(unknown) == len(bundle.names):
        lines.append(
            "本次未提供各模型的声明变量清单（`--declared`），"
            "第 6.2 节的「声明变量数」一律记 —，只比较产物里实际算出来的变量。"
        )
    else:
        lines.append(
            f"本次只为 {'、'.join(n for n in bundle.names if n not in unknown)} 提供了声明变量清单"
            f"（`--declared`），{'、'.join(unknown)} 的「声明变量数」记 —，"
            f"变量覆盖是否完整只能看产物里实际算出来的变量。"
        )

    lines += ["", "## 指标口径说明（FA）", ""]
    first, last, count = _lead_span(bundle.rmse_by_lead[bundle.names[0]].index)
    lines.append(
        f"本报告中的FA仅指Forecast Activity。为避免与降水二分类指标False Alarm"
        f"（同样缩写为FA）混淆，FA统一采用全程{first}–{last}h、共{count}个lead的统计结果，"
        f"不做短、中、长时效分段；文中的分时效比较仅针对RMSE，"
        f"并明确排除{bundle.anomaly_variable}。"
    )
    return lines


def _section_overall(bundle: Bundle) -> List[str]:
    lines = ["# 3. 总体模型表现", "", f"**{len(bundle.names)}模型总体指标**", ""]
    order = bundle.relative_composite.sort_values().index.tolist()
    rank = _rank(bundle.relative_composite)
    rows = []
    for name in order:
        rows.append([
            f"{int(rank[name])}", name,
            _fmt(bundle.relative_composite[name]),
            _fmt(bundle.acc_mean[name]),
            _fmt(bundle.fa_bias[name], 4),
            _fmt(bundle.spectrum_rms[name], 2),
            _fmt(bundle.rank_score[name], 1),
        ])
    lines += _table(
        ["**排名**", "**模型**", "**相对 RMSE**", "**ACC (%)**",
         "**FA 偏差(全程, pp)**", "**频谱 RMS×100**", "**平均排名分**"],
        rows,
    )
    lines += ["", "**综合相对 RMSE 配对 Wilcoxon + Holm 检验**", ""]
    test_rows = []
    for test in bundle.overall_tests:
        test_rows.append([
            f"{test.a} vs {test.b}", _signed(test.delta_pct),
            "是" if test.significant else "否", test.better,
            _fmt(test.p_holm, 4),
        ])
    lines += _table(
        ["**比较**", "**相对变化 A vs B (%)**", "**Holm 显著**", "**显著优者**", "**Holm p**"],
        test_rows,
    )
    lines.append("")
    best = order[0]
    worst = order[-1]
    lines.append(
        f"相对RMSE为100时表示{len(bundle.names)}模型几何均值基线；越低越好。"
        f"{best} 的综合相对 RMSE 最低（{_fmt(bundle.relative_composite[best])}），"
        f"{worst} 最高（{_fmt(bundle.relative_composite[worst])}）；"
        f"表内检验全部基于 {len(bundle.dates)} 个共同日期的逐日配对，"
        f"Holm p < 0.05 才算显著。"
    )
    return lines


def _section_rmse(bundle: Bundle) -> List[str]:
    lines = ["# 4. RMSE 分变量结果", "", "**分变量 RMSE（共同日期平均）**", ""]
    rows = []
    for variable in bundle.common_variables:
        values = pd.Series({name: bundle.rmse_by_date[name][variable].mean() for name in bundle.names})
        best = str(values.idxmin())
        row = [variable] + [_fmt(values[name], 4) for name in bundle.names] + [best]
        rows.append(row)
    lines += _table(["**变量**"] + list(bundle.names) + ["**Best**"], rows)

    lines += ["", "**分变量配对检验显著性汇总**", ""]
    test_rows = []
    for _, record in bundle.win_counts.iterrows():
        a, b = record["模型对"]
        test_rows.append([
            f"{a} vs {b}", str(int(record["A 显著更优"])), str(int(record["B 显著更优"])),
            str(int(record["无显著差异"])), str(int(record["总变量数"])),
        ])
    lines += _table(
        ["**模型对**", "**A 显著更优变量数**", "**B 显著更优变量数**", "**无显著差异**", "**总变量数**"],
        test_rows,
    )
    lines.append("")
    lines.append(
        "表内模型对的次序与第 3 节一致（A、B 即「模型对」列里写明的两个模型，前者为 A）。"
        "逐变量检验用共同日期上的逐日 RMSE 做配对 Wilcoxon，未做多重校正，"
        "因此「显著」只作为方向性提示；要看校正后的结论请看第 3 节的综合检验。"
    )
    return lines


def _section_acc_fa_spectrum(bundle: Bundle) -> List[str]:
    first, last, _ = _lead_span(bundle.rmse_by_lead[bundle.names[0]].index)
    lines = [
        f"# 5. ACC、FA（Forecast Activity，全程{first}–{last}h）与频谱",
        "",
        f"**{len(bundle.names)}模型 ACC、FA 与频谱**",
        "",
    ]
    rows = []
    for name in bundle.names:
        rows.append([
            name, _fmt(bundle.acc_mean[name]),
            _fmt(bundle.fa_bias[name], 4), _fmt(bundle.spectrum_rms[name], 2),
        ])
    lines += _table(["**模型**", "**ACC (%)**", "**FA 偏差(全程, pp)**", "**频谱对数 RMS×100**"], rows)

    lines += ["", "**ACC 配对检验**", ""]
    test_rows = []
    for test in bundle.acc_tests:
        test_rows.append([
            f"{test.a} vs {test.b}", _signed(test.mean_a - test.mean_b, 3),
            "是" if test.significant else "否", test.better, _fmt(test.p_holm, 4),
        ])
    lines += _table(
        ["**比较**", "**ACC 差 A-B (pp)**", "**Holm 显著**", "**显著优者**", "**Holm p**"],
        test_rows,
    )
    lines.append("")

    acc_order = bundle.acc_mean.sort_values(ascending=False)
    spread = float(acc_order.iloc[0] - acc_order.iloc[-1])
    near = []
    for name in bundle.names:
        frame = bundle.acc_by_lead[name]
        if bundle.acc_variable in frame.columns:
            first_lead = float(frame[bundle.acc_variable].iloc[0])
            near.append((name, first_lead))
    near.sort(key=lambda item: -item[1])
    acc_note = (
        f"{bundle.acc_variable} ACC 的极差为 {_fmt(spread)} pp（{acc_order.index[0]} 最高 "
        f"{_fmt(acc_order.iloc[0])}，{acc_order.index[-1]} 最低 {_fmt(acc_order.iloc[-1])}）。"
    )
    if near and near[0][1] > SATURATED_ACC:
        acc_note += (
            f"需要注意的是 {near[0][0]} 在首时效的 {bundle.acc_variable} ACC 达到 "
            f"{_fmt(near[0][1])}%，已逼近饱和，"
            f"存在「ACC 定义过宽、近端几乎不可分辨」的风险（见 6.3）。"
        )
    lines.append(acc_note)
    return lines


def _section_risks(bundle: Bundle, declared: Optional[Mapping[str, Sequence[str]]]) -> List[str]:
    variable = bundle.anomaly_variable
    lines = ["# 6. 异常与风险", "", f"## 6.1 {variable}量级异常", ""]

    values = pd.Series({name: bundle.rmse_by_date[name][variable].mean() for name in bundle.names})
    values = values.sort_values(ascending=False)
    rows = [[name, _fmt(values[name], 4)] for name in values.index]
    lines += _table(["**模型**", f"**共同日期平均 {variable} RMSE**"], rows)

    lines += ["", f"**逐时效倍数（{bundle.anomaly_model} / 其他{len(bundle.names) - 1}模型中位数）**", ""]
    lead_rows = []
    for lead in bundle.lead_relative.index:
        column = bundle.rmse_by_lead[bundle.anomaly_model].get(variable)
        others = [
            bundle.rmse_by_lead[name].get(variable) for name in bundle.names
            if name != bundle.anomaly_model
        ]
        others = [s for s in others if s is not None and lead in s.index]
        if column is None or lead not in column.index or not others:
            continue
        median = float(np.median([float(s.loc[lead]) for s in others]))
        value = float(column.loc[lead])
        lead_rows.append([f"{float(lead):g}", _fmt(value / median, 3) if median else "—"])
    lines += _table(["**Lead (h)**", f"**{bundle.anomaly_model} / 其他{len(bundle.names) - 1}模型中位数**"], lead_rows)
    lines.append("")

    threshold = 3.0
    if math.isfinite(bundle.anomaly_ratio) and bundle.anomaly_ratio >= threshold:
        lines.append(
            f"{bundle.anomaly_model} 的 {variable} RMSE 是其余模型中位数的 "
            f"{_fmt(bundle.anomaly_ratio, 2)} 倍，超出「同量级」的可接受范围（阈值 {threshold:g}×）。"
            f"该差异在所有 lead 上一致存在，不是个别时效的抖动，"
            f"更像是量纲/单位或变量定义层面的问题，而不是预报质量问题——"
            f"建议核对 {variable} 的预报与观测是否使用了同一套单位与层次定义后再采信该变量的结论。"
        )
    else:
        lines.append(
            f"{bundle.anomaly_model} 的 {variable} RMSE 是其余模型中位数的 "
            f"{_fmt(bundle.anomaly_ratio, 2)} 倍，未超过 {threshold:g}× 的异常阈值，"
            f"属于正常的模型间水平差异，不单独作为风险项。"
        )

    lines += ["", "## 6.2 变量覆盖不完整", "", "**声明变量数与汇总变量数**", ""]
    declared = declared or {}
    rows = []
    for name in bundle.names:
        actual = summary_variables(bundle.archives[name])
        claimed = declared.get(name)
        if claimed is None:
            rows.append([name, "—", str(len(actual)), "—"])
        else:
            claimed_list = [str(v).strip() for v in claimed if str(v).strip()]
            gap = sorted(set(claimed_list) - set(actual))
            rows.append([name, str(len(claimed_list)), str(len(actual)), "、".join(gap) if gap else "无"])
    lines += _table(["**模型**", "**声明变量数**", "**汇总变量数**", "**汇总缺失变量**"], rows)
    lines.append("")

    counts = {name: len(summary_variables(bundle.archives[name])) for name in bundle.names}
    if len(set(counts.values())) == 1:
        lines.append(
            f"各模型的产物都覆盖 {list(counts.values())[0]} 个变量，变量覆盖一致；"
            f"本报告只对这 {len(bundle.common_variables)} 个共享 RMSE 变量做横向比较。"
        )
    else:
        detail = "、".join(f"{name} {count} 个" for name, count in sorted(counts.items()))
        lines.append(
            f"各模型的汇总变量数不一致（{detail}），说明有模型没算全变量。"
            f"本报告只对 {len(bundle.common_variables)} 个共享变量做横向比较，"
            f"缺变量的模型在这些变量上的结论不完整，选型时需要一并考虑。"
        )

    lines += ["", "## 6.3 ACC 定义风险", ""]
    first_lead = None
    frame = bundle.acc_by_lead[bundle.names[0]]
    if bundle.acc_variable in frame.columns and not frame.empty:
        first_lead = float(frame[bundle.acc_variable].iloc[0])
    if first_lead is not None and first_lead > SATURATED_ACC:
        lines.append(
            f"首时效 {bundle.acc_variable} ACC 已到 {_fmt(first_lead)}%，接近饱和（阈值 "
            f"{SATURATED_ACC:g}%）。这说明所用 ACC 定义（距平相关）在近端对模型差异几乎不敏感，"
            f"用它排名会把不同模型压成几乎同分，不能单独作为选型依据；"
            f"建议同时看 RMSE 与频谱。"
        )
    else:
        lines.append(
            f"首时效 {bundle.acc_variable} ACC 为 "
            f"{_fmt(first_lead) + '%' if first_lead is not None else '—'}，"
            f"没有出现接近饱和（{SATURATED_ACC:g}%）的现象，"
            f"ACC 在本批次上对模型差异仍有区分度；但 ACC 的口径（距平基准、区域权重）"
            f"由评测侧定义，跨批次比较时需确认口径未变。"
        )

    lines += ["", "## 6.4 原始场独立复算不足", ""]
    lines.append(
        "本报告的 RMSE / ACC / FA / 频谱全部读自各模型产物目录里的 scores.csv 汇总行，"
        "没有回到原始预报场与观测场做独立复算；"
        "因此评测环节（变量映射、插值、单位换算）如果出错，本报告会原样继承，无法自查。"
        "产物目录里也确实不含原始场（只有 scores.csv、manifest.json 与 diagnostics/ 下的谱表），"
        "这一点无法在本报告内弥补。结论用于模型间横向比较是充分的，"
        "用于绝对精度认定前建议抽一个日期回到原始场复算一次。"
    )
    return lines


def _section_acceptance(bundle: Bundle) -> List[str]:
    lines = ["# 7. 验收建议", ""]
    items = []
    best = bundle.relative_composite.idxmin()
    items.append(
        f"以共同日期上的综合相对 RMSE 为准，{best} 最低"
        f"（{_fmt(bundle.relative_composite[best])}），可作为默认首选模型。"
    )
    acc_best = bundle.acc_mean.idxmax()
    items.append(
        f"大尺度形势场优先看 ACC，本批次 {acc_best} 最高（{_fmt(bundle.acc_mean[acc_best])}）。"
    )
    fa_best = bundle.fa_bias.idxmin()
    items.append(
        f"强度/活跃度类应用优先看 FA 偏差，{fa_best} 偏离 1 最少"
        f"（{_fmt(bundle.fa_bias[fa_best], 4)} pp）。"
    )
    worst = bundle.relative_composite.idxmax()
    items.append(
        f"{worst} 的综合相对 RMSE 最高（{_fmt(bundle.relative_composite[worst])}），"
        f"在与 {best} 的配对检验中若被判定显著更差（见第 3 节），暂不建议作为主用模型。"
    )
    items.append(
        f"全部结论都建立在 {len(bundle.dates)} 个共同日期（{_date_span(bundle.dates)}）上，"
        f"样本期外的季节与区域不外推。"
    )
    lines += _bullets(items)
    return lines


def _section_comparison(bundle: Bundle) -> List[str]:
    count = len(bundle.names)
    lines = [
        f"# 8. {count}模型具体对比说明",
        "",
        f"本节在{len(bundle.dates)}个共同日期、{len(bundle.common_variables)}个共享RMSE变量、"
        f"{bundle.acc_variable} ACC、{len(bundle.fa_variables)}个FA ratio变量和"
        f"{len(bundle.spectrum_variables)}个频谱变量的统一口径下，对{count}个模型进行逐项比较。"
        f"所有配对检验均使用Wilcoxon符号秩检验并进行Holm多重校正；"
        f"统计显著不代表业务差异一定具有实际重要性。",
        "",
        f"**{count}模型综合对比**",
        "",
    ]
    order = bundle.relative_composite.sort_values().index.tolist()
    ranks = {
        "rel": _rank(bundle.relative_composite),
        "acc": _rank(bundle.acc_mean, higher_is_better=True),
        "fa": _rank(bundle.fa_bias),
        "spec": _rank(bundle.spectrum_rms),
    }
    rows = []
    for name in order:
        row = [
            name,
            f"{_fmt(bundle.relative_composite[name])}（第{int(ranks['rel'][name])}）",
            f"{_fmt(bundle.acc_mean[name])}（第{int(ranks['acc'][name])}）",
            f"{_fmt(bundle.fa_bias[name], 4)}（第{int(ranks['fa'][name])}）",
            f"{_fmt(bundle.spectrum_rms[name], 2)}（第{int(ranks['spec'][name])}）",
        ]
        strengths, weaknesses = _pros_cons(bundle, name, ranks, count)
        rows.append(row + [strengths, weaknesses])
    lines += _table(
        ["**模型**", "**综合相对RMSE**", "**ACC**", "**FA偏差(pp)**", "**频谱对数RMS×100**",
         "**主要优势**", "**主要短板**"],
        rows,
    )

    # 此处原本是「## 8.1 逐模型结论」小标题，已去掉（只留结论内容），
    # 后面的「关键配对差异」顺势补位成 8.1。
    lines += [""]
    items = []
    for name in order:
        strengths, weaknesses = _pros_cons(bundle, name, ranks, count)
        items.append(
            f"{name}：综合相对RMSE {_fmt(bundle.relative_composite[name])}"
            f"（第{int(ranks['rel'][name])}）、ACC {_fmt(bundle.acc_mean[name])}%"
            f"（第{int(ranks['acc'][name])}）、FA偏差 {_fmt(bundle.fa_bias[name], 4)} pp"
            f"（第{int(ranks['fa'][name])}）、频谱对数RMS×100 {_fmt(bundle.spectrum_rms[name], 2)}"
            f"（第{int(ranks['spec'][name])}）；优势在{strengths}，短板在{weaknesses}。"
        )
    lines += _bullets(items)

    lines += ["", "## 8.1 关键配对差异", "", "**关键配对差异**", ""]
    rows = []
    for test in bundle.overall_tests:
        record = bundle.win_counts[
            bundle.win_counts["模型对"].apply(lambda pair: set(pair) == {test.a, test.b})
        ]
        if record.empty:
            wins = "—"
        else:
            wins_a = int(record.iloc[0]["A 显著更优"])
            wins_b = int(record.iloc[0]["B 显著更优"])
            wins = f"{test.a} {wins_a} : {wins_b} {test.b}"
        if test.significant:
            summary = f"综合相对RMSE 差异 {_signed(test.delta_pct)}%，Holm p = {_fmt(test.p_holm, 4)}"
            explain = f"{test.better} 在综合口径上显著更优"
        else:
            summary = f"综合相对RMSE 差异 {_signed(test.delta_pct)}%，Holm p = {_fmt(test.p_holm, 4)}"
            explain = "未通过 Holm 校正，两者的总体差距不足以判定优劣"
        rows.append([f"{test.a} vs {test.b}", summary, wins, explain])
    lines += _table(["**模型对**", "**综合差异**", "**显著更优变量数**", "**解释**"], rows)
    return lines


def _pros_cons(bundle: Bundle, name: str, ranks: Mapping[str, pd.Series], count: int) -> Tuple[str, str]:
    labels = {
        "rel": "综合 RMSE", "acc": "ACC",
        "fa": "FA 偏差", "spec": "频谱保真",
    }
    ordered = sorted(ranks, key=lambda key: int(ranks[key][name]))
    best = ordered[0]
    worst = ordered[-1]
    if int(ranks[best][name]) == 1:
        strength = f"{labels[best]}（{count}模型第一）"
    elif int(ranks[best][name]) <= max(1, count // 2):
        strength = f"{labels[best]}（第{int(ranks[best][name])}）"
    else:
        strength = f"各维度均无领先项（最好的一项{labels[best]}也只是第{int(ranks[best][name])}）"
    weakness = f"{labels[worst]}（第{int(ranks[worst][name])}）"
    return strength, weakness


def _section_sensitivity(bundle: Bundle) -> List[str]:
    variable = bundle.anomaly_variable
    lines = [f"# 9. {variable}敏感性分析与分时效对比", ""]
    lines.append(
        f"{variable} 在模型间存在量级差异（见 6.1），为确认它是否主导了总体排序，"
        f"下表给出「含 {variable}」与「不含 {variable}」两种口径下的综合相对 RMSE。"
    )
    lines += ["", f"**含/不含{variable}两种口径**", ""]

    without = [v for v in bundle.common_variables if v != variable]
    if without:
        recomputed = _composite(_relative(
            pd.DataFrame({
                name: bundle.rmse_by_date[name][without].mean(axis=0) for name in bundle.names
            }).reindex(without)
        ))
        without_order = recomputed.sort_values().index.tolist()
        without_note = " < ".join(f"{n} {_fmt(recomputed[n])}" for n in without_order)
        without_cells = [_fmt(recomputed[name]) for name in bundle.names]
    else:
        without_order = None
        without_note = "没有其他共享变量可比"
        without_cells = ["—"] * len(bundle.names)

    with_order = bundle.relative_composite.sort_values().index.tolist()
    with_note = " < ".join(f"{n} {_fmt(bundle.relative_composite[n])}" for n in with_order)
    lines += _table(
        ["**口径**"] + list(bundle.names) + ["**结论**"],
        [
            [f"含{variable}"] + [_fmt(bundle.relative_composite[n]) for n in bundle.names] + [with_note],
            [f"不含{variable}"] + without_cells + [without_note],
        ],
    )
    lines.append("")
    if without_order is None or with_order == without_order:
        lines.append(
            f"两种口径下排序完全一致，说明 {variable} 的量级差异虽然明显，"
            f"但没有改变模型间的相对格局，总体结论对该变量不敏感。"
        )
    else:
        lines.append(
            f"两种口径下排序发生变化，说明 {variable} 主导了总体排序；"
            f"在 {variable} 的单位与定义核对清楚之前，不应据此对相关模型下最终结论。"
        )

    lines += ["", "**分时效段相对综合RMSE**", ""]
    rows = []
    for low, high in LEAD_BANDS:
        leads = [lead for lead in bundle.lead_relative.index if low < float(lead) <= high]
        if not leads:
            continue
        band = bundle.lead_relative.loc[leads].mean(axis=0)
        order = band.sort_values()
        span = f"{float(min(leads)):g}–{float(max(leads)):g}h"
        rows.append([
            span, str(order.index[0]),
            " < ".join(f"{n} {_fmt(band[n])}" for n in order.index),
            f"{len(leads)} 个 lead",
        ])
    lines += _table(
        ["**时效段**", "**最佳模型**", f"**{len(bundle.names)}模型相对综合RMSE排序（越低越好）**", "**说明**"],
        rows,
    )
    lines.append("")
    lines.append(
        "分时效口径与第 3 节一致：逐 lead 先按变量做几何均值归一，再对共享变量等权平均；"
        "因此这里的数值与第 3 节的全程值同量纲，但不能直接与第 3 节的表相加。"
    )
    return lines


def _section_conclusion(bundle: Bundle) -> List[str]:
    lines = ["# 10. 总述结论", ""]
    order = bundle.relative_composite.sort_values().index.tolist()
    best = order[0]
    items = [
        f"在 {len(bundle.dates)} 个共同日期（{_date_span(bundle.dates)}）上，"
        f"{best} 的综合相对 RMSE 最低（{_fmt(bundle.relative_composite[best])}），"
        f"较几何均值基线低 {_fmt(BASELINE - bundle.relative_composite[best])}。",
        f"ACC 最高的是 {bundle.acc_mean.idxmax()}（{_fmt(bundle.acc_mean.max())}%），"
        f"最低的是 {bundle.acc_mean.idxmin()}（{_fmt(bundle.acc_mean.min())}%），"
        f"极差 {_fmt(bundle.acc_mean.max() - bundle.acc_mean.min())} pp。",
        f"FA 偏差最小的是 {bundle.fa_bias.idxmin()}（{_fmt(bundle.fa_bias.min(), 4)} pp），"
        f"最大的是 {bundle.fa_bias.idxmax()}（{_fmt(bundle.fa_bias.max(), 4)} pp）。",
        f"频谱对数 RMS×100 最小的是 {bundle.spectrum_rms.idxmin()}"
        f"（{_fmt(bundle.spectrum_rms.min(), 2)}），说明其小尺度能量分布最接近观测。",
        f"平均排名分最低（最好）的是 {bundle.rank_score.idxmin()}"
        f"（{_fmt(bundle.rank_score.min(), 1)}），最高的是 {bundle.rank_score.idxmax()}"
        f"（{_fmt(bundle.rank_score.max(), 1)}）。",
        f"{bundle.anomaly_variable} 的量级差异是本批次最需要先核实的一项"
        f"（{bundle.anomaly_model} 为其余模型中位数的 {_fmt(bundle.anomaly_ratio, 2)} 倍），"
        f"在单位与定义确认前，该变量相关结论只作参考。",
    ]
    lines += _bullets(items)
    return lines


def _section_selection(bundle: Bundle) -> List[str]:
    lines = ["# 11. 最终选型建议", "", "**分场景选型建议**", ""]
    rows = [
        ["综合精度优先（默认）", str(bundle.relative_composite.idxmin()),
         f"综合相对 RMSE 最低（{_fmt(bundle.relative_composite.min())}）"],
        ["大尺度形势场", str(bundle.acc_mean.idxmax()),
         f"ACC 最高（{_fmt(bundle.acc_mean.max())}%）"],
        ["强度与活跃度", str(bundle.fa_bias.idxmin()),
         f"FA 偏差最小（{_fmt(bundle.fa_bias.min(), 4)} pp）"],
        ["小尺度结构与谱保真", str(bundle.spectrum_rms.idxmin()),
         f"频谱对数 RMS×100 最小（{_fmt(bundle.spectrum_rms.min(), 2)}）"],
    ]
    if bundle.rank_score.notna().any():
        rows.append([
            "多维度均衡", str(bundle.rank_score.idxmin()),
            f"平均排名分最低（{_fmt(bundle.rank_score.min(), 1)}）",
        ])
    rows.append([
        "暂缓采用", str(bundle.relative_composite.idxmax()),
        f"综合相对 RMSE 最高（{_fmt(bundle.relative_composite.max())}）",
    ])
    lines += _table(["**应用目标**", "**推荐模型**", "**理由**"], rows)
    return lines


def _season_block() -> List[str]:
    """「附 S 分季节结果（可选）」：固定输出「本批未出」的静态说明。

    这一块**不预生成**：长表里没有季节列，季节要从 ``init_time`` 现推，而分季节
    对比是按需的（用户点名要哪一季、哪个纬度带才切）。口径 2026-09-22 已定，
    工序写在 ``references/rmse-batch-evaluation.md`` §6；这里只留节位与指路，
    **纯静态文本、不含任何占位符**，与 ``weather_rmse_single.md`` 的「附 S」逐字一致。
    """
    return [
        "## 附 S 分季节结果（可选）",
        "",
        "**本块本批未出。** 产物长表里没有季节维度（只有 `init_time`），季节要现推；",
        "分季节对比是**按需补**的——用户点名要哪一季、哪个纬度带，就临时切一份出来，",
        "**不需要重跑评测**。渲染器不预生成这一块，也不静默省略、不留空表。",
        "",
        "两条口径 2026-09-22 已定，照做即可（完整工序见",
        "`references/rmse-batch-evaluation.md` §6「按需子集对比」）：",
        "",
        "- 季节按**起报时刻**（`init_time`）切，**不按** `valid_time`——报告整套契约建立在",
        "  「共同日期 = 共同 `init_date`」上，一组 `init_date` 必须干净地属于一个季节；",
        "- DJF 按**气象冬季**归组（当年 12 月 + 次年 1、2 月）。不满整年的批次两个冬季",
        "  **各只有一截**（实测 `20250102–20251215`：DJF(2024/25) 只有 1/2–2/28，",
        "  DJF(2025/26) 只有 12/1–12/15），所以报 DJF 要说清是**哪一个冬季**，或者干脆",
        "  只出 MAM/JJA/SON；**不要**把同一自然年的 12 月拼进去充数。",
        "",
        "补数之后这一块的形态与「附 L」一致：一张分季节表（一行一个季节，按 DJF、MAM、JJA、",
        "SON 升序，各模型各占一列，末列 `Best`；基线在**子集内**重算并写在表下），加一条",
        "带天数的判读句。`season_summary.png`（`图 S1`）与 `season_rmse_vs_lead.png`（`图 S2`）",
        "是可选图——**没出图就别留号**，不指不存在的文件。",
    ]


def _best_names(series: pd.Series, digits: int = 3, *, mode: str = "min") -> List[str]:
    """并列最优的**全体**名字（``mode="max"`` 改成取最大）。

    两个模型出自同一批产物时相对值会精确相等，只取 ``idxmin`` / ``idxmax``
    挑中的那一个，报告读起来就是「A 全面领先」，而事实是分不出高下。按**显示精度**
    判并列，保证表里印出来的数字和这一列说的是同一件事。
    """
    finite = series.dropna()
    if finite.empty:
        return []
    target = finite.max() if mode == "max" else finite.min()
    key = f"{float(target):.{digits}f}"
    return [
        str(name) for name, value in finite.items() if f"{float(value):.{digits}f}" == key
    ]


def _best_cell(series: pd.Series, digits: int = 3, *, mode: str = "min") -> str:
    """附表「Best」/「更接近1」/「更优」列：并列者逐个列出，不只报挑中的那个。"""
    names = _best_names(series, digits, mode=mode)
    if not names:
        return "—"
    return " / ".join(names) + ("（并列）" if len(names) > 1 else "")


class RegionTable(NamedTuple):
    """「分纬度带综合相对RMSE」那张表的三件套，外加全球口径的冠军。"""

    rows: List[List[str]]                 # 表体（不含表头），一行一个带
    best_of: Dict[str, List[str]]         # 带名 -> 并列第一的全体模型名
    levels: Dict[str, Dict[str, float]]   # 带名 -> 变量 -> 该带该变量平均 RMSE
    best_global: List[str]                # 全球口径并列第一的全体模型名


def region_table(bundle: Bundle) -> RegionTable:
    """构造「分纬度带综合相对RMSE」表。

    det / ens / wave 三份报告的「附 L」第一张表完全同构，共用这一个构造函数——
    各写一遍的话，并列判定之类的修订迟早只在其中一份上生效。
    """
    rows: List[List[str]] = []
    best_of: Dict[str, List[str]] = {}
    levels: Dict[str, Dict[str, float]] = {}
    for region in bundle.region_names:
        series = bundle.region_relative_composite.loc[region]
        best_of[region] = _best_names(series)
        rows.append(
            [region_display(region, bundle.region_labels)]
            + [_fmt(series[name]) for name in bundle.names]
            + [_best_cell(series)]
        )
        # index=(模型, lead)、columns=变量 → 按变量对「模型 × lead」平均，
        # 得到「这个带里这个变量的 RMSE 量级」。**不跨变量平均**，见 _region_reading。
        stacked = pd.concat(bundle.region_rmse_by_lead[region].values())
        levels[region] = {str(key): float(value) for key, value in stacked.mean(axis=0).items()}
    return RegionTable(rows, best_of, levels, _best_names(bundle.relative_composite))


def _region_reading(
    bundle: Bundle,
    best_global: Mapping[str, Any],
    best_of: Mapping[str, Mapping[str, Any]],
    levels: Mapping[str, Mapping[str, float]],
) -> str:
    """分带判读句：点名名次反转的带 + 带间绝对量级差。

    ``levels`` 是 ``{带名: {变量: 该带该变量的 RMSE}}``。量级差**逐变量算比值再取
    中位数**，不跨变量平均 RMSE——不同变量单位不同（``z500`` 是 m²/s²、``t2m`` 是 K），
    先平均再比大小等于把量纲加在一起，出来的倍数没有意义。
    """
    global_names = list(best_global)
    tied = len(global_names) > 1
    head_global = "、".join(global_names) if global_names else "—"
    tie_note = "（并列，分不出高下）" if tied else ""
    global_value = _fmt(bundle.relative_composite[global_names[0]]) if global_names else "—"

    flips = [
        (region, list(best_of[region]))
        for region in bundle.region_names
        if set(best_of[region]) != set(global_names)
    ]
    if flips:
        detail = "；".join(
            f"{region_display(region, bundle.region_labels)} 的第一名是 {'、'.join(names)}"
            f"（{_fmt(bundle.region_relative_composite.loc[region, names[0]])}）"
            for region, names in flips
        )
        head = (
            f"全球口径下综合相对RMSE最低的是 {head_global}{tie_note}（全球 {global_value}），"
            f"但分带看这个结论并不是处处成立：{detail}——优势不是全纬度一致的，"
            f"选型时要写清用在哪一纬带。"
        )
    elif tied:
        head = (
            f"各带的第一名与全球口径一致，都是 {head_global}{tie_note}"
            f"（全球 {global_value}）——全球与各带都分不出高下，"
            f"谈不上哪个模型在某一纬带上占优。"
        )
    else:
        head = (
            f"各带的第一名与全球口径一致，都是 {head_global}"
            f"（全球 {global_value}）——它的优势在全纬度上都成立。"
        )

    variables = sorted({v for per_band in levels.values() for v in per_band})
    spreads: List[float] = []
    high_votes: Dict[str, int] = {}
    low_votes: Dict[str, int] = {}
    for variable in variables:
        finite = {
            region: per_band[variable]
            for region, per_band in levels.items()
            if np.isfinite(per_band.get(variable, np.nan)) and per_band.get(variable, 0) > 0
        }
        if len(finite) < 2:
            continue
        high = max(finite, key=lambda key: finite[key])
        low = min(finite, key=lambda key: finite[key])
        high_votes[high] = high_votes.get(high, 0) + 1
        low_votes[low] = low_votes.get(low, 0) + 1
        spreads.append(finite[high] / finite[low])
    if spreads:
        high_band = max(high_votes, key=lambda key: high_votes[key])
        low_band = max(low_votes, key=lambda key: low_votes[key])
        head += (
            f" 各带的绝对 RMSE 量级差（同一变量在各带之间的极差，"
            f"{len(spreads)}个变量的中位数）是 {_fmt(float(np.median(spreads)), 1)} 倍，"
            f"最大出现在 {region_display(high_band, bundle.region_labels)}、"
            f"最小出现在 {region_display(low_band, bundle.region_labels)}"
            f"——带与带之间只能比相对值，比绝对值没有意义。"
        )
    return head


def _region_block(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    """「附 L 分纬度带结果（可选）」。

    没有共同的 region 行时保留节位、写一句「本批未出」并说明原因，不静默省略、
    也不留一张空表——与球谐带那三节同一套约定。
    """
    count = len(bundle.names)
    word = _model_count_word(count)
    lines = ["## 附 L 分纬度带结果（可选）", ""]
    if not bundle.region_names:
        lines += [f"**本块本批未出。** {bundle.region_note}", ""]
        return lines

    table = region_table(bundle)
    rows, best_of, levels, best_global = (
        table.rows, table.best_of, table.levels, table.best_global,
    )

    lines += [
        "本块把长表里 `region` 非空的行**单独汇总**，全球行**不参与**——全球平均与纬度带平均",
        "是两个量，混在一起算会把带间差异整个抹平，这一块也就没有意义了。",
        "带名与边界照搬评测配置 `options[\"regions\"]`：**报告不翻译、也不写死任何带名**，",
        "配置里叫 `mid_latitudes` 就显示 `mid_latitudes`；要换一套切法改配置重跑即可，",
        "报告这边一个字都不用动。表里的相对RMSE是**对本带基线**取的（100 = 该带内的",
        f"{word}模型几何均值），不是全球基线——热带和极区的绝对 RMSE 差一个量级，",
        "不除本带基线就没法横向比。",
        "",
        "**分纬度带综合相对RMSE**",
        "",
    ]
    lines += _table(["纬度带"] + list(bundle.names) + ["Best"], rows)
    lines += ["", _region_reading(bundle, best_global, best_of, levels), ""]
    lines += [
        f"![lat_band_rmse_vs_lead]({figures['lat_band_lead']})",
        "",
        f"图 L1：各纬度带综合相对RMSE随预报时效的变化"
        f"（100 = 该带内{word}模型几何均值；{_date_span(bundle.dates)} 共同日期平均）",
        "",
        f"![lat_band_summary]({figures['lat_band_summary']})",
        "",
        f"图 L2：分纬度带综合相对RMSE"
        f"（100 = 该带内{word}模型几何均值；{_date_span(bundle.dates)} 共同日期平均）",
    ]
    return lines


def _heatmap_block(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    """「附 5 变量 × 时效 RMSE 热力图」（``图 5``）。"""
    if "rmse_heatmap" not in figures:
        return [
            "## 附 5 变量 × 时效 RMSE 热力图",
            "",
            "**本批未出。** 只有单个 lead 时热力图没有可读的横向变化，"
            "渲染器不出这张图。",
        ]
    count = len(bundle.names)
    variables = len(bundle.common_variables)
    first, last, leads = _lead_span(bundle.rmse_by_lead[bundle.names[0]].index)
    return [
        "## 附 5 变量 × 时效 RMSE 热力图",
        "",
        f"{count}幅子图，每个模型一幅：纵轴是{variables}个变量、横轴是全部{leads}个lead",
        f"（{first}–{last}h）。格子里的数**不是 RMSE 本身，而是该模型在该变量、该 lead 上",
        f"相对自己首时效（{first}h）的 RMSE 倍数**——除以首时效就把变量的量纲与气候态差异",
        f"约掉了，{variables}个变量才能摆在同一根色标下横向比。",
        "读法：同一行里颜色随列单调加深是正常的，**加深得慢**才说明这个变量扛得住长时效；",
        "同一行里某一段突然跳深，通常不是模式变差，而是该时效上样本数或变量路由变了，",
        "要回第 2 节核对样本。",
        "",
        f"![multi_model_rmse_heatmap]({figures['rmse_heatmap']})",
        "",
        f"图 5：{variables}个变量 × {leads}个 lead 的 RMSE 倍数热力图"
        f"（各自除以本模型首时效 {first}h 的 RMSE；{_date_span(bundle.dates)} 共同日期平均）",
    ]


def _single_init_block(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    """「附 6 单起报功率谱曲线」（``图 6``）。"""
    count = len(bundle.dates)
    init = str(bundle.dates[0]) if bundle.dates else ""
    lines = [
        f"## 附 6 单起报 {bundle.spectrum_variable} 功率谱曲线",
        "",
        f"附 4 是{count}个日期平均后的谱，平均会把个例差异抹平。这一节换成**单个起报**",
        f"（{init}）的谱，用来核对平均谱上的结论在个例上是否成立——平均谱上「小尺度偏低」",
        "如果只在少数个例出现，就不该写成模式的普遍特征。",
        "",
    ]
    if "spectrum_curve" in figures:
        lines += [
            f"![spectrum_curve]({figures['spectrum_curve']})",
            "",
            f"图 6：{init} 单起报的 {bundle.spectrum_variable} 纬向功率谱"
            f"（双对数；黑色虚线为同时刻观测谱）",
        ]
    else:
        lines += [
            f"**本批未出。** 产物里没有 {bundle.spectrum_variable} 的逐起报谱"
            f"（`diagnostics/spectrum_by_init.csv`），出不了这张图。",
        ]
    return lines


def _appendix(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    count = len(bundle.names)
    listing = "、".join(bundle.names)
    first, last, leads = _lead_span(bundle.rmse_by_lead[bundle.names[0]].index)
    lines = [
        f"# 附录：{count}模型曲线图",
        "",
        f"本附录使用{count}个模型共同的{len(bundle.dates)}个日期"
        f"（{bundle.dates[0]} 至 {bundle.dates[-1]}）。"
        f"以下每张图均包含 {listing} {count} 条模型曲线。"
        f"正文统计表与图中曲线使用同一批共同日期。",
        "",
        f"## 附 1 RMSE：{len(bundle.common_variables)}个变量{count}模型曲线",
        "",
        f"每幅子图包含 {listing} {count} 条曲线。",
        "",
        f"![multi_model_rmse_vs_lead]({figures['rmse_plot']})",
        "",
        f"图 1：{len(bundle.common_variables)}个变量 RMSE 随预报时效的变化（{_date_span(bundle.dates)} 共同日期平均）",
        "",
        f"## 附 2 {bundle.acc_variable} ACC：{count}模型曲线",
        "",
        f"{count}条曲线对应{count}个模型，纵轴越高越好。",
        "",
        f"![multi_model_acc_vs_lead]({figures['acc_plot']})",
        "",
        f"图 2：{bundle.acc_variable} ACC 随预报时效的变化（{_date_span(bundle.dates)} 共同日期平均）",
        "",
        f"## 附 3 FA（Forecast Activity）：{len(bundle.fa_variables)}个变量、全程{first}–{last}h {count}模型曲线",
        "",
        f"每个子图均包含{count}条模型曲线，并覆盖全部{leads}个lead（{first}–{last}h）；"
        f"ratio越接近1越好。",
        "",
        f"![multi_model_fa_ratio_vs_lead]({figures['fa_plot']})",
        "",
        f"图 3：{len(bundle.fa_variables)}个变量 FA ratio 随预报时效的变化（{_date_span(bundle.dates)} 共同日期平均）",
        "",
        f"## 附 4 {bundle.spectrum_variable} 功率谱：{count}模型曲线",
        "",
        f"双对数坐标；{count}条颜色曲线对应{count}个模型的预测谱。观测谱为黑色虚线。",
        "",
        f"![multi_model_spectrum]({figures['spectrum_plot']})",
        "",
        f"图 4：{bundle.spectrum_variable} 纬向功率谱（{_date_span(bundle.dates)} 共同日期平均）",
        "",
    ]
    lines += _heatmap_block(bundle, figures)
    lines += [""]
    lines += _single_init_block(bundle, figures)
    lines += [""]
    lines += _capability_block(bundle, figures)
    lines += [""]
    lines += _region_block(bundle, figures)
    lines += [""]
    lines += _season_block()
    return lines


#: 能力雷达图的轴：``(轴名, 取自 Bundle 的字段, 是否越高越好)``。
#:
#: 四根都是**综合评分**——单一标量、衡量一种能力、可跨模型比。逐变量 / 逐时效的
#: 细粒度指标**不进雷达图**：雷达图一根轴只画一个顶点，细粒度指标要么先汇总
#: （那已经变成另一个综合评分），要么把轴撑爆、读不出形状。
CAPABILITY_AXES: Tuple[Tuple[str, str, bool], ...] = (
    ("综合相对RMSE", "relative_composite", False),
    ("ACC (%)", "acc_mean", True),
    ("FA偏差 (pp)", "fa_bias", False),
    ("频谱对数RMS", "spectrum_rms", False),
)


def capability_profile(
    bundle: Bundle,
) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, bool]]:
    """组装雷达图要的三样东西：轴序、``{模型: {轴: 原始值}}``、各轴方向。"""
    axes = [label for label, _, _ in CAPABILITY_AXES]
    higher = {label: flag for label, _, flag in CAPABILITY_AXES}
    raw = {
        str(name): {
            label: getattr(bundle, field)[name] for label, field, _ in CAPABILITY_AXES
        }
        for name in bundle.names
    }
    return axes, raw, higher


def _capability_block(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    """「附 模型能力雷达图」——综合评分的批内归一化对比（``图 7``）。"""
    axes, raw, higher = capability_profile(bundle)
    scores = radar_scores(raw, axes, higher)
    names = [str(name) for name in bundle.names]
    kept = [axis for axis in axes if names and axis in scores[names[0]]]
    dropped = [axis for axis in axes if axis not in kept]

    lines = [
        "## 附 7 模型能力雷达图",
        "",
        "这一节把正文第 3 节那张总表的**综合评分**摆成多边形：每根轴一种能力，",
        "每个模型一个多边形，**越靠外越好**。它不引入任何新算法，只是把总表的数",
        "换个读法——看的是「能力形状」，不是「谁排第一」：两个模型综合分接近时，",
        "雷达图能显出差距集中在哪几项上。",
        "",
        f"共 {len(kept)} 根轴，方向已在括号里标明（越高越好 / 越低越好）："
        + "；".join(
            f"{axis}（{'越高越好' if higher[axis] else '越低越好'}）" for axis in kept
        )
        + "。",
        "",
    ]

    if dropped:
        lines += [
            f"**有 {len(dropped)} 根轴本批未画**：{'、'.join(dropped)}——"
            "这些轴上有模型缺值。雷达图没有断点，少画一个顶点会把多边形拉歪，"
            "所以整根剔除，不拿 0 顶替（那是「这项能力为零」，不是「没有这项能力」）。",
            "",
        ]

    lines += [
        "**读法**：纵轴 0–1 是**批内相对分**，每根轴上 1 = 本批最好、0 = 本批最差"
        "（各轴方向先行统一）。"
        f"这是{_model_count_word(len(names))}模型互比出来的相对位置，**不是绝对能力分**——"
        "全批都差时每根轴照样有人拿 1；只有两个模型时必然是 1 和 0。"
        "某一根轴上所有模型打平时，该轴一律画在满格 1.0（画成 0 会让多边形凭空凹进去），"
        "所以**满格不代表领先**，要回正文看绝对数值。",
        "",
    ]

    if "capability_radar" in figures:
        lines += [
            f"![capability_radar]({figures['capability_radar']})",
            "",
            f"图 7：{_model_count_word(len(names))}模型能力雷达图（{len(kept)} 根综合评分轴；"
            f"各轴批内 min-max 归一化，1 = 本批最好；{_date_span(bundle.dates)} 共同日期）",
        ]
    else:
        lines += [
            "**本批未出。** 渲染器没拿到雷达图的落盘路径。这一节**不像别处那样"
            "依赖可选数据**——综合评分四根轴每个批次都有，正常跑 "
            "`generate_det_report.py` 必然出图；缺了就是渲染器或调用方出了问题，"
            "该回去查，不要当成「这批数据没跑到」。",
        ]

    if scores and names:
        outer = max(names, key=lambda name: sum(scores[name].values()))
        inner = min(names, key=lambda name: sum(scores[name].values()))
        if outer != inner:
            lines += [
                "",
                f"多边形面积最大的是 {outer}（各轴得分之和 "
                f"{_fmt(sum(scores[outer].values()), 2)}），"
                f"最小的是 {inner}（{_fmt(sum(scores[inner].values()), 2)}）。",
            ]
    return lines


# ------------------------------------------------------------------ 装配


def report_date() -> str:
    """报告**生成**日期（``YYYYMMDD``）。

    抬头那个「报告日期」是跑这份报告的当天，**不是数据末日期**——两者很容易
    被看成一回事，但含义完全不同：数据覆盖到哪天，正文第 1/2 节已经写了
    「覆盖 N 天（起，止）」，抬头再写一遍等于把同一个信息抄两处，还挤掉了
    「这份报告是什么时候出的」这个只有抬头能承载的信息。

    代价是产物**不再逐位可复现**：同一天跑两次一样，隔天跑就不同。
    """
    return datetime.now().strftime("%Y%m%d")


def render(
    bundle: Bundle,
    *,
    title: Optional[str] = None,
    change: Optional[str] = None,
    archive: Optional[str] = None,
    declared: Optional[Mapping[str, Sequence[str]]] = None,
    figures: Optional[Mapping[str, str]] = None,
) -> str:
    """把 :class:`Bundle` 渲染成完整 Markdown。"""
    count = len(bundle.names)
    figures = figures or {
        "rmse_plot": "multi_model_rmse_vs_lead.png",
        "acc_plot": "multi_model_acc_vs_lead.png",
        "fa_plot": "multi_model_fa_ratio_vs_lead.png",
        "spectrum_plot": f"multi_model_spectrum_{bundle.spectrum_variable}.png",
        "lat_band_lead": "lat_band_rmse_vs_lead.png",
        "lat_band_summary": "lat_band_summary.png",
    }
    first, last, leads = _lead_span(bundle.rmse_by_lead[bundle.names[0]].index)

    lines: List[str] = []
    lines.append(title or f"# 确定性预报{_model_count_word(count)}模型综合评估报告")
    lines += ["", f"**产物：{archive or '、'.join(bundle.names)}**", ""]
    lines += [
        f"报告日期：{report_date()}    评估层级：输出级复核",
        "",
    ]
    lines += _conclusion_summary(bundle, change, first=first, last=last, leads=leads)
    lines += [""]
    lines += _section_samples(bundle, declared)
    lines += [""]
    lines += _section_overall(bundle)
    lines += [""]
    lines += _section_rmse(bundle)
    lines += [""]
    lines += _section_acc_fa_spectrum(bundle)
    lines += [""]
    lines += _section_risks(bundle, declared)
    lines += [""]
    lines += _section_acceptance(bundle)
    lines += [""]
    lines += _section_comparison(bundle)
    lines += [""]
    lines += _section_sensitivity(bundle)
    lines += [""]
    lines += _section_conclusion(bundle)
    lines += [""]
    lines += _section_selection(bundle)
    lines += [""]
    lines += _appendix(bundle, figures)
    lines.append("")
    return "\n".join(lines)


def _conclusion_summary(
    bundle: Bundle,
    change: Optional[str],
    *,
    first: float,
    last: float,
    leads: int,
) -> List[str]:
    order = bundle.relative_composite.sort_values().index.tolist()
    best, worst = order[0], order[-1]
    items = [
        f"综合相对 RMSE 最好 {best}（{_fmt(bundle.relative_composite[best])}），"
        f"最差 {worst}（{_fmt(bundle.relative_composite[worst])}）。",
        f"ACC 最好 {bundle.acc_mean.idxmax()}（{_fmt(bundle.acc_mean.max())}%），"
        f"FA 偏差最小 {bundle.fa_bias.idxmin()}（{_fmt(bundle.fa_bias.min(), 4)} pp），"
        f"频谱对数 RMS×100 最小 {bundle.spectrum_rms.idxmin()}（{_fmt(bundle.spectrum_rms.min(), 2)}）。",
        f"平均排名分："
        + "、".join(f"{name} {_fmt(bundle.rank_score[name], 1)}" for name in bundle.rank_score.sort_values().index)
        + "。",
    ]
    significant = [t for t in bundle.overall_tests if t.significant]
    if significant:
        items.append(
            "综合相对 RMSE 上通过 Holm 校正的显著差异："
            # 方向只能来自检验结论（t.better），不能拿参数顺序 t.a/t.b 当优劣——
            # 否则 B 更优时会写成「A 优于 B」，和第三节的表直接打架。
            + "；".join(
                f"{t.better} 优于 {t.b if t.better == t.a else t.a}（p={_fmt(t.p_holm, 4)}）"
                for t in significant
            )
            + "。"
        )
    else:
        items.append(
            "综合相对 RMSE 上没有任何模型对通过 Holm 校正（p < 0.05），"
            "模型间的总体差距在统计上不稳健，选型应更依赖分维度表现。"
        )
    items.append(
        f"{bundle.anomaly_variable} 存在模型间量级差异"
        f"（{bundle.anomaly_model} 为其余模型中位数的 {_fmt(bundle.anomaly_ratio, 2)} 倍），"
        f"已在 6.1 与第 9 节单独分析。"
    )
    return [
        "# 1. 结论摘要",
        "",
        # 先交代评估基本信息（评的是谁、多少天、什么指标），再给结论
        f"本报告评估{'、'.join(bundle.names)}共{len(bundle.names)}个模型，"
        f"覆盖{len(bundle.dates)}天（{_date_span(bundle.dates)}）；"
        f"口径为RMSE、ACC、FA与纬向谱。正文与附录统一使用同一批共同日期，"
        f"逐日结果按 lead 平均后比较，非共同日期不参与任何统计。"
        + (f"本次相对上一版的改动：{change}" if change else ""),
        "",
        # 结论并成一段而不是逐条列点：读起来是判断，不是清单
        "".join(items),
        "",
        f"本报告中的FA指Forecast Activity，统一使用全程{first}–{last}h、"
        f"{leads}个lead，不对FA做短中长时效分段。",
        "",
    ]


# ------------------------------------------------------------------ 入口


def build_det_report(
    archives: Sequence[Archive],
    out_dir: Path,
    *,
    title: Optional[str] = None,
    change: Optional[str] = None,
    archive: Optional[str] = None,
    declared: Optional[Mapping[str, Sequence[str]]] = None,
    ncols: int = 4,
) -> Dict[str, Path]:
    """出图 + 写 ``REPORT.md``，返回产物名 → 路径。

    Raises:
        ValueError / FileNotFoundError: 见 :func:`analyze`。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = analyze(archives, declared=declared)
    plotter = DetPlotter()

    figures = {
        "rmse_plot": "multi_model_rmse_vs_lead.png",
        "acc_plot": "multi_model_acc_vs_lead.png",
        "fa_plot": "multi_model_fa_ratio_vs_lead.png",
        "spectrum_plot": f"multi_model_spectrum_{bundle.spectrum_variable}.png",
    }

    rmse_series = {name: bundle.rmse_by_lead[name] for name in bundle.names}
    plotter.plot_model_panels(
        rmse_series, bundle.common_variables, ylabel="RMSE",
        ncols=ncols, suptitle=f"{len(bundle.names)}模型 RMSE 随预报时效的变化",
        save_path=out_dir / figures["rmse_plot"],
    )

    acc_series = {
        name: bundle.acc_by_lead[name].reindex(columns=[bundle.acc_variable])
        for name in bundle.names
    }
    plotter.plot_model_panels(
        acc_series, [bundle.acc_variable], ylabel="ACC", ncols=1,
        suptitle=f"{bundle.acc_variable} ACC 随预报时效的变化（越高越好）",
        save_path=out_dir / figures["acc_plot"],
    )

    plotter.plot_model_panels(
        {name: bundle.fa_ratio_by_lead[name] for name in bundle.names},
        bundle.fa_variables, ylabel="ratio（预报/观测）", ref_line=1.0, ncols=ncols,
        suptitle="FA（Forecast Activity）ratio 随预报时效的变化（越接近 1 越好）",
        save_path=out_dir / figures["fa_plot"],
    )

    plotter.plot_model_spectrum(
        bundle.spectrum, bundle.spectrum_variable,
        save_path=out_dir / figures["spectrum_plot"],
    )

    # 图 5：变量 × 时效 RMSE 热力图。格子是「除以本模型首时效」的倍数——各变量单位
    # 不同，不归一化就没法共用一根色标。只有一个 lead 时没有横向变化可读，不出图。
    ratios: Dict[str, pd.DataFrame] = {}
    for name in bundle.names:
        frame = bundle.rmse_by_lead[name]
        if frame.empty:
            continue
        leads = sorted(frame.index)
        if len(leads) < 2:
            continue
        base_lead = frame.loc[leads[0]]
        ratios[name] = frame.div(base_lead.replace(0.0, np.nan))
    if ratios:
        figures["rmse_heatmap"] = "multi_model_rmse_heatmap.png"
        plotter.plot_rmse_heatmap(
            ratios, bundle.common_variables,
            leads=sorted(bundle.rmse_by_lead[bundle.names[0]].index),
            suptitle=f"{_model_count_word(len(bundle.names))}模型 RMSE 相对本模型首时效的倍数",
            save_path=out_dir / figures["rmse_heatmap"],
        )

    # 图 7：单起报谱曲线。取共同日期里最早的那个起报，其余日期仍进平均谱（附 4）。
    # 产物没有逐起报谱就**不出图**，报告那边写「本批未出」。
    if bundle.dates:
        init = str(bundle.dates[0])
        single: Dict[str, pd.DataFrame] = {}
        # 循环变量**不能**叫 archive——那是本函数的参数名（`--archive` 传进来的归档名），
        # 遮蔽掉之后 render 拿到的是 Archive 对象，抬头会打印整个 repr。
        for item in archives:
            try:
                pred, obs = load_spectrum(item.root, [init], bundle.spectrum_variable,
                                          item.name)
            except (FileNotFoundError, ValueError):
                continue
            if init in pred.columns:
                single[item.name] = pd.DataFrame({"pred": pred[init], "obs": obs[init]})
        if single:
            figures["spectrum_curve"] = f"spectrum_curve_{bundle.spectrum_variable}_{init}.png"
            plotter.plot_model_spectrum(
                single, bundle.spectrum_variable,
                suptitle=f"{bundle.spectrum_variable} 纬向功率谱（{init} 单起报）",
                save_path=out_dir / figures["spectrum_curve"],
            )

    # 分纬度带（可选块）：产物里没有共同的 region 行就**不出图**，也不往 artifacts 里
    # 塞不存在的路径——报告那边会写「本批未出」，指一张没生成的图比不指更糟。
    if bundle.region_names:
        band_labels = [
            region_display(region, bundle.region_labels) for region in bundle.region_names
        ]
        figures["lat_band_lead"] = "lat_band_rmse_vs_lead.png"
        figures["lat_band_summary"] = "lat_band_summary.png"
        plotter.plot_model_panels(
            {
                name: pd.DataFrame({
                    label: bundle.region_lead_relative[region][name]
                    for region, label in zip(bundle.region_names, band_labels)
                })
                for name in bundle.names
            },
            band_labels,
            ylabel="综合相对 RMSE（100 = 该带内几何均值）",
            ref_line=BASELINE,
            ncols=min(len(band_labels), 3),
            suptitle="分纬度带综合相对 RMSE 随预报时效的变化",
            save_path=out_dir / figures["lat_band_lead"],
        )
        summary = bundle.region_relative_composite.copy()
        summary.index = band_labels
        plotter.plot_group_bars(
            summary,
            ylabel="综合相对 RMSE（100 = 该带内几何均值）",
            ref_line=BASELINE,
            suptitle="分纬度带综合相对 RMSE",
            save_path=out_dir / figures["lat_band_summary"],
        )

    # 图 7：能力雷达图。轴与归一化都在渲染器那边（`capability_profile` +
    # `radar_scores`），这里只负责出图——两边各算一遍会把图与图注算出分歧。
    figures["capability_radar"] = "capability_radar.png"
    axes, raw, higher = capability_profile(bundle)
    plotter.plot_capability_radar(
        raw, axes, higher,
        suptitle=f"{_model_count_word(len(bundle.names))}模型能力雷达图"
                 f"（批内相对分，1 = 本批最好）",
        save_path=out_dir / figures["capability_radar"],
    )

    text = render(
        bundle, title=title, change=change, archive=archive,
        declared=declared, figures=figures,
    )
    report_path = out_dir / "REPORT.md"
    report_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"已保存: {report_path}")

    artifacts: Dict[str, Path] = {"report": report_path}
    for key, filename in figures.items():
        artifacts[key] = out_dir / filename
    return artifacts
