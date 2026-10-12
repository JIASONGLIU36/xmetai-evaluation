#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS 结果表 → Markdown 报告。

一句话用法::

    python -m xmetai_evaluation.visualization.ts_report --csv ts.csv --result reports/fgvp --model FGVP

报告按**论文实验章节**的写法组织：每一节都是「引导句 → 图表 → 定量分析」。
图就地带着图注出现（`图 N：…` / `表 N：…`，按出现顺序编号），图下必有图注、
表上必有表题，每节末尾有一段**小结**——小结里的每个论断都带具体数值，由
``analysis_*`` 系列函数从已经算好的统计量里推出来，不是套话。

章节：结论摘要（含评估对象与口径）→ 逐等级表现 → 随时效变化 → 误差形态与偏差结构 →
跨时效分布（箱线图）→（可选，集合）概率评分 AROC/BSS →（可选）与对比模型的比较 →
（多模型）模型能力雷达图 → 改进建议 → 口径与注意事项。

**概率评分那一节只在拿到概率宽表时出现**（``diagnostics/probability_wide.csv``，
由 ``weather_ts_ens_prob`` 能力写出）。它的累积窗口与阈值都和 TS 不同，所以
单独立节、单独编号；它一出现，后面所有章节号整体后移一位。产物里没有这张表时
整节不出现，章节号原样——**编号跟着实际出现了什么走**，这是 ``_Numbering`` 的
同一条规矩。集合专属的措辞（标题的「集合预报」、第一节的「集合口径」段、
第十节的两条集合提示）则统一由 ``build_ts_report(ensemble=True)`` 打开，
默认关：确定性与 FDP 两条流程共用本模块，不能被牵连。

雷达图那一节把前面几张表的**综合评分**摆成多边形（批内 min-max，1 = 本批最好），
只在两个及以上模型时出现——一个多边形没有比较对象。「综合」是硬要求：
POD / FAR 和 TS 是恒等式关系，不能各占一根轴。

结论摘要**整段成文、不列点**：开头先把「评的是谁、多少天、什么指标」交代清楚，
再接一整段判断，末尾是业务定性与指标定义。

``analysis_*`` 函数只吃 ``summarize()`` / ``box_stats()`` 已经算好的数字，
不重新读表、不重新实现指标——口径只有一处，表、图、文字不可能各说各话。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: 第五节（跨时效分布）里逐字固定的三行。``assets/templates/weather_ts_single.md`` 的骨架要抄同一份字节，
#: 两边差一个全角/半角字符测试就会红——所以只在**这里**定义一次。
_BOX_HEADER = "| 模型 | 降水等级 | 中位数 | IQR | Q1–Q3 | 须线范围 | 离群时效 | 时效数 |"
_BOX_SEP = "|---|---|---|---|---|---|---|---|"
#: 概率评分宽表的必需列。缺任一列就不是这份表——**不猜、不补**（见 ``load_probability``）。
PROBABILITY_COLUMNS = (
    "lead_h", "grade", "threshold_mm", "AROC", "BS", "BS_ref", "BSS",
    "base_rate", "n_points",
)

_BOX_NOTE = (
    "> 每个箱体的样本是**同模型、同量级在全部时效上的 TS 取值**"
    "（无有效 TS 的时效不计入，样本数见「时效数」列）：箱体为四分位距（IQR），中线为中位数，"
    "须线延伸到 1.5×IQR 内最远的点，空心圆为离群时效。**TS 越高越优**，"
    "所以箱体**越高**表示该量级整体技巧越好、**越窄**表示跨时效越稳定；"
    "这里的统计量与第二节的「全时效平均」不是同一个量（等权平均 vs 中位数/IQR），不可互相替代。"
)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


class _Numbering:
    """图和表**按在报告里出现的顺序**编号，图注/表题由它生成。

    编号跟着出现顺序走而不是写死，是因为图会缺：某份结果表没有 POD/FAR 就没有
    性能图，单模型分支没有 multi_model_TS_delta。缺图时**连图注一起不出现**，
    序号仍然连续（图 1、图 2…），不会出现「图 3 不见了」这种断号。
    """

    def __init__(self) -> None:
        self._figures: Dict[str, int] = {}
        self._tables = 0

    def figure(self, name: str) -> int:
        number = len(self._figures) + 1
        self._figures[name] = number
        return number

    def table(self) -> int:
        self._tables += 1
        return self._tables


def _figure_block(
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    name: str,
    caption: str,
) -> List[str]:
    """图片 + 图注。产物里没有这张图（或不是图片）时返回空列表，编号也不占位。"""
    path = (artifacts or {}).get(name)
    if path is None or Path(path).suffix.lower() not in (".png", ".jpg", ".svg"):
        return []
    number = numbering.figure(name)
    return [f"![{name}]({Path(path).name})", "", f"图 {number}：{caption}", ""]


def _table_caption(numbering: _Numbering, text: str) -> List[str]:
    """表题**在表上方**（图注在下方）——这是中文论文的排版惯例。"""
    return [f"表 {numbering.table()}：{text}", ""]


def _grade_sort_key(grade: str, threshold: Optional[float]) -> float:
    if threshold is not None and np.isfinite(threshold):
        return float(threshold)
    text = str(grade).replace("≥", "").strip()
    try:
        return float(text)
    except ValueError:
        return 1e9


def summarize(data: pd.DataFrame) -> Dict[str, object]:
    """把 TS 表压成报告需要的统计量。"""
    grades: List[str] = []
    per_grade: List[Dict[str, object]] = []
    for grade, subset in data.groupby("grade", observed=True):
        thresholds = subset["threshold_mm"].dropna() if "threshold_mm" in subset else pd.Series(dtype=float)
        threshold = float(thresholds.iloc[0]) if len(thresholds) else None
        totals = {
            key: float(subset[key].sum()) if key in subset.columns else float("nan")
            for key in ("hits", "misses", "false_alarms", "n_pairs")
        }
        hits = totals["hits"]
        misses = totals["misses"]
        false_alarms = totals["false_alarms"]

        def _ratio(numerator: float, denominator: float) -> float:
            if not np.isfinite(denominator) or denominator <= 0:
                return float("nan")
            return numerator / denominator

        entry = {
            "grade": str(grade),
            "threshold": threshold,
            # 主口径：先合计列联表再算比率；计数与之一致（都是合计）
            "TS_sum": _ratio(hits, hits + misses + false_alarms),
            "POD_sum": _ratio(hits, hits + misses),
            "FAR_sum": _ratio(false_alarms, hits + false_alarms),
            "漏报率_sum": _ratio(misses, hits + misses),
            "BIAS_sum": _ratio(hits + false_alarms, hits + misses),
            # 参考口径：各时效先算比率再等权平均
            "TS_mean": float(subset["TS"].mean()),
            "TS_last": float(subset.sort_values("lead_h")["TS"].iloc[-1]),
            "hits": int(hits) if np.isfinite(hits) else 0,
            "misses": int(misses) if np.isfinite(misses) else 0,
            "false_alarms": int(false_alarms) if np.isfinite(false_alarms) else 0,
            "n_pairs": int(totals["n_pairs"]) if np.isfinite(totals["n_pairs"]) else 0,
            "n_leads": int(subset["lead_h"].nunique()),
            "events": int(hits + misses) if np.isfinite(hits) else 0,
        }
        entry["TS"] = entry["TS_sum"]
        entry["POD"] = entry["POD_sum"]
        entry["FAR"] = entry["FAR_sum"]
        entry["漏报率"] = entry["漏报率_sum"]
        entry["BIAS"] = entry["BIAS_sum"]
        grades.append(str(grade))
        per_grade.append(entry)
    per_grade.sort(key=lambda item: _grade_sort_key(item["grade"], item["threshold"]))

    leads = sorted(float(value) for value in pd.unique(data["lead_h"]))
    first, last = leads[0], leads[-1]
    overall = float(np.nanmean([entry["TS_sum"] for entry in per_grade]))
    overall_mean = float(np.nanmean([entry["TS_mean"] for entry in per_grade]))
    weakest = per_grade[0] if per_grade else None
    strongest = per_grade[-1] if per_grade else None

    decay = None
    if weakest and strongest:
        first_ts = float(data[data["lead_h"] == first]["TS"].mean())
        last_ts = float(data[data["lead_h"] == last]["TS"].mean())
        decay = {
            "first_lead": first,
            "last_lead": last,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "drop": first_ts - last_ts,
            "relative": (first_ts - last_ts) / first_ts if first_ts else np.nan,
            "per_day": (first_ts - last_ts) / max((last - first) / 24.0, 1e-9),
        }

    pivot = data.pivot_table(index="lead_h", columns="grade", values="TS", observed=True)
    return {
        "grades": per_grade,
        "leads": leads,
        "overall_ts": overall,
        "overall_ts_mean": overall_mean,
        "decay": decay,
        "pivot": pivot,
        "n_rows": int(len(data)),
    }


def diagnose(summary: Dict[str, object]) -> List[str]:
    """规则化诊断：每条结论都带具体数值。"""
    grades = summary["grades"]
    findings: List[str] = []
    if not grades:
        return ["数据里没有可用的降水等级"]

    weak = grades[0]
    strong = grades[-1]
    overall = summary["overall_ts"]
    level = "较好" if overall >= 0.3 else ("中等" if overall >= 0.2 else "偏弱")
    findings.append(
        f"整体技巧 **{level}**：各等级全时效平均 TS = {_fmt(overall)}"
        f"（参考经验分级：≥0.3 较好 / 0.2–0.3 中等 / <0.2 偏弱）。"
    )

    if len(grades) > 1:
        findings.append(
            f"技巧随量级**单调衰减**：{weak['grade']}mm 的 TS {_fmt(weak['TS'])}"
            f" → {strong['grade']}mm 的 TS {_fmt(strong['TS'])}"
            f"（高量级命中数 {strong['hits']}，样本 {strong['n_pairs']}）。"
        )

    bias_entries = [entry for entry in grades if np.isfinite(entry.get("BIAS", np.nan))]
    if bias_entries:
        worst_bias = max(bias_entries, key=lambda entry: abs(np.log(max(entry["BIAS"], 1e-6))))
        direction = "系统性高报" if worst_bias["BIAS"] > 1 else "系统性漏报"
        findings.append(
            f"偏差结构：偏离理想值 1 最多的是 {worst_bias['grade']}mm，"
            f"BIAS = {_fmt(worst_bias['BIAS'])}（{direction}）；"
            f"BIAS 是频率偏差（预报事件数/观测事件数），不是平均误差。"
        )

    far_entries = [
        entry
        for entry in grades
        if np.isfinite(entry.get("FAR", np.nan)) and np.isfinite(entry.get("漏报率", np.nan))
    ]
    if far_entries:
        entry = far_entries[-1]
        far_text, miss_text = _fmt(entry["FAR"]), _fmt(entry["漏报率"])
        # 比**格式化之后**的文本：两个值差在小数点后第四位时，正文里看到的仍是
        # 「1.000 高于 1.000」——按显示值判平，才不会写出这种自相矛盾的句子。
        if far_text == miss_text:
            findings.append(
                f"误差形态：{entry['grade']}mm 的 FAR 与漏报率均为 {far_text}，"
                f"空报与漏报比重相当，无主导方向。"
            )
        elif entry["FAR"] > entry["漏报率"]:
            findings.append(
                f"误差形态偏**空报**：{entry['grade']}mm 的 FAR {far_text}"
                f" 高于漏报率 {miss_text}。"
            )
        else:
            findings.append(
                f"误差形态偏**漏报**：{entry['grade']}mm 的漏报率 {miss_text}"
                f" 高于 FAR {far_text}。"
            )

    decay = summary.get("decay")
    if decay:
        findings.append(
            f"时效衰减：TS 从 {decay['first_lead']:g}h 的 {_fmt(decay['first_ts'])}"
            f" 降到 {decay['last_lead']:g}h 的 {_fmt(decay['last_ts'])}"
            f"（绝对 -{_fmt(decay['drop'])}，约 {_fmt(decay['per_day'])}/天）。"
        )

    zero_hit = [entry for entry in grades if entry["hits"] == 0]
    if zero_hit:
        names = "、".join(entry["grade"] for entry in zero_hit)
        findings.append(
            f"{names}mm 在全部时效中**命中为 0**，因此 TS/漏报率 显示为 —（无有效配对或无事件），"
            f"这与「评分为 0.000」含义不同，解读时注意区分。"
        )
    no_event = [entry for entry in grades if entry["events"] == 0]
    if no_event:
        names = "、".join(entry["grade"] for entry in no_event)
        findings.append(
            f"{names}mm 在评估期内**观测事件合计为 0**，属于样本库的物理上限："
            f"该量级技巧**不可评定**（不具备统计学意义），不是模型缺陷，也不应当作结论使用。"
        )
    sparse = [
        entry
        for entry in grades
        if 0 < entry["events"] < 50 and entry not in no_event
    ]
    if sparse:
        names = "、".join(
            f"{entry['grade']}（{entry['events']} 次事件）" for entry in sparse
        )
        findings.append(f"样本量过少、方差大，仅供参照：{names}。")
    return findings


def _grade_table(grades: Sequence[Dict[str, object]]) -> str:
    """（见 _source_of 与表格生成）"""
    header = (
        "| 降水等级 | 阈值(mm) | TS | POD | FAR | 漏报率 | BIAS "
        "| TS(等权平均) | 命中 | 漏报 | 空报 | 样本合计 |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for entry in grades:
        lines.append(
            f"| {entry['grade']} | {_fmt(entry['threshold'], 3)} "
            f"| {_fmt(entry['TS'])} {_ts_mark(entry['TS'])}"
            f" | {_fmt(entry['POD'])} | {_fmt(entry['FAR'])} | {_fmt(entry['漏报率'])}"
            f" | {_fmt(entry['BIAS'])} | {_fmt(entry['TS_mean'])}"
            f" | {entry['hits']} | {entry['misses']} | {entry['false_alarms']}"
            f" | {entry['n_pairs']} |"
        )
    return "\n".join(lines)


def _ts_mark(value: float) -> str:
    """技巧分级标记：≥0.3 好 / 0.2–0.3 中 / <0.2 弱。"""
    if not np.isfinite(value):
        return "—"
    if value >= 0.3:
        return "🟢"
    if value >= 0.2:
        return "🟡"
    return "🔴"


def _source_of(sources: Optional[Dict[str, str]], name: str) -> str:
    """报告头里的来源标注：给了文件路径就写路径，否则如实写"未提供"。"""
    if sources and sources.get(name):
        return f"`{sources[name]}`"
    return "未提供（调用方应在 sources 里给出文件路径）"


def _coverage_sentence(coverage: Optional[Dict[str, int]]) -> str:
    """第一节的实测覆盖句：各份给了多少对有效样本。

    这句必须存在：对比表只写分数、不写样本量，而各份来源的起报日数可以差一个
    量级（本批是 29 天对 348 天）。不写出来，读者会把样本最少的那份的名次当成
    结论。样本数取**最小阈值档的全时效合计**，与第二节「样本合计」列同口径。

    只有一个对象（或全都是空值）时返回空串——单模型报告里这句话没有比较对象，
    留着只是噪音。
    """
    entries = [(str(name), int(value)) for name, value in (coverage or {}).items() if value]
    if len(entries) < 2:
        return ""
    primary, base = entries[0]
    detail = "、".join(f"{name} {value:,}" for name, value in entries)
    text = (
        f"**本批实测覆盖并不齐**（最小阈值档的全时效配对样本数）：{detail}。"
        f"站点数每天大体固定，这组数的比值基本就是各份起报日数的比值。"
    )
    lowest = min(entries, key=lambda item: item[1])
    if base and lowest[0] != primary and lowest[1] / base < 0.5:
        text += (
            f"样本最少的是 {lowest[0]}，只有 {primary} 的 {lowest[1] / base:.1%}——"
            f"它的分数抽样波动明显更大，名次不宜当结论用。"
        )
    return text


def business_conclusion(grades: Sequence[Dict[str, object]]) -> str:
    """把技巧水平翻译成业务语言（基于数据自动生成，不是套话）。"""
    def _entry(threshold: float):
        for item in grades:
            if item.get("threshold") is not None and abs(item["threshold"] - threshold) < 1e-6:
                return item
        return None

    light = _entry(0.1)
    moderate = _entry(10.0) or _entry(13.0)
    heavy = _entry(25.0) or _entry(50.0)

    parts: List[str] = []
    if light:
        usable = "可作为有无降水的定性参考" if light["TS"] >= 0.3 else "仅能给出粗略的有雨/无雨趋势"
        parts.append(f"小雨（≥0.1mm）TS {_fmt(light['TS'])}，{usable}")
    if moderate:
        parts.append(
            f"中雨（≥{_fmt(moderate['threshold'], 0)}mm）TS {_fmt(moderate['TS'])}，"
            + ("可作量级参考" if moderate["TS"] >= 0.2 else "已基本失去量级判别能力")
        )
    if heavy:
        parts.append(f"强降水（≥{_fmt(heavy['threshold'], 0)}mm）TS {_fmt(heavy['TS'])}")

    conclusion = "；".join(parts)
    if heavy and np.isfinite(heavy["TS"]) and heavy["TS"] < 0.15:
        conclusion += "。**结论定性：当前模型不宜直接用于强降水预警，需先做后处理订正。**"
    elif conclusion:
        conclusion += "。"
    return conclusion or "数据不足以给出业务定性。"


def recommendations(grades: Sequence[Dict[str, object]]) -> List[str]:
    """按数据条件给出的改进建议（只在条件成立时出现）。"""
    items: List[str] = []
    light = next((item for item in grades if item.get("threshold") == 0.1), None)
    heavy = next(
        (item for item in grades if item.get("threshold") in (25.0, 50.0)), None
    )

    if light and np.isfinite(light["BIAS"]) and light["BIAS"] > 1.3 and light["FAR"] > 0.45:
        items.append(
            f"小雨量级 BIAS {_fmt(light['BIAS'])}、FAR {_fmt(light['FAR'])} → "
            f"模型倾向「到处报小雨」、空间上过度平滑。损失函数可引入**焦点损失（Focal Loss）**"
            f"降低无雨背景点权重，并在频次上做**频率匹配**订正。"
        )
    if heavy and np.isfinite(heavy["TS"]) and heavy["TS"] < 0.15:
        items.append(
            f"强降水（≥{_fmt(heavy['threshold'], 0)}mm）TS {_fmt(heavy['TS'])} → "
            f"高频细节丢失。可引入**空间谱损失（Spectral Loss）**恢复小尺度结构，"
            f"并对极端事件做**样本重加权**以缓解长尾稀疏。"
        )
    sparse = [item for item in grades if item["events"] < 50]
    if sparse:
        names = "、".join(item["grade"] for item in sparse)
        items.append(
            f"{names}mm 的事件数不足 50（合计 {sum(item['events'] for item in sparse)} 次）→ "
            f"先延长评估期或扩大统计区域补齐样本，再评该量级技巧；"
            f"在样本补齐前不要据此调参。"
        )
    items.append(
        "后处理路径：分位数映射（QM）或频率匹配（FM）可同时压缩 BIAS 与空报；"
        "对强降水优先保证不漏报（提高 POD），再压 FAR。"
    )
    return items


# --------------------------------------------------------------------------- #
# 各节小结
#
# 这些函数是报告的「分析」部分：只吃 summarize() / box_stats() 已经算好的数字，
# **不重新读表、不重新实现指标**——口径只有一处，表、图、文字就不可能各说各话。
#
# 一条硬规矩：每句话都得带具体数值，或者是个能被数据证伪的判断。
# 写不出结论时如实说「数据不足以判断」，不拿「表现良好」这类话凑数。
# --------------------------------------------------------------------------- #
def analysis_by_grade(grades: Sequence[Dict[str, object]]) -> str:
    """第二节小结：量级维度——技巧分级、单调性、断崖在哪一档、样本随量级的收缩。"""
    usable = [entry for entry in grades if np.isfinite(entry.get("TS", np.nan))]
    if not usable:
        return "各降水等级的 TS 均不可定义（无有效配对或无命中），无法做量级间比较。"

    marks = [_ts_mark(entry["TS"]) for entry in usable]
    tally = "、".join(
        f"{mark} {marks.count(mark)} 档" for mark in ("🟢", "🟡", "🔴") if marks.count(mark)
    )
    parts = [f"{len(usable)} 个可评定的降水等级里 {tally}。"]

    steps = [
        (usable[index], usable[index + 1], usable[index]["TS"] - usable[index + 1]["TS"])
        for index in range(len(usable) - 1)
    ]
    rises = [step for step in steps if step[2] < -0.005]
    if rises:
        where = "、".join(f"{left['grade']}→{right['grade']}" for left, right, _ in rises)
        parts.append(
            f"TS 并非随量级单调下降：{where} 处出现回升，说明该处的样本波动已经盖过技巧的真实差异，"
            f"这个位置上的名次不可当真。"
        )
    elif steps:
        parts.append(
            f"TS 随量级**单调下降**，从 {usable[0]['grade']}mm 的 {_fmt(usable[0]['TS'])} "
            f"一路降到 {usable[-1]['grade']}mm 的 {_fmt(usable[-1]['TS'])}。"
        )
    if steps:
        left, right, drop = max(steps, key=lambda step: step[2])
        if drop > 0:
            parts.append(
                f"相邻量级之间落差最大的是 **{left['grade']}→{right['grade']}**"
                f"（{_fmt(left['TS'])} → {_fmt(right['TS'])}，掉 {_fmt(drop)}），"
                f"技巧的断崖就出现在这一档之间。"
            )

    business = [entry for entry in usable if entry["TS"] >= 0.2]
    if business:
        best = business[-1]
        parts.append(
            f"以 TS ≥ 0.2（🟡 及以上）作为最低可用门槛，可用量级最高到 {best['grade']}mm"
            f"（TS {_fmt(best['TS'])}），再往上不具备定量判别能力。"
        )
    else:
        parts.append("没有任何量级达到 TS ≥ 0.2，全部等级都不满足最低可用门槛。")

    first, last = usable[0], usable[-1]
    if first["events"] > 0 and 0 < last["events"] < first["events"]:
        ratio = first["events"] / last["events"]
        parts.append(
            f"观测事件数从 {first['grade']}mm 的 {first['events']:,} 次收缩到 "
            f"{last['grade']}mm 的 {last['events']:,} 次"
            + (f"（约 1/{ratio:.0f}）" if ratio >= 10 else "")
            + "，高量级的统计不确定性随之放大，其 TS 的小数位不宜过度解读。"
        )
    return "".join(parts)


def analysis_by_lead(summary: Dict[str, object], *, floor: float = 0.10) -> str:
    """第三节小结：时效维度——有效时效上限、单调性、平台期、衰减速度。"""
    pivot = summary.get("pivot")
    leads = list(summary.get("leads") or [])
    if pivot is None or len(pivot.index) < 2 or len(leads) < 2:
        return "时效点不足两个，无法判断技巧的时效衰减。"

    limits: List[tuple] = []
    for grade in pivot.columns:
        series = pivot[grade].dropna()
        below = series[series < floor]
        limits.append((str(grade), float(below.index.min()) if len(below) else None))

    parts: List[str] = []
    broken = [(grade, lead) for grade, lead in limits if lead is not None]
    kept = [grade for grade, lead in limits if lead is None]
    if broken:
        detail = "、".join(f"{grade}mm 在 {lead:g}h" for grade, lead in broken)
        parts.append(
            f"以 **TS 首次跌破 {floor:g}** 作为「有效时效上限」：{detail} 已跌破"
            + (f"；{'、'.join(f'{grade}mm' for grade in kept)} 全程未跌破。" if kept else "。")
        )
    else:
        parts.append(
            f"全部等级到 {leads[-1]:g}h 都未跌破 {floor:g}，时效衰减不构成本次评估的限制因素。"
        )

    curve = pivot.mean(axis=1).dropna()
    if len(curve) >= 2:
        steps = curve.diff().dropna()
        rises = steps[steps > 0.005]
        if rises.empty:
            parts.append(
                f"各等级等权平均的 TS 随时效**单调下降**"
                f"（{curve.index[0]:g}h 的 {_fmt(curve.iloc[0])} → "
                f"{curve.index[-1]:g}h 的 {_fmt(curve.iloc[-1])}）。"
            )
        else:
            parts.append(
                f"等权平均 TS 总体随时效下降，但在 {float(rises.idxmax()):g}h 出现 "
                f"+{_fmt(float(rises.max()))} 的回升，属于抽样波动而非真实的技巧反转。"
            )
        tail = steps.tail(3)
        if len(tail) == 3 and bool((tail.abs() < 0.005).all()):
            parts.append(
                f"最后三个时效区间（{float(tail.index[0]):g}–{curve.index[-1]:g}h）的相邻变化都小于 0.005，"
                f"技巧已进入平台期：再往后延长时间不会带来等比例的进一步损失。"
            )

    decay = summary.get("decay") or {}
    if decay and np.isfinite(decay.get("relative", np.nan)):
        judge = (
            "相对衰减已超过 50%，长时效技巧明显退化，结论应限定在较短时效内。"
            if decay["relative"] > 0.5
            else "相对衰减未超过 50%，长时效仍保有一定技巧。"
        )
        parts.append(
            f"整体来看，等权平均 TS 从 {decay['first_lead']:g}h 的 {_fmt(decay['first_ts'])} 降到 "
            f"{decay['last_lead']:g}h 的 {_fmt(decay['last_ts'])}：绝对衰减 {_fmt(decay['drop'])}、"
            f"相对衰减 {decay['relative']:.0%}、约 {_fmt(decay['per_day'])}/天。{judge}"
        )

    speeds = []
    for grade in pivot.columns:
        series = pivot[grade].dropna()
        if len(series) >= 2 and series.iloc[0] > 0:
            speeds.append((str(grade), float((series.iloc[0] - series.iloc[-1]) / series.iloc[0])))
    speeds = [item for item in speeds if item[1] > 0]
    if speeds:
        grade, relative = max(speeds, key=lambda item: item[1])
        parts.append(f"逐等级比较，相对衰减最快的是 {grade}mm（{relative:.0%}）。")
    return "".join(parts)


def analysis_error_structure(grades: Sequence[Dict[str, object]]) -> str:
    """第四节小结：误差形态——空报与漏报谁主导、在哪一档反转、BIAS 的走向。"""
    parts: List[str] = []
    pairs = [
        entry for entry in grades
        if np.isfinite(entry.get("FAR", np.nan)) and np.isfinite(entry.get("漏报率", np.nan))
    ]
    if not pairs:
        parts.append("FAR 与漏报率在各等级上都不可用，无法判断误差形态。")
    else:
        dominant = [entry["FAR"] > entry["漏报率"] for entry in pairs]
        flips = [
            index for index in range(1, len(dominant)) if dominant[index] != dominant[index - 1]
        ]
        if not flips:
            kind = "空报" if dominant[0] else "漏报"
            span = (
                f"{pairs[0]['grade']}–{pairs[-1]['grade']}mm"
                if len(pairs) > 1
                else f"{pairs[0]['grade']}mm"
            )
            parts.append(f"{span} 全部以**{kind}为主**，误差形态在量级之间没有反转。")
        else:
            index = flips[0]
            before, after = pairs[index - 1], pairs[index]
            parts.append(
                f"误差形态在 {before['grade']} 与 {after['grade']} 之间**发生反转**："
                f"{before['grade']} 以空报为主（FAR {_fmt(before['FAR'])} > 漏报率 {_fmt(before['漏报率'])}），"
                f"{after['grade']} 转为漏报为主（漏报率 {_fmt(after['漏报率'])} > FAR {_fmt(after['FAR'])}）。"
            )
            if len(flips) > 1:
                parts.append(
                    f"整条量级链上共有 {len(flips)} 处反转，说明该模型没有贯穿各量级的一致误差倾向，"
                    f"订正策略需要分档设定。"
                )
        entry = max(pairs, key=lambda item: abs(item["FAR"] - item["漏报率"]))
        kind = "空报" if entry["FAR"] > entry["漏报率"] else "漏报"
        parts.append(
            f"空报与漏报相差最悬殊的是 {entry['grade']}mm"
            f"（FAR {_fmt(entry['FAR'])} 对漏报率 {_fmt(entry['漏报率'])}，{kind}主导）。"
        )

    biases = [entry for entry in grades if np.isfinite(entry.get("BIAS", np.nan))]
    if biases:
        low, high = biases[0], biases[-1]
        closest = min(biases, key=lambda item: abs(item["BIAS"] - 1.0))
        drift = high["BIAS"] - low["BIAS"]
        trend = (
            "随量级升高而下降" if drift < -0.05
            else ("随量级升高反而上升" if drift > 0.05 else "在各量级之间大致持平")
        )
        parts.append(
            f"BIAS 从 {low['grade']}mm 的 {_fmt(low['BIAS'])} 变到 {high['grade']}mm 的 "
            f"{_fmt(high['BIAS'])}（{trend}）；最接近理想值 1 的是 {closest['grade']}mm"
            f"（{_fmt(closest['BIAS'])}）。"
        )
        if low["BIAS"] > 1.1 and high["BIAS"] < 0.9:
            parts.append(
                "**小雨偏多、强降水偏少**，这是空间过度平滑的典型征兆："
                "模式把小尺度强中心抹成了大范围弱降水，既抬高小雨的空报，也吃掉强降水的命中。"
            )
    return "".join(parts) or "误差形态与偏差结构均不可评定。"


def analysis_stability(
    models: Dict[str, pd.DataFrame], primary: str, metric: str = "TS"
) -> str:
    """第五节小结：跨时效稳定性——谁最稳、离群时效落在哪、两种口径的名次是否一致。"""
    per_model: Dict[str, List[Dict[str, object]]] = {}
    for name, frame in models.items():
        rows = [row for row in box_stats_by_grade(frame, metric=metric) if row["n"]]
        if rows:
            per_model[name] = rows
    if not per_model:
        return "各模型在各量级上都没有有效样本，无法评估跨时效稳定性。"

    parts: List[str] = []
    spread = {
        name: float(np.median([row["iqr"] for row in rows]))
        for name, rows in per_model.items()
    }
    ranked = sorted(spread, key=lambda name: spread[name])
    if len(ranked) == 1:
        parts.append(
            f"单模型评估：{ranked[0]} 各量级 IQR 的中位数为 {_fmt(spread[ranked[0]])}，"
            f"这是本次评估的跨时效稳定性基准。"
        )
    elif spread[ranked[-1]] - spread[ranked[0]] < 0.01:
        parts.append(
            f"各模型的跨时效稳定性接近（IQR 中位数 {_fmt(spread[ranked[0]])}–"
            f"{_fmt(spread[ranked[-1]])}，极差不足 0.01），稳定性不构成它们之间的区分度。"
        )
    else:
        parts.append(
            f"按各量级 IQR 的中位数排序，跨时效最稳定的是 **{ranked[0]}**"
            f"（{_fmt(spread[ranked[0]])}），最不稳定的是 **{ranked[-1]}**"
            f"（{_fmt(spread[ranked[-1]])}）——箱体越窄，说明该模型各时效的技巧越均衡。"
        )

    outliers = sorted(
        {float(lead) for row in per_model.get(primary, []) for lead in row["outlier_leads"]}
    )
    if outliers and primary in models:
        all_leads = sorted(float(value) for value in pd.unique(models[primary]["lead_h"]))
        middle = (all_leads[0] + all_leads[-1]) / 2
        early = sum(1 for lead in outliers if lead <= middle)
        where = (
            "短时效段" if early * 2 > len(outliers)
            else ("长时效段" if early * 2 < len(outliers) else "首尾两端")
        )
        shown = "、".join(f"{lead:g}h" for lead in outliers[:6])
        parts.append(
            f"{primary} 的离群时效共 {len(outliers)} 个（{shown}"
            + ("…" if len(outliers) > 6 else "")
            + f"），偏向{where}——这些时效的技巧与其余时效明显脱节，"
            f"只看等权平均会把这种脱节抹平。"
        )
    elif primary in models:
        parts.append(f"{primary} 的各量级都没有离群时效，TS 的跨时效分布连续、没有突变的时效点。")

    flips = []
    for grade in ordered_grades(list(models.values())):
        medians: Dict[str, float] = {}
        means: Dict[str, float] = {}
        for name, frame in models.items():
            subset = frame[frame["grade"].astype(str) == grade]
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            if values.empty:
                continue
            medians[name] = float(values.median())
            means[name] = float(values.mean())
        if len(medians) < 2:
            continue
        by_median = max(medians, key=medians.get)
        by_mean = max(means, key=means.get)
        if by_median != by_mean:
            flips.append((grade, by_median, by_mean))
    if flips:
        detail = "、".join(
            f"{grade}（中位数 {left} 领先、等权平均 {right} 领先）" for grade, left, right in flips
        )
        parts.append(
            f"**注意口径**：{detail}。本节表里的统计量是中位数/IQR，第二节表里的是等权平均，"
            f"两者给出不同的领先者是正常现象（前者抗离群、后者被离群时效拉动），不可互相替代。"
        )
    return "".join(parts)


def analysis_comparison(
    data: pd.DataFrame, comparisons: Dict[str, pd.DataFrame], lead: float
) -> str:
    """第六节小结：对比维度——谁在哪些量级领先、名次是否随量级翻转、差距分布。

    ``comparisons`` 既可以是多个对比模型，也可以是单个基准——单基准时
    「领先者随量级翻转」照样能看出两者的互补性，所以两种分支共用这一个函数。
    """
    frames: Dict[str, pd.DataFrame] = {"本模型": data, **comparisons}
    table: Dict[str, Dict[str, float]] = {}
    for grade in ordered_grades(list(frames.values())):
        row: Dict[str, float] = {}
        for name, frame in frames.items():
            subset = frame[
                (pd.to_numeric(frame["lead_h"], errors="coerce") == lead)
                & (frame["grade"].astype(str) == grade)
            ]
            values = pd.to_numeric(subset["TS"], errors="coerce").dropna()
            if len(values):
                row[name] = float(values.iloc[0])
        if len(row) >= 2:
            table[grade] = row
    if not table:
        return f"{lead:g}h 时效上没有两个及以上对象同时有有效 TS，无法比较。"

    parts: List[str] = []
    # 差距小于 0.005 视为并列——和 `_baseline_table` 里 ↑/↓/≈ 的门槛同一个数，
    # 免得表里写着「≈」、正文却说某一方领先。
    def _tie(row: Dict[str, float]) -> bool:
        order = sorted(row.values(), reverse=True)
        return len(order) > 1 and order[0] - order[1] < 0.005

    tied = [grade for grade, row in table.items() if _tie(row)]
    winners = {
        grade: max(row, key=row.get) for grade, row in table.items() if grade not in tied
    }
    if not winners:
        parts.append(
            f"各对象在 {lead:g}h、各量级上的差距都小于 0.005，视为并列，无法据此分出高下。"
        )
    else:
        counts: Dict[str, int] = {}
        for winner in winners.values():
            counts[winner] = counts.get(winner, 0) + 1
        if len(counts) == 1:
            only = next(iter(counts))
            parts.append(
                f"{lead:g}h 时效上，**{only}** 在全部 {len(winners)} 个可分出高下的量级上都领先。"
            )
        else:
            detail = "、".join(f"{grade} 由{winner}领先" for grade, winner in winners.items())
            parts.append(f"{lead:g}h 时效上的领先者随量级变化：{detail}。")
    if tied:
        # 并列要说清是"谁和谁并列"——只写"各对象并列"会把差距明显的第三名也算进去
        detail = "；".join(
            f"{grade} 的前两名 "
            + "与".join(
                f"{name}（{table[grade][name]:.3f}）"
                for name in sorted(table[grade], key=table[grade].get, reverse=True)[:2]
            )
            for grade in tied
        )
        parts.append(f"{detail} 差距不足 0.005，分不出领先者。")

    sequence = list(winners.items())
    crossovers = [
        index for index in range(1, len(sequence)) if sequence[index][1] != sequence[index - 1][1]
    ]
    if crossovers:
        index = crossovers[0]
        before, after = sequence[index - 1], sequence[index]
        parts.append(
            f"名次在 {before[0]} 与 {after[0]} 之间**发生翻转**（领先者由{before[1]}变为"
            f"{after[1]}）——两者的优势区间不同，只报一个跨量级平均值会把这种互补性掩盖掉。"
        )
        if len(crossovers) > 1:
            parts.append(f"整条量级链上共出现 {len(crossovers)} 处领先者交替。")

    gaps: Dict[str, float] = {}
    for grade, row in table.items():
        if "本模型" in row:
            others = [value for name, value in row.items() if name != "本模型"]
            if others:
                gaps[grade] = row["本模型"] - max(others)
    # 各量级的差距都一样（比如全部并列）时，这句话不含信息，不如不说
    if gaps and max(gaps.values()) - min(gaps.values()) >= 0.005:
        best = max(gaps, key=gaps.get)
        worst = min(gaps, key=gaps.get)
        parts.append(
            f"以最强的对比对象为参照，本模型相对领先最多的是 {best}（{gaps[best]:+.3f}）、"
            f"相对落后最多的是 {worst}（{gaps[worst]:+.3f}）。"
        )

    # 名次只在分得出高下的量级上算——并列的量级硬排名次就是在报告噪声
    ranks = []
    for grade in winners:
        row = table[grade]
        if "本模型" in row:
            better = sum(1 for value in row.values() if value > row["本模型"] + 1e-12)
            ranks.append(better + 1)
    if ranks:
        widest = max(len(row) for row in table.values())
        parts.append(f"本模型在 {len(ranks)} 个量级上的平均名次为 {np.mean(ranks):.1f}/{widest}。")
    return "".join(parts)


def analysis_probability(summaries: Dict[str, Dict[str, object]]) -> str:
    """第六节小结：区分能力、Brier 技巧的正负与走向、高阈值档的样本、AVG 的时效衰减。

    与其余 ``analysis_*`` 同一条规矩：只吃 ``probability_summary`` 已经算好的数字，
    不重新读表；每句话都要带具体数值，写不出结论时如实说。
    """
    if not summaries:
        return "没有概率评分结果，本节无法给出定量分析。"
    names = list(summaries)
    primary = names[0]
    rows = summaries[primary]["rows"]
    average = summaries[primary]["average"]
    parts: List[str] = []

    usable = [entry for entry in rows if np.isfinite(entry.get("AROC", np.nan))]
    if usable:
        best = max(usable, key=lambda item: item["AROC"])
        worst = min(usable, key=lambda item: item["AROC"])
        below = [entry for entry in usable if entry["AROC"] <= 0.5]
        head = (
            f"{primary} 各档 AROC 在 {_fmt(worst['AROC'])}–{_fmt(best['AROC'])} 之间，"
        )
        if below:
            head += (
                "其中 " + "、".join(entry["grade"] for entry in below)
                + " 未超过 0.5 的无技巧线，这几档的概率预报分不开事件与非事件；"
            )
        else:
            head += (
                f"全部高于 0.5 的无技巧线（最低的 {worst['grade']} 档高出 "
                f"{_fmt(worst['AROC'] - 0.5)}）；"
            )
        parts.append(
            head + f"区分能力最强的是 {best['grade']}（{_fmt(best['AROC'])}）。"
        )

    scored = [entry for entry in rows if np.isfinite(entry.get("BSS", np.nan))]
    if scored:
        positive = [entry for entry in scored if entry["BSS"] > 0]
        best = max(scored, key=lambda item: item["BSS"])
        worst = min(scored, key=lambda item: item["BSS"])
        parts.append(
            f"BSS 有 {len(positive)}/{len(scored)} 档为正：最好的是 {best['grade']}"
            f"（{_fmt(best['BSS'])}），最差的是 {worst['grade']}（{_fmt(worst['BSS'])}）。"
        )
        if worst["BSS"] < 0:
            parts.append(
                f"{worst['grade']} 档为负，说明这一档的概率预报**还不如气候基准**——"
                f"它的 BS {_fmt(worst['BS'])} 高于 BS_ref {_fmt(worst['BS_ref'])}，"
                f"即预报概率的分布比「一律报基准率 {_fmt(worst['base_rate'])}」还差。"
            )

    if rows:
        lightest, heaviest = rows[0], rows[-1]
        # 各档的 ``n_points`` 是**同一个数**（阈值只改事件的定义，不改配对点），
        # 真正随量级收缩的是**事件次数** = n_points × 基准率。写成「样本量收缩」
        # 是错的，而且两档数字并排一样，一眼就露馅。
        if (
            lightest is not heaviest
            and lightest.get("n_points")
            and np.isfinite(lightest.get("base_rate", np.nan))
            and np.isfinite(heaviest.get("base_rate", np.nan))
            and heaviest["base_rate"] > 0
        ):
            ratio = lightest["base_rate"] / heaviest["base_rate"]
            extra = f"（约 {ratio:.0f} 分之一）" if ratio >= 2 else ""
            parts.append(
                f"各档的样本点数相同（{lightest['n_points']:,} 对——阈值只改事件的定义、"
                f"不改配对点），但**事件次数**随量级急剧收缩：{lightest['grade']} 档的"
                f"基准率 {_fmt(lightest['base_rate'])}、{heaviest['grade']} 档只剩 "
                f"{_fmt(heaviest['base_rate'])}{extra}"
                f"——高阈值档的 AROC/BSS 波动更大，小数位不宜过度解读。"
            )

    by_lead = summaries[primary].get("by_lead")
    if by_lead is not None and len(by_lead) >= 2:
        first_lead, last_lead = float(by_lead.index[0]), float(by_lead.index[-1])
        aroc_first, aroc_last = float(by_lead["AROC"].iloc[0]), float(by_lead["AROC"].iloc[-1])
        bss_first, bss_last = float(by_lead["BSS"].iloc[0]), float(by_lead["BSS"].iloc[-1])
        parts.append(
            f"AVG 档随时效：AROC 从 {first_lead:g}h 的 {_fmt(aroc_first)} 变到 "
            f"{last_lead:g}h 的 {_fmt(aroc_last)}（{aroc_last - aroc_first:+.3f}），"
            f"BSS 从 {_fmt(bss_first)} 变到 {_fmt(bss_last)}（{bss_last - bss_first:+.3f}）——"
            + (
                "区分能力随时效退化，长时效的结论同样要限定范围。"
                if aroc_last < aroc_first
                else "区分能力并不随时效单调下降，这时段内的分档主要由样本波动支配。"
            )
        )

    if len(names) > 1:
        pool = {
            name: summary["average"]
            for name, summary in summaries.items()
            if np.isfinite(summary["average"].get("AROC", np.nan))
            and np.isfinite(summary["average"].get("BSS", np.nan))
        }
        if len(pool) >= 2:
            best_aroc = max(pool, key=lambda name: pool[name]["AROC"])
            best_bss = max(pool, key=lambda name: pool[name]["BSS"])
            parts.append(
                f"AVG 档上 AROC 最高的是 {best_aroc}（{_fmt(pool[best_aroc]['AROC'])}）、"
                f"BSS 最高的是 {best_bss}（{_fmt(pool[best_bss]['BSS'])}）。"
                f"各模型的概率样本量并不相同（见「样本点数」列），样本少的那个名次波动更大。"
            )

    if np.isfinite(average.get("AROC", np.nan)):
        parts.append(
            f"{primary} 的 AVG 档：AROC {_fmt(average['AROC'])}、BSS {_fmt(average['BSS'])}"
            f"（对 {average['n_grades']} 个可算阈值等权平均）。"
        )
    return "".join(parts) or "概率评分的各档取值都不可用，本节无法给出定量分析。"


def _lead_table(pivot: pd.DataFrame) -> str:
    columns = [str(column) for column in pivot.columns]
    lines = ["| 时效(h) | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for lead, row in pivot.iterrows():
        values = " | ".join(_fmt(row[column]) for column in pivot.columns)
        lines.append(f"| {lead:g} | {values} |")
    return "\n".join(lines)


def _baseline_table(data: pd.DataFrame, baseline: pd.DataFrame, lead: float) -> str:
    left = data[data["lead_h"] == lead].set_index("grade")["TS"]
    right = baseline[baseline["lead_h"] == lead].set_index("grade")["TS"]
    lines = [
        "| 降水等级 | 本模型 TS | 对比模型 TS | 差值 |",
        "|---|---|---|---|",
    ]
    for grade in left.index:
        if grade in right.index:
            delta = float(left[grade]) - float(right[grade])
            flag = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
            lines.append(
                f"| {grade} | {_fmt(left[grade])} | {_fmt(right[grade])} | {flag} {delta:+.3f} |"
            )
    return "\n".join(lines)


def box_stats(data: pd.DataFrame, metric: str = "TS") -> Dict[str, object]:
    """一个箱体的 Tukey 统计量——**与 matplotlib 的 boxplot 同口径**。

    样本是"同一模型、同一量级在全部时效上的取值"，一行一个时效。
    放进 ``ts_report`` 而不是画图模块，是为了让**报告表格和图上箱体同源**：
    两边都调这个函数，就不存在"表里的中位数和图上的线对不上"。

    返回 ``{"n", "q1", "median", "q3", "iqr", "whisker_lo", "whisker_hi",
    "outlier_leads", "outlier_values"}``；没有有效值时 ``n=0``、统计量为 ``nan``、列表为空。

    Raises:
        KeyError: 表里没有 ``metric`` 列。
    """
    if metric not in data.columns:
        raise KeyError(f"结果表里没有指标列 '{metric}'")

    values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float)
    leads = (
        pd.to_numeric(data["lead_h"], errors="coerce").to_numpy(dtype=float)
        if "lead_h" in data.columns
        else np.full(values.shape, np.nan)
    )
    # 与 prepare_ts_dataframe 的 dropna(subset=["lead_h","TS"]) 同口径：非有限值一律不算样本
    keep = np.isfinite(values)
    values, leads = values[keep], leads[keep]

    if values.size == 0:
        nan = float("nan")
        return {
            "n": 0, "q1": nan, "median": nan, "q3": nan, "iqr": nan,
            "whisker_lo": nan, "whisker_hi": nan,
            "outlier_leads": [], "outlier_values": [],
        }

    # 分位数用 np.percentile 的默认参数（线性插值）——matplotlib 的 cbook.boxplot_stats
    # 用的也是它，所以口径天然一致；千万别传 method=，一传两边就可能对不上。
    q1, median, q3 = (float(value) for value in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    # 须线画到"栅栏内最远的点"，不是把极值夹到栅栏上；栅栏内一个点都没有（全是离群点）
    # 时退回四分位——这是 matplotlib 的规则，不是常见的 max(min, Q1-1.5*IQR) 写法。
    inside = values[(values >= low_fence) & (values <= high_fence)]
    outliers = (values < low_fence) | (values > high_fence)
    return {
        "n": int(values.size),
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "whisker_lo": float(inside.min()) if inside.size else q1,
        "whisker_hi": float(inside.max()) if inside.size else q3,
        "outlier_leads": [float(value) for value in leads[outliers]],
        "outlier_values": [float(value) for value in values[outliers]],
    }


def box_stats_by_grade(data: pd.DataFrame, metric: str = "TS") -> List[Dict[str, object]]:
    """``[{"grade", "threshold", **box_stats(...)}]``，按阈值升序。"""
    rows: List[Dict[str, object]] = []
    for grade, subset in data.groupby("grade", observed=True):
        thresholds = (
            subset["threshold_mm"].dropna()
            if "threshold_mm" in subset.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "grade": str(grade),
                "threshold": float(thresholds.iloc[0]) if len(thresholds) else None,
                **box_stats(subset, metric=metric),
            }
        )
    rows.sort(key=lambda row: _grade_sort_key(row["grade"], row["threshold"]))
    return rows


def ordered_grades(frames: Sequence[pd.DataFrame]) -> List[str]:
    """多张结果表里出现过的降水等级**并集**，按 ``threshold_mm`` 升序。

    缺 ``threshold_mm`` 时退回解析 ``"≥X"`` 文本（``_grade_sort_key`` 已经带了这条路）。
    各模型的等级集合未必一样（``ts_multi_fuxi_ens`` 就只有 5 档），所以要取并集。
    """
    thresholds: Dict[str, float] = {}
    for frame in frames:
        if "grade" not in frame.columns:
            continue
        first = (
            frame.groupby("grade", observed=True)["threshold_mm"].first()
            if "threshold_mm" in frame.columns
            else pd.Series(dtype=float)
        )
        for grade in frame["grade"].astype(str):
            if grade in thresholds:
                continue
            value = first.get(grade, float("nan")) if len(first) else float("nan")
            thresholds[grade] = float(value) if pd.notna(value) else float("nan")
    return sorted(thresholds, key=lambda grade: _grade_sort_key(grade, thresholds[grade]))


def collect_box_models(
    model_name: str,
    data: pd.DataFrame,
    *,
    baseline_df: Optional[pd.DataFrame] = None,
    baseline_name: Optional[str] = None,
    baselines: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """箱线图画哪些模型：本模型在前，然后是单基准、各对比模型。

    图和表都调这一个函数，免得两边的模型集合悄悄分叉。
    """
    models: Dict[str, pd.DataFrame] = {model_name: data}
    if baseline_df is not None and baseline_name:
        models[baseline_name] = baseline_df
    models.update(baselines or {})
    return models


def _box_table(models: Dict[str, pd.DataFrame], metric: str = "TS") -> str:
    """一行一个（模型 × 降水等级）：量级升序，同量级内按 ``models`` 的顺序。"""
    names = list(models)
    grades = ordered_grades(list(models.values()))
    stats = {
        name: {row["grade"]: row for row in box_stats_by_grade(frame, metric=metric)}
        for name, frame in models.items()
    }

    lines = [_BOX_HEADER, _BOX_SEP]
    for grade in grades:
        for name in names:
            row = stats[name].get(grade)
            if row is None or row["n"] == 0:
                # 该模型没有这个量级（或全无有效值）：照出一行全 —，和图上那个空槽位对齐。
                # 五格 —— 对应中位数/IQR/Q1–Q3/须线/离群时效，末格才是时效数 0；
                # 写成 6 格会多出一列，Markdown 表格跟着错位。
                cells = ["—"] * 5 + ["0"]
            else:
                leads = "、".join(f"{lead:g}h" for lead in sorted(row["outlier_leads"])) or "—"
                cells = [
                    _fmt(row["median"]),
                    _fmt(row["iqr"]),
                    f"{_fmt(row['q1'])}–{_fmt(row['q3'])}",
                    f"{_fmt(row['whisker_lo'])}–{_fmt(row['whisker_hi'])}",
                    leads,
                    str(row["n"]),
                ]
            lines.append(f"| {name} | {grade} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _multi_baseline_table(
    data: pd.DataFrame, baselines: Dict[str, pd.DataFrame], lead: float
) -> str:
    """一行一个降水等级，一列一个模型（本模型 + 各对比模型）。"""
    names = list(baselines.keys())
    left = data[data["lead_h"] == lead].set_index("grade")["TS"]
    columns = {
        name: frame[frame["lead_h"] == lead].set_index("grade")["TS"]
        for name, frame in baselines.items()
    }
    header = "| 降水等级 | 本模型 | " + " | ".join(names) + " | 本模型名次 |"
    lines = [header, "|---|" + "---|" * (len(names) + 2)]
    for grade in left.index:
        values = {name: columns[name].get(grade, float("nan")) for name in names}
        pool = {**{name: float(value) for name, value in values.items() if np.isfinite(value)},
                "本模型": float(left[grade]) if np.isfinite(left[grade]) else float("nan")}
        ranked = [name for name, value in pool.items() if np.isfinite(value)]
        # 并列同名次（competition ranking）：名次 = 比自己高的对象个数 + 1。
        # 若按 sorted 的下标算，数值相同的两家谁在前全看字典插入顺序，
        # 表里会出现「3/3」而正文按同一套数据算出「并列第 2」。
        rank = (
            1 + sum(1 for value in pool.values() if value > pool["本模型"])
            if np.isfinite(pool["本模型"]) else 0
        )
        cells = " | ".join(_fmt(values[name]) for name in names)
        lines.append(
            f"| {grade} | {_fmt(left[grade])} | {cells} | {rank}/{len(ranked) if ranked else 0} |"
        )
    return "\n".join(lines)


# ------------------------------------------------------------- 概率评分（第六节）
#
# 这一节吃的是**另一张表**：``diagnostics/probability_wide.csv``，由
# ``weather_ts_ens_prob`` 能力写出（6h 累积窗、逐成员概率、需要外部气候概率参考）。
# 它的窗口与阈值都和 TS 那套不同，所以单独立节、单独编号——见骨架
# ``assets/templates/weather_ts_ens.md`` 第六节。


def _members_phrase(members: Optional[int]) -> str:
    """``51个成员`` / ``集合成员``。

    产物里**没有**成员数（``manifest.json`` 不写这个字段），所以拿不到时老老实实
    写「集合成员」，不编一个数字出来。拿到时由调用方的 ``--members`` 给。
    """
    return f"{int(members)}个成员" if members else "集合成员"


def load_probability(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """校验概率宽表。

    ``None`` 原样返回 ``None``——这份产物没跑概率评分那一段（比如对比模型给的是
    确定性的裸宽表），第六节整节不出现。给了表但缺列则**报错**：少一列的表算出来的
    AROC/BSS 与评测流程自己写的不是一回事，宁可停下。

    Raises:
        ValueError: 表里缺 :data:`PROBABILITY_COLUMNS` 里的列。
    """
    if frame is None:
        return None
    missing = [column for column in PROBABILITY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"概率宽表缺列 {missing}；现有列 {list(frame.columns)}。"
            f"这份表由 weather_ts_ens_prob 能力写出，配置里写 "
            f'pipeline=["weather_ts_ens", "weather_ts_ens_prob"] 才会落到 '
            f"diagnostics/probability_wide.csv。"
        )
    return frame


def probability_summary(frame: pd.DataFrame) -> Dict[str, object]:
    """单模型的概率评分汇总：逐档一行 + 一行 ``AVG`` + ``by_lead``。

    每一档的取值是**各时效等权平均**——AROC / BSS 没有可加的充分统计量，不像 TS
    那样能「先合计列联表再算比率」，逐时效平均是这里唯一站得住的做法。
    ``AVG`` 行再对**可算阈值**等权平均（某档全无有效值时跳过，不拿 0 顶替）。
    ``n_points`` 是该档**全时效合计**的配对数，与第二节的「样本合计」同口径；
    ``AVG`` 行的这一格写的是**参与平均的档数**（老表里叫 ``n_used``），
    不是样本数——它本来就没有单一阈值可数。

    产物自己带的 ``grade="AVG"`` 行**丢掉不要**：那份是产物按它自己的口径折的，
    与本报告「可算阈值等权平均」未必是同一个数，两行并排会自相矛盾。
    """
    if "grade" in frame.columns:
        frame = frame[frame["grade"].astype(str) != "AVG"]

    rows: List[Dict[str, object]] = []
    for grade, subset in frame.groupby("grade", observed=True):
        thresholds = pd.to_numeric(subset["threshold_mm"], errors="coerce").dropna()
        entry: Dict[str, object] = {
            "grade": str(grade),
            "threshold": float(thresholds.iloc[0]) if len(thresholds) else None,
            "n_leads": int(pd.to_numeric(subset["lead_h"], errors="coerce").nunique()),
        }
        for key in ("AROC", "BS", "BS_ref", "BSS", "base_rate"):
            values = pd.to_numeric(subset[key], errors="coerce").dropna()
            entry[key] = float(values.mean()) if len(values) else float("nan")
        points = pd.to_numeric(subset["n_points"], errors="coerce").dropna()
        entry["n_points"] = int(points.sum()) if len(points) else 0
        rows.append(entry)
    rows.sort(key=lambda item: _grade_sort_key(item["grade"], item["threshold"]))

    average: Dict[str, object] = {
        "grade": "AVG", "threshold": None, "base_rate": None, "n_points": None,
    }
    for key in ("AROC", "BS", "BS_ref", "BSS"):
        values = [float(item[key]) for item in rows if np.isfinite(item[key])]
        average[key] = float(np.mean(values)) if values else float("nan")
    average["n_grades"] = len(
        [item for item in rows if np.isfinite(item["AROC"]) and np.isfinite(item["BSS"])]
    )
    average["n_leads"] = max((item["n_leads"] for item in rows), default=0)

    numeric = frame.assign(
        lead_h=pd.to_numeric(frame["lead_h"], errors="coerce"),
        AROC=pd.to_numeric(frame["AROC"], errors="coerce"),
        BSS=pd.to_numeric(frame["BSS"], errors="coerce"),
    )
    by_lead = numeric.groupby("lead_h")[["AROC", "BSS"]].mean().dropna(how="all")
    return {"rows": rows, "average": average, "by_lead": by_lead}


def _probability_window(frame: pd.DataFrame) -> Optional[float]:
    """概率宽表的累积窗口（``window_h`` 列）——表题与引用块都要写它。"""
    if "window_h" not in frame.columns:
        return None
    values = pd.to_numeric(frame["window_h"], errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else None


def _threshold_list(frame: pd.DataFrame) -> str:
    """``0.1/4/13/25mm``——引用块里要和 TS 的阈值列表并排写，得各自现取。"""
    values = sorted(
        {float(value) for value in pd.to_numeric(frame["threshold_mm"], errors="coerce").dropna()}
    )
    return "/".join(f"{value:g}" for value in values) + "mm" if values else "—"


def _probability_note(window: Optional[float]) -> str:
    """概率评分表下的口径引用块——与骨架第六节逐字一致。

    只有累积窗口那一处是变量：骨架写死的 ``6h`` 就是本仓库当前的概率窗口常量，
    换个窗口这句话会变成假话，所以从表里现取。
    """
    span = f"{window:g}h" if window else "概率"
    return (
        f"> BS_ref 是气候基准的 Brier 分数，**只有外部给了气候概率参考的 {span} 窗才有值**，"
        f"其余窗口整列写 —；BSS = 1 − BS/BS_ref。AROC 在**逐成员插值后**按超越式阈值"
        f'（x ≥ 阈值）计算，与前面章节"插值到站再配对"的做法不是一回事。'
        f"AVG 行是对**可算阈值**的等权平均（该档无事件的跳过），不是按样本数加权。"
    )


def _probability_table(
    summaries: Dict[str, Dict[str, object]], *, with_model: bool
) -> str:
    """概率评分表：一行一个「模型 × 降水等级」，每个模型末尾再加一行 ``AVG``。

    ``with_model`` 由调用方按模型数决定——骨架单模型分支的表头里没有「模型」列，
    多模型分支才需要它来分清是谁的行。
    """
    header = (["模型"] if with_model else []) + [
        "降水等级", "阈值(mm)", "AROC", "BS", "BS_ref", "BSS", "基准率", "样本点数",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, summary in summaries.items():
        for entry in list(summary["rows"]) + [summary["average"]]:
            # AVG 行没有单一阈值/基准率/样本数，那三格写 —；末格改成**参与平均的档数**
            # （老表里那一列的 ``n_used`` 就是这个意思），比一个空的「—」有信息量。
            cells = [
                _fmt(entry.get("threshold"), 3),
                _fmt(entry.get("AROC")),
                _fmt(entry.get("BS")),
                _fmt(entry.get("BS_ref")),
                _fmt(entry.get("BSS")),
                _fmt(entry.get("base_rate")),
                (
                    f"{entry['n_points']:,}" if entry.get("n_points")
                    else (f"{entry['n_grades']} 档" if entry.get("n_grades") else "—")
                ),
            ]
            prefix = [str(name)] if with_model else []
            lines.append("| " + " | ".join(prefix + [str(entry["grade"])] + cells) + " |")
    return "\n".join(lines)


def _section_probability(
    probability: Dict[str, pd.DataFrame],
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    *,
    number: str = "六",
    members: Optional[int] = None,
    ts_window: Optional[float] = None,
    ts_thresholds: str = "—",
) -> List[str]:
    """「概率评分（AROC / BSS）」一节——只在拿到概率宽表时出现。

    引导语、引用块、表头都照骨架写；只有那一堆数字和窗口/阈值是从表里现取的。
    ``number`` 是章号（默认「六」）：这一节之后所有节号都要顺延，所以由调用方给。
    """
    summaries = {name: probability_summary(frame) for name, frame in probability.items()}
    primary = next(iter(probability))
    window = _probability_window(probability[primary])
    window_text = f"{window:g}h" if window else "概率"

    leads = sorted(
        float(value) for value in pd.to_numeric(probability[primary]["lead_h"], errors="coerce").dropna()
    )
    span = f"{leads[0]:g}–{leads[-1]:g}h" if leads else "各时效"

    lines = [f"## {number}、概率评分（AROC / BSS，{window_text} 累积窗口）", ""]
    lines += [
        f"前面五节评的都是{_members_phrase(members)}的**平均场**，成员之间的离散信息"
        f"一点没进指标——一个成员高度一致、另一个成员四散分布的集合，前五节的分数"
        f"可以完全相同。本节改用全体成员的概率分布：AROC 衡量预报概率把事件与非事件"
        f"分开的能力（0.5 为无技巧），BSS 衡量相对气候基准的 Brier 技巧（>0 才算有正技巧）。",
        "",
        f"**本节与前面五节不同口径**：概率评分用 {window_text} 累积窗口、"
        f"阈值 {_threshold_list(probability[primary])}，与前面 "
        + (f"{ts_window:g}h / " if ts_window else "")
        + f"{ts_thresholds} 既不可比，也不要画进同一张图。",
        "",
    ]
    lines += _table_caption(
        numbering, "各降水等级的概率评分（各时效等权平均；AVG 为可算阈值的等权平均）"
    )
    lines.append(_probability_table(summaries, with_model=len(summaries) > 1))
    lines.append("")
    lines.append(_probability_note(window))
    lines.append("")
    caption = (
        f"{span} 概率评分的 AROC（上排，虚线为 0.5 的无技巧线）与 BSS（下排，虚线为 0 的"
        f"无正技巧线）随时效变化，每列一个阈值；AROC 越靠上区分能力越强，"
        f"BSS 越靠上越优于气候基准"
    )
    lines += _figure_block(artifacts, numbering, "AROC_BSS_vs_lead", caption)
    lines.append(f"**小结**：{analysis_probability(summaries)}")
    lines.append("")
    return lines


def _figure_captions(
    summary: Dict[str, object], *, window: Optional[float], lead: float, model_name: str
) -> Dict[str, str]:
    """每张图的图注——说明**怎么读这张图**，而不是重复图的标题。"""
    leads = summary["leads"]
    span = f"{leads[0]:g}–{leads[-1]:g}h"
    window_text = f"{window:g}h 累积窗口" if window else "本次评估窗口"
    return {
        "metrics_by_threshold": (
            f"{lead:g}h 时效（{window_text}）各降水等级的 TS、POD、FAR 与 BIAS 对比"
        ),
        "TS_vs_lead": (
            f"{span} 各降水等级的 TS 曲线；线越靠上技巧越好，"
            f"虚线表示该等级事件数不足 50、仅作参照"
        ),
        "TS_heatmap": (
            f"降水等级 × 预报时效的 TS 热力图（{window_text}）；"
            f"颜色越深技巧越高，格子里的数字随底色自动切黑/白"
        ),
        "POD_vs_lead": (
            f"{span} 各降水等级的命中率 POD；POD 低意味着漏报多，"
            f"但单独抬高 POD 往往以增大空报为代价，需与下一张 FAR 图联合判读"
        ),
        "FAR_vs_lead": f"{span} 各降水等级的空报率 FAR（纵轴锁定 0–1）；FAR 越低误报越少",
        "BIAS_vs_lead": (
            f"{span} 各降水等级的频率偏差 BIAS，虚线为理想值 1（纵轴自适应、不截断）；"
            f"曲线在 1 上方表示预报事件偏多、下方表示偏少"
        ),
        "performance_diagram": (
            f"{lead:g}h 时效的性能图：横轴为成功比 1−FAR、纵轴为命中率 POD，"
            f"灰色虚线为 TS 等值线、蓝色点线为 BIAS 参考线；点越靠近右上角技巧越好"
        ),
        "TS_boxplot": (
            "各模型 TS 的跨时效分布；箱体越高表示该量级整体技巧越好、"
            "越窄表示跨时效越稳定，空心圆为离群时效"
        ),
        "multi_model_TS_delta": (
            f"{span} 各对比模型相对 {model_name} 的 TS 差值"
            f"（纵轴是「对比模型 − 本模型」：正值表示对比模型领先，负值表示本模型领先）"
        ),
    }


# ------------------------------------------------------------- 能力雷达图
#
# 选轴有一条硬约束：**恒等式的两端不能各占一根轴**。TS = f(POD, FAR)
# （1/TS = 1/POD + 1/(1−FAR) − 1），三根同放是同一件事计三次分，多边形会被这组
# 强相关的轴带偏——POD / FAR 的绝对水平回第四节那张表里看，不进雷达图。

#: 雷达图至少两个模型才画：一个多边形没有比较对象。
CAPABILITY_MIN_MODELS = 2

#: 强降水那根轴取哪一档：不低于这个阈值（mm）的**最小**一档。
HEAVY_RAIN_THRESHOLD = 25.0

_CN_NUM = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}


def _model_count_word(count: int) -> str:
    """模型数的中文写法——报告其余部分一律写「二模型」，这里不能冒出「2模型」。"""
    return _CN_NUM.get(int(count), str(int(count)))


def _heavy_grade_threshold(models: Dict[str, pd.DataFrame]) -> Optional[float]:
    """强降水轴的档位门槛，**全批取同一个**。

    逐模型各取各的会出现「甲模型比 ≥25mm、乙模型比 ≥50mm」——名次没法解释，
    而且雷达图上完全看不出来。所以按全批的阈值集合定一档，缺这一档的模型
    那条轴按缺值处理（整根剔除，见 ``radar_scores``）。
    """
    thresholds = set()
    for frame in models.values():
        if "threshold_mm" not in frame.columns:
            continue
        values = pd.to_numeric(frame["threshold_mm"], errors="coerce").dropna()
        thresholds |= {float(value) for value in values if float(value) >= HEAVY_RAIN_THRESHOLD}
    return min(thresholds) if thresholds else None


def capability_profile(
    models: Dict[str, pd.DataFrame],
) -> Tuple[List[str], Dict[str, Dict[str, Optional[float]]], Dict[str, bool]]:
    """雷达图的三样东西：轴序、``{模型: {轴: 原始值}}``、各轴方向。

    四根轴都取自前面几节**已经算好的统计量**（``summarize`` 的总体 TS 与 BIAS、
    ``box_stats`` 的跨时效 IQR），不另算一套指标——图、表、正文同源。

    返回的取值**还没归一化**：归一化只有 ``det_plots.radar_scores`` 一处实现，
    绘图和正文的判读句都从那里拿分。
    """
    heavy = _heavy_grade_threshold(models)
    heavy_label = f"强降水 TS（≥{heavy:g}mm）" if heavy is not None else "强降水 TS"
    labels = [
        "TS（全等级等权平均）",
        heavy_label,
        "|BIAS−1|（全等级平均）",
        "跨时效 TS 的 IQR（全等级平均）",
    ]
    higher = {
        labels[0]: True,
        labels[1]: True,
        labels[2]: False,
        labels[3]: False,
    }

    raw: Dict[str, Dict[str, Optional[float]]] = {}
    for name, frame in models.items():
        summary = summarize(frame)
        grades = summary["grades"]
        heavy_entry = None
        if heavy is not None:
            heavy_entry = next(
                (
                    item
                    for item in grades
                    if item["threshold"] is not None
                    and np.isclose(float(item["threshold"]), heavy)
                ),
                None,
            )
        # BIAS 与 IQR 都按等级等权平均：各档样本量差几个量级，按样本数加权会让
        # 小量级那几档把整根轴吃掉，而这根轴问的是「整体偏差/整体稳定性」。
        bias = [
            abs(float(item["BIAS"]) - 1.0)
            for item in grades
            if np.isfinite(item.get("BIAS", np.nan))
        ]
        iqr = [
            float(box_stats(frame[frame["grade"] == item["grade"]], "TS")["iqr"])
            for item in grades
        ]
        iqr = [value for value in iqr if np.isfinite(value)]
        raw[str(name)] = {
            labels[0]: float(summary["overall_ts"]),
            labels[1]: None if heavy_entry is None else float(heavy_entry["TS_sum"]),
            labels[2]: float(np.mean(bias)) if bias else None,
            labels[3]: float(np.mean(iqr)) if iqr else None,
        }
    return labels, raw, higher


def _section_grade(
    summary: Dict[str, object], artifacts, numbering: _Numbering, captions: Dict[str, str]
) -> List[str]:
    grades = summary["grades"]
    lines = ["## 二、逐降水等级表现（全时效平均）", ""]
    lines.append(
        "下表汇总各降水等级在整个时效范围内的表现，"
        "用于判断模型在不同量级上的技巧水平与偏差结构。"
    )
    lines.append("")
    lines += _table_caption(numbering, "各降水等级的全时效平均技巧与列联表计数（按阈值升序）")
    lines.append(_grade_table(grades))
    lines.append("")
    lines.append(
        "> 全时效平均为**各时效等权平均**（不是按样本数加权）；判断强降水能力请看 ≥25mm 及以上等级。"
    )
    lines.append("")
    lines += _figure_block(artifacts, numbering, "metrics_by_threshold", captions["metrics_by_threshold"])
    lines.append(f"**小结**：{analysis_by_grade(grades)}")
    lines.append("")
    return lines


def _section_lead(
    summary: Dict[str, object], artifacts, numbering: _Numbering, captions: Dict[str, str]
) -> List[str]:
    leads = summary["leads"]
    lines = [f"## 三、TS 随时效变化（{leads[0]:g}–{leads[-1]:g}h）", ""]
    lines.append(
        "按预报时效展开，用于确定该模型的有效预报时效——技巧掉到不可用之前能推多远。"
    )
    lines.append("")
    for name in ("TS_vs_lead", "TS_heatmap"):
        lines += _figure_block(artifacts, numbering, name, captions[name])
    lines += _table_caption(numbering, "各降水等级在各预报时效上的 TS")
    lines.append(_lead_table(summary["pivot"]))
    lines.append("")
    lines.append(
        "> 单元格是该时效该等级的 TS；同一行内横向比较看衰减速度，同一列内纵向比较看量级差异。"
    )
    lines.append("")
    lines.append(f"**小结**：{analysis_by_lead(summary)}")
    lines.append("")
    return lines


def _section_error_shape(
    summary: Dict[str, object], artifacts, numbering: _Numbering, captions: Dict[str, str]
) -> List[str]:
    lines = ["## 四、误差形态与偏差结构", ""]
    lines.append(
        "技巧高低之外还要看误差从哪来：本节从命中率、空报率与频率偏差三个方面拆解误差的来源与形态。"
    )
    lines.append("")
    for name in ("POD_vs_lead", "FAR_vs_lead", "BIAS_vs_lead", "performance_diagram"):
        lines += _figure_block(artifacts, numbering, name, captions[name])
    lines.append(f"**小结**：{analysis_error_structure(summary['grades'])}")
    lines.append("")
    return lines


def _section_boxplot(
    models: Dict[str, pd.DataFrame],
    artifacts,
    numbering: _Numbering,
    captions: Dict[str, str],
    primary: str,
) -> List[str]:
    lines = ["## 五、跨时效分布（箱线图，TS）", ""]
    lines.append(
        "把同一模型、同一量级在各时效上的 TS 视为一个样本，用箱线图刻画其离散程度——"
        "平均技巧相同的两个模型，稳定性可能差很远。"
    )
    lines.append("")
    lines += _table_caption(numbering, "各模型 × 各降水等级 TS 的跨时效分布统计量")
    lines.append(_box_table(models))
    lines.append("")
    lines.append(_BOX_NOTE)
    lines.append("")
    lines += _figure_block(artifacts, numbering, "TS_boxplot", captions["TS_boxplot"])
    lines.append(f"**小结**：{analysis_stability(models, primary)}")
    lines.append("")
    return lines


def _section_comparison(
    data: pd.DataFrame,
    comparisons: Dict[str, pd.DataFrame],
    artifacts,
    numbering: _Numbering,
    captions: Dict[str, str],
    *,
    lead: float,
    heading: str,
    caption: str,
    note: Optional[str],
) -> List[str]:
    lines = [heading, ""]
    lines.append("以下比较在同一时效、同一阈值口径下进行。")
    lines.append("")
    lines += _table_caption(numbering, caption)
    lines.append(_table_body_of(comparisons, data, lead))
    lines.append("")
    if note:
        lines.append(note)
        lines.append("")
    lines += _figure_block(artifacts, numbering, "multi_model_TS_delta", captions["multi_model_TS_delta"])
    lines.append(f"**小结**：{analysis_comparison(data, comparisons, lead)}")
    lines.append("")
    return lines


def _table_body_of(
    comparisons: Dict[str, pd.DataFrame], data: pd.DataFrame, lead: float
) -> str:
    """单基准是多模型的一个特例——表体按对比对象的个数选，别让两个分支各写一遍。"""
    if len(comparisons) == 1:
        name, frame = next(iter(comparisons.items()))
        return _baseline_table(data, frame, lead)
    return _multi_baseline_table(data, comparisons, lead)


def _section_capability(
    models: Dict[str, pd.DataFrame],
    numbering: _Numbering,
    *,
    heading: str,
    radar_path: Optional[Path],
) -> List[str]:
    """「模型能力雷达图」一节——综合评分的批内归一化对比。

    轴与原始值来自 :func:`capability_profile`，归一化来自 ``det_plots.radar_scores``，
    这一节只把结果写成文；判读句也读同一份 ``scores``，免得图和文字各算一遍。
    """
    # 延迟导入：``det_plots`` 在模块层 import 了 ``precipitation_plots``，后者又
    # import 本模块——放模块层会绕成一个环，``import ts_report`` 直接失败。
    from xmetai_evaluation.visualization.det_plots import radar_scores

    axes, raw, higher = capability_profile(models)
    scores = radar_scores(raw, axes, higher)
    names = [str(name) for name in models]
    kept = [axis for axis in axes if names and axis in scores[names[0]]]
    dropped = [axis for axis in axes if axis not in kept]

    lines = [heading, ""]
    lines += [
        "这一节把前面几张表的**综合评分**摆成多边形：每根轴一种能力，每个模型一个",
        "多边形，**越靠外越好**。它不引入任何新算法，只是把前面的数换个读法——",
        "看的是「能力形状」，不是「谁排第一」：两个模型综合分接近时，雷达图能显出",
        "差距集中在哪几项上。",
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
        "**读法**：纵轴 0–1 是**批内相对分**，每根轴上 1 = 本批最好、0 = 本批最差",
        "（各轴方向先行统一）。"
        f"这是{_model_count_word(len(names))}模型互比出来的相对位置，**不是绝对能力分**",
        "——全批都差时每根轴照样有人拿 1；只有两个模型时必然是 1 和 0。某一根轴上所有",
        "模型打平时，该轴一律画在满格 1.0（画成 0 会让多边形凭空凹进去），所以**满格",
        "不代表领先**，要回前面几节看绝对数值。",
        "",
    ]

    if radar_path is not None and Path(radar_path).exists():
        caption = (
            f"{_model_count_word(len(names))}模型能力雷达图（{len(kept)} 根综合评分轴；"
            "各轴批内 min-max 归一化，1 = 本批最好）"
        )
        lines += [
            f"![capability_radar]({Path(radar_path).name})",
            "",
            f"图 {numbering.figure('capability_radar')}：{caption}",
            "",
        ]
    else:
        # 走到这里说明图没落盘。轴有空缺本来就会整根剔除、不至于一根不剩，
        # 所以这不是「这批数据没跑到」，是渲染器或调用方出了问题。
        lines += [
            "**本批未出。** 渲染器没拿到雷达图的落盘路径。这一节**不像别处那样"
            "依赖可选数据**——综合评分几根轴每个批次都有，正常跑必然出图；"
            "缺了就是渲染器或调用方出了问题，该回去查，不要当成「这批数据没跑到」。",
            "",
        ]

    if names and kept:
        outer = max(names, key=lambda name: sum(scores[name].values()))
        inner = min(names, key=lambda name: sum(scores[name].values()))
        if outer != inner:
            lines += [
                f"多边形面积最大的是 {outer}（各轴得分之和 "
                f"{_fmt(sum(scores[outer].values()), 2)}），"
                f"最小的是 {inner}（{_fmt(sum(scores[inner].values()), 2)}）。",
                "",
            ]
    return lines


# ------------------------------------------------------------- 分纬度带（附 L）
#: 分带小节的图名：既是 ``artifacts`` 里的键，也是报告里 ``![…]()`` 的标签。
REGION_FIGURE = "region_TS_vs_lead"

#: 这份产物**一条带都没落数**时的说明。保留节位、写清原因，不静默省略——
#: 与 ``weather_rmse_wave.md`` 里球谐带那几节、``det_report`` 的「附 L」同一套约定。
_REGION_MISSING_NOTE = (
    "这份产物的宽表里没有 `region` 行：本次评测没配 `options[\"regions\"]`"
    "（或者配了但没有任何一条带算出结果）。"
)


def region_labels_from(root: Path) -> Dict[str, str]:
    """``{带名: 边界串}``，边界取自产物 ``manifest["regions"]``（形如 ``south[0, 15]``）。

    带名与边界都**照搬配置**的 ``options["regions"]``：报告不翻译、也不写死任何
    带名，配置里叫什么就显示什么；要换一套切法改配置重跑即可，报告这边一个字
    都不用动。读不到 manifest（直接调 ``build_ts_report``、产物目录不在手边）就
    返回空——那时带宽只显示带名，不编一个边界出来。

    实现复用 ``det_report`` 的同一份解析：``manifest["regions"]`` 这个键的形状
    只有一处定义，两份报告各解析一遍迟早对不上。**延迟导入**是因为
    ``precipitation_plots`` 在模块层 import 了本模块，本模块在模块层再 import
    回去就绕成一个环。
    """
    from xmetai_evaluation.visualization.det_report import region_labels

    try:
        return region_labels(Path(root))
    except ValueError:  # 没有 manifest.json，或它不是一个合法 JSON
        return {}


def _band_order(names: Sequence[str], labels: Optional[Dict[str, str]]) -> List[str]:
    """带名按**南界从低到高**排。

    纬度带天生分南北，按名字排中文是乱序（东北 / 南方 / 华北 / 长江中下游），
    一张按纬度排好的表读者一眼就能扫完。边界只在 manifest 里，取不到（老产物、
    直接调 ``build_ts_report``）时退回按名字升序——不编一个边界出来排。
    """
    def key(name: str):
        head = str((labels or {}).get(name, "")).split(",")[0].strip()
        try:
            return (0, float(head), name)
        except ValueError:
            return (1, 0.0, name)

    return sorted(names, key=key)


def _region_table(
    band_frames: Dict[str, pd.DataFrame], labels: Optional[Dict[str, str]]
) -> str:
    """带 × 降水等级 的 TS 表，末列是各带自己的「全等级等权平均」。

    口径与第二节、能力雷达图的综合轴**完全同源**——都走 ``summarize``
    （先把列联表合计再算比率，再对各等级等权平均），不在这里另算一遍：
    同一个「综合」两处各算一遍，迟早只在一处修好。
    """
    from xmetai_evaluation.visualization.det_report import region_display

    grades = [str(grade) for grade in ordered_grades(list(band_frames.values()))]
    header = ["纬度带"] + grades + ["综合"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in _band_order(list(band_frames), labels):
        frame = band_frames[name]
        summary = summarize(frame)
        by_grade = {entry["grade"]: entry for entry in summary["grades"]}
        cells = [_fmt(by_grade[grade]["TS"]) if grade in by_grade else "—" for grade in grades]
        lines.append(
            "| "
            + " | ".join(
                [region_display(name, labels or {})] + cells + [_fmt(summary["overall_ts"])]
            )
            + " |"
        )
    return "\n".join(lines)


def _region_reading(
    band_frames: Dict[str, pd.DataFrame], labels: Optional[Dict[str, str]]
) -> str:
    """分带判读句：最强的带、最弱的带、差多少，以及头名是否随预报时效翻转。

    「翻转」按各带**逐时效**的 TS（等级等权平均）比：整段平均出来的名次在
    某些时效上未必成立，而这正是分带要回答的问题。
    """
    overall = {
        name: float(summarize(frame)["overall_ts"]) for name, frame in band_frames.items()
    }
    finite = {name: value for name, value in overall.items() if np.isfinite(value)}
    if len(finite) < 2:
        return "各带的有效样本不足以横向比较（至少要有两条带各自算出 TS）。"

    # 并列判定按**表里显示出来的精度**（三位小数）：表上两行一模一样却写「相差
    # 0.000」，或者硬点一个第一名出来，都是文字和图各说各话。
    ranked = sorted(finite.items(), key=lambda item: item[1], reverse=True)

    def _tied(value: float) -> List[str]:
        key = _fmt(value)
        return [name for name, item_value in ranked if _fmt(item_value) == key]

    tops, bottoms = _tied(ranked[0][1]), _tied(ranked[-1][1])
    if tops == bottoms:
        text = (
            f"各带的全等级等权平均 TS 都是 {_fmt(ranked[0][1])}，"
            f"带间分不出高下（差距小于表里的显示精度）。"
        )
        best_names = set(tops)
    else:
        text = (
            f"各带之中全等级等权平均 TS 最高的是 {_join_bands(tops, labels)}"
            f"（{_fmt(ranked[0][1])}），最低的是 {_join_bands(bottoms, labels)}"
            f"（{_fmt(ranked[-1][1])}），相差 {_fmt(ranked[0][1] - ranked[-1][1])}"
            f"{'——并列。' if len(tops) > 1 else '。'}"
        )
        best_names = set(tops)

    by_lead = {
        name: frame.groupby("lead_h")["TS"].mean() for name, frame in band_frames.items()
    }
    leads = sorted({float(lead) for series in by_lead.values() for lead in series.index})
    flips = 0
    for lead in leads:
        values = {
            name: float(series.get(lead, np.nan))
            for name, series in by_lead.items()
            if lead in series.index
        }
        values = {name: value for name, value in values.items() if np.isfinite(value)}
        if not values:
            continue
        lead_best = max(values.values())
        lead_tops = {name for name, value in values.items() if _fmt(value) == _fmt(lead_best)}
        if not lead_tops & best_names:
            flips += 1
    if flips:
        text += (
            f" 名次并不固定：{len(leads)} 个时效里有 {flips} 个时效的头名是别的带——"
            f"带间差异随时效变化，不能只拿全时效平均下结论。"
        )
    else:
        text += f" {len(leads)} 个时效的头名都在上面那几条带里，带间差异在全时效上一致。"
    return text


def _join_bands(names: Sequence[str], labels: Optional[Dict[str, str]]) -> str:
    """并列的带名连起来写（``south [0, 15]、north [15, 30]``）。"""
    from xmetai_evaluation.visualization.det_report import region_display

    return "、".join(region_display(name, labels or {}) for name in names)


def _section_regions(
    band_frames: Dict[str, pd.DataFrame],
    labels: Optional[Dict[str, str]],
    artifacts: Optional[Dict[str, Path]],
    figure: Optional[Path],
) -> List[str]:
    """「附 L 分纬度带结果（可选）」——空数据时保留节位并写「本批未出」。

    图号走**独立命名空间** `图 L1`：这一块是可选的，删掉它不该动正文任何一个
    编号（与 ``weather_rmse_single.md`` 的「附 L / 附 S」同一条约定）。
    """
    lines = ["## 附 L 分纬度带结果（可选）", ""]
    if not band_frames:
        lines += [f"**本块本批未出。** {_REGION_MISSING_NOTE}", ""]
        return lines

    lines += [
        "本块把宽表里 `region` 非空的行**单独汇总**，全球行**不参与**——全球平均与",
        "纬度带平均是两个量，混在一起算会把带间差异整个抹平，这一块也就没有意义了。",
        "带名与边界照搬评测配置 `options[\"regions\"]`：**报告不翻译、也不写死任何带名**，",
        "配置里叫 `south` 就显示 `south`；要换一套切法改配置重跑即可，报告这边一个字",
        "都不用动。表里的口径与第二节完全一致（先合计列联表再算比率，再对等级等权平均），",
        "末列的「综合」就是该带自己的全等级等权平均。",
        "",
        "**分纬度带 TS（各降水等级，全时效平均）**",
        "",
        _region_table(band_frames, labels),
        "",
        _region_reading(band_frames, labels),
        "",
        "> 这一块**只汇总主模型**：按带切和按模型切是两个正交的维度，硬塞进同一张表，",
        "> 会出现「同一个名字既是带名又是模型名」的歧义。另外，各带的站点数差得远时",
        "> TS 的抽样噪声也不一样，差一两个百分点不必当结论。",
        "",
    ]
    if figure is not None and REGION_FIGURE in (artifacts or {}):
        lines += [
            f"![{REGION_FIGURE}]({figure.name})",
            "",
            "图 L1：各纬度带 TS 随时效变化（每个降水等级一个子图，纵轴 0–1）",
            "",
        ]
    return lines


def build_ts_report(
    data: pd.DataFrame,
    output_path: Path,
    *,
    model_name: str = "model",
    baseline_df: Optional[pd.DataFrame] = None,
    baseline_name: Optional[str] = None,
    baselines: Optional[Dict[str, pd.DataFrame]] = None,
    sources: Optional[Dict[str, str]] = None,
    change_description: Optional[str] = None,
    artifacts: Optional[Dict[str, Path]] = None,
    lead_h: Optional[float] = None,
    box_models: Optional[Dict[str, pd.DataFrame]] = None,
    region_frames: Optional[Dict[str, pd.DataFrame]] = None,
    region_labels: Optional[Dict[str, str]] = None,
    probability: Optional[Dict[str, pd.DataFrame]] = None,
    members: Optional[int] = None,
    coverage: Optional[Dict[str, int]] = None,
    ensemble: bool = False,
) -> Path:
    """写一份 Markdown 报告。

    每一节都是「引导句 → 图表 → 小结」：图就地带着 `图 N：` 图注出现，
    表上方带 `表 N：` 表题，节末一段带具体数值的分析。末尾不再有「图表」索引节——
    图都在正文里，再列一遍只是重复，还会让人以为是两张不同的图。

    Args:
        box_models: 第五节箱线图的模型集合（本模型 + 各对比模型）。
            由 ``create_report`` 算好传进来，与它画图用的是**同一个 dict**；
            不传则按 ``model_name`` + ``baseline_df``/``baseline_name`` + ``baselines``
            自己拼一份（直接调本函数的场景，比如测试）。
        region_frames: ``{带名: 该带的宽表行}``，进末尾的「附 L」。不传就自己从
            ``data`` 的 ``region`` 列里切一份（见 ``precipitation_plots.split_regions``）；
            ``create_report`` 已经把全球行筛出来传进来了，那时从 ``data`` 里切只能
            得到空集，所以它显式传这个参数。
        region_labels: ``{带名: 边界串}``，只影响带名怎么写（``south [0, 15]``），
            取不到就只显示带名。
        probability: ``{模型名: 概率宽表}``（``diagnostics/probability_wide.csv``，
            含主模型）。给了就多出「概率评分（AROC / BSS）」一节，**后面所有章节号
            顺延一位**；不给（确定性产物、或集合没跑 ``weather_ts_ens_prob`` 那一段）
            整节不出现，章节号原样。
        members: 主模型的集合成员数（``--members``）。产物里没有这个字段，
            拿不到时相关句子里不写数字。只有 ``ensemble=True`` 时用得上。
        coverage: ``{模型名: 有效配对数}``，进第一节的实测覆盖句。本报告的对比表
            只写分数、不写样本量，而各份来源的起报日数可以差一个量级——不写出来，
            读者会把样本最少的那份的名次当成结论。
        ensemble: 是不是集合产物。打开才出集合专属措辞（标题的「集合预报」、
            第一节的「集合口径」段、第十节的两条集合提示）。**默认关**：确定性与
            FDP 两条流程共用本模块，骨架与逐字测试都对着它们，不能被牵连。
    """
    output_path = Path(output_path)
    # 全球行与带行分开算：``data`` 里混着带行时，第二～五节会把同一天算好几遍。
    # 传进来的已经是全球行（create_report 切过）时这一步是恒等的。
    # 延迟导入：``precipitation_plots`` 在模块层 import 了本模块。
    from xmetai_evaluation.visualization.precipitation_plots import split_regions

    data, derived_regions = split_regions(data)
    if region_frames is None:
        region_frames = derived_regions
    summary = summarize(data)
    findings = diagnose(summary)
    grades = summary["grades"]
    leads = summary["leads"]
    window = None
    if "window_h" in data.columns and data["window_h"].notna().any():
        window = float(data["window_h"].dropna().iloc[0])
    lead = float(lead_h) if lead_h is not None else leads[0]

    # 模型集合由 create_report 算好传进来（它同时拿去画图）；直接调 build_ts_report 的
    # 调用方没给，就按同样的规则自己拼一份，别让表和图用两套模型集合。
    models = box_models if box_models is not None else collect_box_models(
        model_name,
        data,
        baseline_df=baseline_df,
        baseline_name=baseline_name,
        baselines=baselines,
    )
    # 单基准和多模型走同一条比较路径——单基准只是"一个对比对象"的特例
    comparisons: Dict[str, pd.DataFrame] = dict(baselines or {})
    if baseline_df is not None:
        comparisons.setdefault(baseline_name or "基准", baseline_df)

    # 概率宽表统一在这里过一遍校验：空表 / 缺列在这里就报出清晰的错，而不是等到
    # 第六节算到一半才发现是张别的表。``None`` 与空 dict 等价——都表示这份产物
    # 没有概率评分那一段（确定性产物、或集合没跑 weather_ts_ens_prob）。
    # 注意别把结果写回 ``probability`` 自己：``probability = {}`` 先执行、
    # 循环头里的 ``(probability or {})`` 后求值，等于遍历空 dict，整节会静默消失。
    loaded_tables: Dict[str, pd.DataFrame] = {}
    for name, frame in (probability or {}).items():
        loaded = load_probability(frame)
        if loaded is not None:
            loaded_tables[str(name)] = loaded
    probability = loaded_tables

    # 雷达图：模型集合与第五节箱线图**用的是同一个 dict**（`models`），免得
    # "图里少一个模型"这种错从后门溜回来。单模型不画——一个多边形没有比较对象。
    prefix = f"{window:g}h_" if window else ""
    radar = len(models) >= CAPABILITY_MIN_MODELS
    radar_path: Optional[Path] = None
    if radar:
        radar_path = output_path.parent / f"{prefix}capability_radar.png"
        axes, raw, higher = capability_profile(models)
        # 延迟导入：``det_plots`` 在模块层 import 了 ``precipitation_plots``，
        # 后者又 import 本模块——放模块层会绕成一个环。
        from xmetai_evaluation.visualization.det_plots import DetPlotter

        output_path.parent.mkdir(parents=True, exist_ok=True)
        DetPlotter().plot_capability_radar(
            raw, axes, higher,
            suptitle=f"{_model_count_word(len(models))}模型能力雷达图"
                     f"（批内相对分，1 = 本批最好）",
            save_path=radar_path,
        )
        if artifacts is not None and radar_path.exists():
            artifacts["capability_radar"] = radar_path

    # 分带图与雷达图同一套做法：要画才画，且只有真的落盘了才登记进 artifacts
    # （登记了但没写进正文，`test_every_artifact_is_placed_in_the_report` 会红）。
    region_figure: Optional[Path] = None
    if region_frames:
        region_figure = output_path.parent / f"{prefix}region_TS_vs_lead.png"
        from xmetai_evaluation.visualization.precipitation_plots import PrecipitationPlotter

        PrecipitationPlotter().plot_multi_model_vs_lead(
            dict(region_frames),
            metric="TS",
            mode="absolute",
            legend_title="纬度带",
            suptitle="各纬度带 TS 随时效变化（每个降水等级一个子图）",
            save_path=region_figure,
        )
        if artifacts is not None and region_figure.exists():
            artifacts[REGION_FIGURE] = region_figure

    numbering = _Numbering()
    captions = _figure_captions(summary, window=window, lead=lead, model_name=model_name)

    lines: List[str] = []
    # 集合产物的标题按骨架写成「集合预报N模型降水评估报告」；确定性那份保持
    # 「{模型名}降水评估报告」不变——它有自己的骨架与逐字测试。
    title = (
        f"集合预报{_model_count_word(len(models))}模型降水评估报告"
        if ensemble else f"{model_name}降水评估报告"
    )
    lines.append(f"# {title}")
    lines.append("")

    # 评估对象/口径与结论合成同一节：先交代「评的是谁、多少天、什么指标」，
    # 再一整段给结论，末尾两句固定口径声明。
    if comparisons:
        contrast = (
            "对比对象为"
            + "、".join(f"{name}（来源：{_source_of(sources, name)}）" for name in comparisons)
            + ("——各者应使用同一时段" if len(comparisons) > 1 else "——两者应使用同一时段")
            + "、同一站点集、同一窗口与阈值口径，"
            "若来源文件的口径不同，结论不可直接横向比较。"
        )
    else:
        contrast = "未设对比对象。"

    lines.append("## 一、结论摘要")
    lines.append("")
    lines.append(
        f"本报告评估{model_name}（来源：{_source_of(sources, model_name)}），"
        + (f"集合每例{int(members)}个成员，" if (ensemble and members) else "")
        + contrast
        + _coverage_sentence(coverage)
        + "评估口径为站点分类检验（"
        + (f"{window:g}h 累积窗口，" if window else "")
        + f"时效 {leads[0]:g}–{leads[-1]:g}h 共 {len(leads)} 个，{summary['n_rows']} 条记录），"
        + "指标为 TS、POD、FAR、漏报率与 BIAS。"
        + (f"本次改动：{change_description}。" if change_description else "")
        + f"报告生成时间：{datetime.now():%Y-%m-%d %H:%M}。"
    )
    lines.append("")
    # 结论并成一段而不是逐条列点：读起来是判断，不是清单
    lines.append("".join(findings))
    lines.append("")
    lines.append(f"**业务定性**：{business_conclusion(grades)}")
    lines.append("")
    if ensemble:
        # 集合产物必须点明：这五个指标算在**平均场**上，成员离散信息一点没进去。
        # 不写这一句，读者会把「集合」当成「集合技巧比确定性好」，而这里根本没比。
        lines.append(
            f"集合口径：TS / POD / FAR / 漏报率 / BIAS 均在{_members_phrase(members)}的"
            f"**平均场**上计算（确定性口径），不是逐成员评分后再平均；"
            + (
                "成员离散信息只在第六节的概率评分里体现。"
                if probability
                else "本批没有概率评分表，成员离散信息在这份报告里没有体现。"
            )
        )
        lines.append("")
    lines.append(
        "指标定义：TS = hits/(hits+misses+false_alarms)，POD = hits/(hits+misses)，"
        "FAR = false_alarms/(hits+false_alarms)，漏报率 = 1−POD，"
        "BIAS = (hits+false_alarms)/(hits+misses)（频率偏差）"
    )
    lines.append("")

    lines += _section_grade(summary, artifacts, numbering, captions)
    lines += _section_lead(summary, artifacts, numbering, captions)
    lines += _section_error_shape(summary, artifacts, numbering, captions)
    lines += _section_boxplot(models, artifacts, numbering, captions, model_name)

    # 章节号跟着**实际出现了哪几节**走：概率评分与雷达图都是可选的，每出现一节，
    # 后面所有节号顺延一位。写死数字会在缺节时留下断号。
    next_number = 6
    if probability:
        lines += _section_probability(
            probability, artifacts, numbering,
            number=_CN_NUM[next_number],
            members=members,
            ts_window=window,
            ts_thresholds=_threshold_list(data),
        )
        next_number += 1

    if comparisons:
        if baselines:
            heading = f"## {_CN_NUM[next_number]}、与对比模型的比较（{lead:g}h，TS）"
            caption = f"{lead:g}h 时效上各模型的 TS 与本模型名次"
            note: Optional[str] = (
                "> 名次按该时效的 TS 排序（仅统计有有效值的模型）；"
                "不同数据的起报时段/时效范围要一致才可直接比较。"
            )
        else:
            heading = (
                f"## {_CN_NUM[next_number]}、与基准 {baseline_name or '基准'} 的对比（{lead:g}h）"
            )
            caption = f"{lead:g}h 时效上本模型与基准的 TS 及差值"
            note = None
        lines += _section_comparison(
            data, comparisons, artifacts, numbering, captions,
            lead=lead, heading=heading, caption=caption, note=note,
        )
        next_number += 1

    # 雷达图只在多模型（也就是必有对比对象）时出现——一个多边形没有比较对象
    if radar:
        lines += _section_capability(
            models, numbering,
            heading=f"## {_CN_NUM[next_number]}、模型能力雷达图",
            radar_path=radar_path,
        )
        next_number += 1

    lines.append(f"## {_CN_NUM[next_number]}、改进建议")
    lines.append("")
    for index, item in enumerate(recommendations(grades), start=1):
        lines.append(f"{index}. {item}")
    lines.append("")
    next_number += 1
    lines.append(f"## {_CN_NUM[next_number]}、口径与注意事项")
    lines.append("")
    lines.append("- **值缺失的含义**：TS 为 “—” 表示该等级没有有效配对或无事件，与“评分为 0”不同；")
    lines.append("- **BIAS 不是平均误差**：它是频率偏差，>1 表示预报事件偏多，<1 表示偏少；")
    if ensemble:
        # 没跑概率评分时不能写「要看第六节」——那一节根本没出现，会指到空处。
        lines.append(
            "- **前五节评的是集合平均场**：TS / POD / FAR / 漏报率 / BIAS 都不含成员离散信息，"
            + ("集合的相对优势要看第六节；" if probability else "本批未跑概率评分；")
        )
        if probability:
            lines.append(
                "- **概率评分与 TS 不可混比**：两者的累积窗口与阈值都不同，"
                "AROC 还是逐成员概率口径；"
            )
    if coverage and len(coverage) > 1:
        lines.append(
            "- **各模型的样本量并不相同**：第一节列的是各份实测配对数，"
            "差值来自起报日数而非口径；样本最少的那份名次波动最大，横向比高低前先看那一行；"
        )
    lines.append("- **跨时效不要直接平均后比较模型**：正式对比应使用共同有效样本与固定验证口径；")
    lines.append(
        "- 结果来自统一长表（`scores.csv` / `categorical_wide.csv`"
        # 概率表的文件名新老不一：v2 产物是 diagnostics/probability_wide.csv，
        # 老归档把两张裸 CSV 并排放，叫 aroc_bss*.csv。两个名字都写出来，
        # 免得读者拿着新名字在老归档目录里找不到第六节的数据来源。
        + (
            "，概率评分另见同目录的 `probability_wide.csv`（老归档叫 `aroc_bss*.csv`）"
            if probability
            else ""
        )
        + "），逐行可追溯到具体时效与阈值。"
    )
    lines.append("")

    # 附 L 放在「口径与注意事项」之后：它是可选块，编号走自己的 `图 L*` 命名空间，
    # 有它没它都不动正文任何一个号。
    lines += _section_regions(region_frames, region_labels, artifacts, region_figure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 TS 结果表画成图并写成 Markdown 报告",
        epilog="示例: python -m xmetai_evaluation.visualization.ts_report "
        "--csv ts_fgvp_2025.csv --result reports/fgvp --model FGVP",
    )
    parser.add_argument("--csv", type=Path, required=True, help="TS 结果表（csv）")
    parser.add_argument("--result", type=Path, required=True, help="输出目录")
    parser.add_argument("--model", default=None, help="模型名（默认取文件名）")
    parser.add_argument("--baseline-csv", type=Path, default=None, help="可选：基准模型结果表")
    parser.add_argument("--baseline-name", default=None, help="基准模型名")
    parser.add_argument("--lead-h", type=float, default=None, help="基准时效（默认取最小时效）")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    from xmetai_evaluation.visualization.precipitation_plots import PrecipitationPlotter

    plt_dpi = args.dpi
    import matplotlib.pyplot as plt

    plt.rcParams["savefig.dpi"] = plt_dpi
    model = args.model or args.csv.stem
    plotter = PrecipitationPlotter()
    artifacts = plotter.create_report(
        pd.read_csv(args.csv),
        output_dir=args.result,
        model_name=model,
        lead_h=args.lead_h,
        baseline_df=pd.read_csv(args.baseline_csv) if args.baseline_csv else None,
        baseline_name=args.baseline_name,
    )
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
