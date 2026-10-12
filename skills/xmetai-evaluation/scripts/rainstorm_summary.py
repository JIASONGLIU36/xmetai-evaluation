#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""43 个暴雨过程 × 2 口径 × N 模型 → 挑过程对比总表。

用法::

    python skills/xmetai-evaluation/scripts/rainstorm_summary.py \\
      --model FuXi=/workspace/szwCode/evaluation_results/rainstorm_ts_single_fuxi \\
      --model FGVP-ctrl=/workspace/szwCode/evaluation_results/rainstorm_ts_single_fgvp_ctrl \\
      --result /workspace/szwCode/evaluation_results/rainstorm_summary

``--model NAME=目录`` 可重复；目录是 rainstorm_ts_single_* 配置的输出根：
口径 A 的产物在 ``{编号}_24h/{起报日}/scores.csv``（每天一个结果），
口径 B 在 ``{编号}_total/scores.csv``（一个过程一个结果）。
不给 ``--model`` 时用下面的 DEFAULT_MODELS 字面量，不给 ``--result`` 用
DEFAULT_RESULT——换机器直接改这两个常量。

产物两件（都写进 ``--result``）：

    summary_long.csv   长表：一行 = (过程, 口径, [起报日], 阈值, 指标)，
                       各模型的 value / n_valid / status 并排 + TS 差值
    screening.md       宽表：口径 A 一行一个起报日、口径 B 一行一个过程，
                       筛选阈值 ≥10 / ≥25 / ≥50 逐模型并排 + Δ；
                       末尾各口径一份挑过程排序参考（≥25mm 全球 TS）；
                       缺数据的过程段标「缺」

**口径 A 的过程级数值一律是「汇总重算」**：把该过程各起报日的
hits / misses / false_alarms 加总后按定义再算一次 TS，**不是**各起报日 TS 的
算术平均——TS 是比率，简单平均会让只报出十几个站、TS 恰好为 0 的起报日与事件
上千的起报日等权。与报告的算法一致（见 ``visualization/rainstorm_report.py``
的 ``daily_pooled``）。逐日那一列仍是当天的原值，没有汇总。

口径提醒：逐日24h 是 day-1 的 08–08 窗（起报日 08 时 → 次日 08 时），
过程累积是 S-1 日一根预报从头累到尾。两种口径的 TS 因基准率不同
**互相不可比**，只在本口径内跨模型、跨过程比。TS 均取全球行
（zd_sta_10285 全部站，评测已不分带；老产物里若还有分带行，只取全球行）。

过程编目（编号 → 起止 / 等级）**不在本文件里**，从主仓库的
``xmetai_evaluation.rainstorm_catalog`` 取——那份才是真值，改纪要表改那里。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

#: scripts/ -> xmetai-evaluation/ -> skills/ -> 仓库根
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# 过程编目（编号 → 起止/等级）的真值在主仓库，这里不再自带一份——
# 历史上本文件与评测配置各有一份，改一头另一头不会跟着动。
from xmetai_evaluation.rainstorm_catalog import codes, expected_days, get

#: 模型名 → rainstorm_ts_single_* 的输出根（字面量默认，换机器直接改）。
DEFAULT_MODELS = [
    ("FuXi", "/workspace/szwCode/evaluation_results/rainstorm_ts_single_fuxi"),
    ("FGVP-ctrl", "/workspace/szwCode/evaluation_results/rainstorm_ts_single_fgvp_ctrl"),
]

#: 汇总输出目录（字面量默认）。
DEFAULT_RESULT = "/workspace/szwCode/evaluation_results/rainstorm_summary"

#: 挑过程宽表里展示的筛选阈值（mm）。
SCREENING_THRESHOLDS = [10.0, 25.0, 50.0]

#: 排序参考用的阈值（mm）。
RANK_THRESHOLD = 25.0

#: 指标在 scores.csv 里的 metric 名。
METRICS = ["ts", "pod", "far", "miss_rate", "frequency_bias"]

#: 全部过程编号（升序）。编目本身见 ``xmetai_evaluation.rainstorm_catalog``。
ALL_CODES = codes()


def parse_model(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--model 要写成 NAME=目录，收到的是 {value!r}")
    name, directory = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--model 的模型名不能为空: {value!r}")
    return name, Path(directory.strip())


def read_scores(path: Path):
    """读一份 scores.csv，只留全球行；缺文件/空表返回 None。"""
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    frame["region"] = frame["region"].fillna("全球")
    return frame[frame["region"] == "全球"]


def load_total(root: Path, code: str):
    """口径 B 的产物表（{编号}_total/scores.csv）；缺产物返回 None。"""
    return read_scores(root / f"{code}_total" / "scores.csv")


def load_days(root: Path, code: str):
    """口径 A 的逐起报日产物：{起报日: 表}；缺目录/缺表的起报日跳过。"""
    base = root / f"{code}_24h"
    days = {}
    if not base.is_dir():
        return days
    for day_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        frame = read_scores(day_dir / "scores.csv")
        if frame is not None:
            days[day_dir.name] = frame
    return days


def load_wide(path: Path):
    """读一份 ``{起报日}/diagnostics/categorical_wide.csv``（带四格表计数）。"""
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return None if frame.empty else frame


def _wide_counts(frame, threshold: float):
    """宽表里某一档的 hits / misses / false_alarms；缺档或缺计数返回 None。"""
    if frame is None:
        return None
    part = frame[frame["threshold_mm"] == float(threshold)]
    if part.empty:
        return None
    first = part.iloc[0]
    counts = {
        key: pd.to_numeric(first[key], errors="coerce")
        for key in ("hits", "misses", "false_alarms")
    }
    if any(pd.isna(value) for value in counts.values()):
        return None
    return {key: float(value) for key, value in counts.items()}


def pooled_ts(root: Path, code: str, threshold: float):
    """口径 A 的过程级 TS：各起报日的 hits / misses / false_alarms 加总后重算。

    与报告（``visualization/rainstorm_report.py``）用的是**同一个定义**，
    不是各起报日 TS 的算术平均——TS 是比率，简单平均会让只报出十几个站、
    TS 恰好为 0 的起报日与事件上千的起报日等权，把均值拖偏。
    ``scores.csv`` 里没有四格表计数，所以这里读
    ``{起报日}/diagnostics/categorical_wide.csv``；缺产物返回 None。
    """
    base = root / f"{code}_24h"
    if not base.is_dir():
        return None
    total = {"hits": 0.0, "misses": 0.0, "false_alarms": 0.0}
    seen = False
    for day_dir in sorted(item for item in base.iterdir() if item.is_dir()):
        counts = _wide_counts(load_wide(day_dir / "diagnostics" / "categorical_wide.csv"),
                              threshold)
        if counts is None:
            continue
        seen = True
        for key in total:
            total[key] += counts[key]
    denominator = sum(total.values())
    return total["hits"] / denominator if seen and denominator else None


def segment_view(frame, metric: str = "ts"):
    """(阈值 -> value) 的小视图。"""
    if frame is None:
        return None
    part = frame[frame["metric"] == metric]
    return dict(zip(part["threshold"], part["value"]))


def fmt_ts(view, threshold: float) -> str:
    if view is None:
        return "缺"
    value = view.get(threshold)
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.3f}"


def frame_keys(frames):
    """长表要出的 (阈值, 指标) 组合：各模型帧里出现过的并集。"""
    pairs = set()
    for frame in frames:
        if frame is not None:
            part = frame[frame["metric"].isin(METRICS)]
            pairs.update(zip(part["threshold"], part["metric"]))
    return sorted(pairs)


def fill_model_cells(row, names, frames, threshold, metric):
    """把各模型的 value / n_valid / status 填进长表行；TS 再加一列差值。"""
    values = []
    for name, frame in zip(names, frames):
        part = (
            frame[(frame["metric"] == metric) & (frame["threshold"] == threshold)]
            if frame is not None
            else None
        )
        if part is None or part.empty:
            row[f"{name}.value"], row[f"{name}.n_valid"] = None, None
            row[f"{name}.status"] = "缺"
            values.append(None)
        else:
            first = part.iloc[0]
            row[f"{name}.value"] = float(first["value"])
            row[f"{name}.n_valid"] = int(first["n_valid"])
            row[f"{name}.status"] = first["status"]
            values.append(row[f"{name}.value"])
    if metric == "ts" and len(values) >= 2 and all(v is not None for v in values):
        row["delta_ts"] = values[0] - values[1]


def build_long(models, codes):
    """长表：一行 = (过程, 口径, [起报日], 阈值, 指标)，模型列并排。"""
    names = [name for name, _ in models]
    rows = []
    for code in codes:
        process = get(code)

        # 口径 A：一行 = (过程, 起报日, 阈值, 指标)
        per_model = {name: load_days(root, code) for name, root in models}
        all_days = (
            sorted(set().union(*[set(d) for d in per_model.values()]))
            if any(per_model.values())
            else []
        )
        for day in all_days:
            frames = [per_model[name].get(day) for name in names]
            for threshold, metric in frame_keys(frames):
                row = {
                    "编号": code,
                    "起止": process.span,
                    "等级": process.grade,
                    "口径": "逐日24h",
                    "起报日": day,
                    "threshold": threshold,
                    "metric": metric,
                }
                fill_model_cells(row, names, frames, threshold, metric)
                rows.append(row)

        # 口径 B：一行 = (过程, 阈值, 指标)，起报日留空
        totals = [load_total(root, code) for _, root in models]
        if any(f is not None for f in totals):
            for threshold, metric in frame_keys(totals):
                row = {
                    "编号": code,
                    "起止": process.span,
                    "等级": process.grade,
                    "口径": "过程累积",
                    "起报日": "",
                    "threshold": threshold,
                    "metric": metric,
                }
                fill_model_cells(row, names, totals, threshold, metric)
                rows.append(row)
    return pd.DataFrame(rows)


def _screening_row(cells, names, views, threshold):
    """宽表一行里一个阈值的小节：各模型一格 + Δ。"""
    values = []
    for name in names:
        text = fmt_ts(views.get(name), threshold)
        cells.append(text)
        values.append(float(text) if text not in ("缺", "—") else None)
    cells.append(
        f"{values[0] - values[1]:+.3f}"
        if len(values) >= 2 and all(v is not None for v in values)
        else "—"
    )


def build_screening_markdown(models, codes):
    """宽表：口径 A 逐日表 + 口径 B 过程表 + 各口径一份排序参考。"""
    names = [name for name, _ in models]
    main_root = dict(models)[names[0]]
    lines = [
        "# 2025 年暴雨过程个例检验 · 挑过程总表",
        "",
        "TS 均为**全球行**（zd_sta_10285 全部站）。两种口径互不可比：",
        "「逐日24h」= 每个起报日的 day-1 窗（起报日 08 时 → 次日 08 时），**每天一个结果**；",
        "「过程累积」= 过程起点前一天的一根预报把整段降水累加后评总量。",
        "「缺」= 该段产物缺失（多半是那几天的预报数据缺）。",
        "",
    ]

    # ── 口径 A：一行一个起报日 ────────────────────────────────────
    lines += ["## 口径：逐日24h（每行一个起报日）", ""]
    header = ["编号", "起止", "等级", "起报日"]
    for threshold in SCREENING_THRESHOLDS:
        header += [f"≥{threshold:g} {name}" for name in names] + [f"Δ({names[0]}-{names[1]})"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for code in codes:
        process = get(code)
        per_model = {name: load_days(root, code) for name, root in models}
        all_days = (
            sorted(set().union(*[set(d) for d in per_model.values()]))
            if any(per_model.values())
            else []
        )
        for day in all_days:
            cells = [code, process.span, process.grade, day]
            for threshold in SCREENING_THRESHOLDS:
                views = {name: segment_view(per_model[name].get(day)) for name in names}
                _screening_row(cells, names, views, threshold)
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ── 口径 B：一行一个过程 ──────────────────────────────────────
    lines += ["## 口径：过程累积", ""]
    header = ["编号", "起止", "等级"]
    for threshold in SCREENING_THRESHOLDS:
        header += [f"≥{threshold:g} {name}" for name in names] + [f"Δ({names[0]}-{names[1]})"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for code in codes:
        process = get(code)
        views = {name: segment_view(load_total(root, code)) for name, root in models}
        cells = [code, process.span, process.grade]
        for threshold in SCREENING_THRESHOLDS:
            _screening_row(cells, names, views, threshold)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ── 排序参考（口径 A 按天平均，口径 B 直接累积 TS；都 ≥25mm）────
    lines += [
        f"## 排序参考（逐日24h 汇总，{names[0]} ≥{RANK_THRESHOLD:g}mm TS，前 10）",
        "",
        "「汇总」= 把该过程各起报日的 hits / misses / false_alarms 加总后重算，"
        "**不是**各起报日 TS 的算术平均（与报告的算法一致）。",
        "",
    ]
    entries = []
    for code in codes:
        value = pooled_ts(main_root, code, RANK_THRESHOLD)
        if value is not None:
            entries.append((value, code))
    entries.sort(reverse=True)
    for rank, (value, code) in enumerate(entries[:10], start=1):
        process = get(code)
        lines.append(f"{rank}. **{code}**（{process.span}，{process.grade}）汇总 TS = {value:.3f}")
    lines.append("")

    lines += [f"## 排序参考（过程累积，{names[0]} ≥{RANK_THRESHOLD:g}mm TS，前 10）", ""]
    entries = []
    for code in codes:
        view = segment_view(load_total(main_root, code))
        if view is not None:
            value = view.get(RANK_THRESHOLD)
            if value is not None and not pd.isna(value):
                entries.append((float(value), code))
    entries.sort(reverse=True)
    for rank, (value, code) in enumerate(entries[:10], start=1):
        process = get(code)
        lines.append(f"{rank}. **{code}**（{process.span}，{process.grade}）TS = {value:.3f}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="43 个暴雨过程 × 2 口径 × N 模型 → 挑过程对比总表",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        metavar="NAME=DIR",
        help="模型名=rainstorm_ts_single_* 输出根，可重复；缺省用脚本内 DEFAULT_MODELS",
    )
    parser.add_argument(
        "--result",
        default=DEFAULT_RESULT,
        help="汇总输出目录（缺省用脚本内 DEFAULT_RESULT）",
    )
    args = parser.parse_args(argv)

    models = args.model if args.model else [parse_model(f"{n}={d}") for n, d in DEFAULT_MODELS]
    if len(models) < 2:
        parser.error("至少两个 --model（对比表没有意义），或改脚本内 DEFAULT_MODELS")
    all_codes = ALL_CODES

    result_dir = Path(args.result)
    result_dir.mkdir(parents=True, exist_ok=True)

    long_table = build_long(models, all_codes)
    long_path = result_dir / "summary_long.csv"
    long_table.to_csv(long_path, index=False, encoding="utf-8-sig")

    markdown = build_screening_markdown(models, all_codes)
    md_path = result_dir / "screening.md"
    md_path.write_text(markdown, encoding="utf-8")

    missing = []
    for code in all_codes:
        got = set(load_days(models[0][1], code))
        missing += [f"{code}_24h_{day}" for day in expected_days(code) if day not in got]
        if load_total(models[0][1], code) is None:
            missing.append(f"{code}_total")
    print(f"长表 {len(long_table)} 行 -> {long_path}")
    print(f"宽表 -> {md_path}")
    if missing:
        print(f"注意：{models[0][0]} 缺 {len(missing)} 段产物：{', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
