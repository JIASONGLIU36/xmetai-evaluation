# -*- coding: utf-8 -*-
"""2025 年暴雨过程个例检验的出图器（多模型 × 多过程 × 双口径）。

只画图，不写报告——正文与统计量在 :mod:`xmetai_evaluation.visualization.rainstorm_report`。
图与正文共用同一批统计量（由 ``rainstorm_report`` 算好传进来），避免"表、图、文字
各说各话"。

四类图，都是**多模型并排**：

- :meth:`RainstormPlotter.plot_process_overview`：43 个过程的分组条形，
  按主模型降序——这是「挑过程」那一步看的图。**两种口径各画一张**
  （``metric_label`` 只改文字，数值由调用方按口径算好传进来）；
- :meth:`RainstormPlotter.plot_daily_ts`：单个过程的逐日 24h TS 折线，
  按降水等级分面，每个模型一条线；
- :meth:`RainstormPlotter.plot_threshold_bars`：单个过程的累积 TS 按阈值分组的条形；
- :meth:`RainstormPlotter.plot_caliber_scatter`：逐过程散点，横轴口径 A 汇总、
  纵轴口径 B 累积，带 ``y = x`` 参考线——回答"换口径名次会不会翻转"。

**不出降水空间分布图**：评测产物里只有站点列联表的聚合统计
（``diagnostics/categorical_wide.csv``：hits / misses / false_alarms / n_pairs / TS…），
没有逐站或逐格的降水量，画不出实况场、也画不出预报−实况差值场。要画得回头读
原始预报 nc 与站点报文，那是另一个脚本的事。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from xmetai_evaluation.visualization.precipitation_plots import (
    MODEL_COLORS,
    setup_chinese_font,
)

#: 折线/条形里"没有有效取值"的图元直接跳过，而不是画成 0——
#: TS 的 NaN 是无有效配对或无事件，与"技巧为 0"是两回事（SKILL.md 硬性规则 4）。
_MISSING = "—"


class RainstormPlotter:
    """暴雨过程个例检验的绘图器。

    Args:
        style: 可选 matplotlib 样式名；默认不套样式（样式会覆盖中文字体设置）。
    """

    def __init__(self, style: Optional[str] = None):
        import matplotlib.pyplot as plt

        self._plt = plt
        self.font = setup_chinese_font()
        if style:
            plt.style.use(style)
            setup_chinese_font()
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 150
        plt.rcParams["savefig.bbox"] = "tight"

    # ------------------------------------------------------------------ 私有

    def _save(self, fig, save_path: Optional[Path]) -> None:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"已保存: {save_path}")
        self._plt.close(fig)

    @staticmethod
    def _colors(names: Sequence[str]) -> Dict[str, str]:
        """模型配色按"名字在列表里的次序"取，与箱线图/差值图同一套"""
        return {
            str(name): MODEL_COLORS[index % len(MODEL_COLORS)]
            for index, name in enumerate(names)
        }

    # ------------------------------------------------------------------ 图

    def plot_process_overview(
        self,
        codes: Sequence[str],
        values: Dict[str, Dict[str, float]],
        *,
        threshold: float,
        save_path: Optional[Path] = None,
        order: Optional[Sequence[str]] = None,
        metric_label: str = "过程累积",
    ) -> None:
        """43 个过程的分组条形，按 ``order``（默认第一个模型）降序。

        Args:
            codes: 过程编号，决定横轴顺序。
            values: ``{模型名: {编号: TS}}``；缺的过程画成空位（不补 0）。
            threshold: 图题里的阈值，只影响文字。
            order: 排序依据的模型名；缺省取 ``values`` 的第一个模型。
            metric_label: 纵轴与图题里的口径名，如 ``过程累积`` / ``逐日 24h 汇总``；
                **只影响文字**，数值由调用方决定用哪个口径。
        """
        plt = self._plt
        names = list(values)
        ranked_by = order or (names[0] if names else None)
        ordered = sorted(
            codes,
            key=lambda code: (
                values.get(ranked_by, {}).get(code) is None,
                -(values.get(ranked_by, {}).get(code) or 0.0),
            ),
        )
        colors = self._colors(names)
        positions = np.arange(len(ordered), dtype=float)
        width = 0.8 / max(len(names), 1)

        fig, ax = plt.subplots(figsize=(max(10.0, 0.34 * len(ordered)), 5.4))
        for index, name in enumerate(names):
            series = [values.get(name, {}).get(code) for code in ordered]
            xs = [x for x, value in zip(positions, series) if value is not None]
            ys = [value for value in series if value is not None]
            ax.bar(
                np.array(xs) + (index - (len(names) - 1) / 2) * width,
                ys,
                width=width,
                label=name,
                color=colors[name],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(ordered, rotation=90, fontsize=7)
        ax.set_ylabel(f"{metric_label} TS（≥{threshold:g}mm）")
        ax.set_title(
            f"各过程{metric_label} TS（≥{threshold:g}mm，按 {ranked_by} 降序）",
            fontsize=11,
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, ncols=min(len(names), 4))
        fig.tight_layout()
        self._save(fig, save_path)

    def plot_daily_ts(
        self,
        days: Sequence[str],
        series: Dict[str, Dict[str, Dict[float, Optional[float]]]],
        thresholds: Sequence[float],
        *,
        code: str,
        span: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """单个过程的逐日 24h TS 折线，一个阈值一个子图。

        Args:
            days: 起报日，按顺序决定横轴。
            series: ``{模型名: {起报日: {阈值: TS}}}``。
            thresholds: 要分面的阈值（升序）。
        """
        plt = self._plt
        names = list(series)
        colors = self._colors(names)
        short = [_short_day(day) for day in days]

        fig, axes = plt.subplots(
            1, max(len(thresholds), 1), figsize=(4.2 * max(len(thresholds), 1), 3.8),
            squeeze=False, sharey=True,
        )
        for column, threshold in enumerate(thresholds):
            ax = axes[0][column]
            for name in names:
                values = [
                    series.get(name, {}).get(day, {}).get(threshold) for day in days
                ]
                xs = [x for x, value in zip(short, values) if value is not None]
                ys = [value for value in values if value is not None]
                if xs:
                    ax.plot(xs, ys, marker="o", markersize=4, label=name, color=colors[name])
            ax.set_title(f"≥{threshold:g}mm", fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="x", labelsize=8)
        axes[0][0].set_ylabel("TS")
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"{code}（{span}）逐日 24h TS（08–08 北京时，day-1）", fontsize=11)
        fig.tight_layout()
        self._save(fig, save_path)

    def plot_threshold_bars(
        self,
        thresholds: Sequence[float],
        values: Dict[str, Dict[float, Optional[float]]],
        *,
        code: str,
        span: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """单个过程的累积 TS 按阈值分组的条形，每个模型一组。"""
        plt = self._plt
        names = list(values)
        colors = self._colors(names)
        labels = [f"≥{threshold:g}" for threshold in thresholds]
        positions = np.arange(len(thresholds), dtype=float)
        width = 0.8 / max(len(names), 1)

        fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(thresholds)), 4.6))
        for index, name in enumerate(names):
            series = [values.get(name, {}).get(threshold) for threshold in thresholds]
            xs = [x for x, value in zip(positions, series) if value is not None]
            ys = [value for value in series if value is not None]
            ax.bar(
                np.array(xs) + (index - (len(names) - 1) / 2) * width,
                ys,
                width=width,
                label=name,
                color=colors[name],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("降水等级（mm）")
        ax.set_ylabel("过程累积 TS")
        ax.set_title(f"{code}（{span}）过程累积 TS 随降水等级的变化", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, ncols=min(len(names), 4))
        fig.tight_layout()
        self._save(fig, save_path)

    def plot_caliber_scatter(
        self,
        points: Dict[str, Sequence[tuple]],
        *,
        threshold: float,
        save_path: Optional[Path] = None,
    ) -> None:
        """逐过程散点：横轴口径 A 起报日汇总、纵轴口径 B 累积，带 ``y = x`` 参考线。

        落在这条线**上方**的过程，说明它的累积 TS 高于按逐日平均外推的水平；
        偏离越大，两个口径给出的"这个过程好不好预报"差得越远。点都是有效
        配对（两个口径都有数）——缺一样的那个过程整点不画。

        Args:
            points: ``{模型名: [(口径A 汇总 TS, 口径B 累积 TS), …]}``。
        """
        plt = self._plt
        names = list(points)
        colors = self._colors(names)

        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        for name in names:
            pairs = [(x, y) for x, y in points.get(name, []) if x is not None and y is not None]
            if not pairs:
                continue
            xs = [pair[0] for pair in pairs]
            ys = [pair[1] for pair in pairs]
            ax.scatter(xs, ys, s=26, label=f"{name}（{len(pairs)} 个过程）",
                       color=colors[name], alpha=0.8, edgecolors="none")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.9, color="#888888")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"口径 A：逐日 24h 汇总 TS（≥{threshold:g}mm）")
        ax.set_ylabel(f"口径 B：过程累积 TS（≥{threshold:g}mm）")
        ax.set_title("两种口径的逐过程对照（虚线为 y = x）", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, save_path)


def _short_day(day: str) -> str:
    """``20250311`` → ``0311``，折线横轴用。不是八位数字就原样返回。"""
    text = str(day)
    return text[4:] if len(text) == 8 and text.isdigit() else text
