#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""降水分类检验（TS 系列）的可视化。

输入：一张 TS 结果表，列至少包含 ``lead_h / grade / TS``，通常还有
``window_h / threshold_mm / hits / misses / false_alarms / n_pairs / POD / FAR / BIAS``
（``漏报率`` 缺失时由 ``1 - POD`` 推导；多出来的列如 ``run_id`` 会被忽略）。

输出：若干 PNG，以及一份 Markdown 报告（报告正文见 ``ts_report`` 模块）。

三条与旧实现的区别（旧实现在服务器上会出豆腐块/空图/被截断的柱子）：

* 中文字体**自动探测**，找不到中文字体时退化为英文标签并给出警告；
* 时效**不写死**：默认取数据里最小的时效，也可显式指定；
* 每个指标用自己的纵轴范围，``BIAS`` 不截断并画 ``=1`` 参考线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from xmetai_evaluation.visualization.ts_report import (
    box_stats,
    build_ts_report,
    collect_box_models,
    load_probability,
    ordered_grades,
)

try:  # seaborn 只用于热力图配色，缺了也能跑
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

REQUIRED_COLUMNS = ("lead_h", "grade", "TS")
OPTIONAL_COLUMNS = (
    "window_h",
    "threshold_mm",
    "hits",
    "misses",
    "false_alarms",
    "n_pairs",
    "POD",
    "FAR",
    "BIAS",
)

_CJK_FONTS = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti TC",
    "Arial Unicode MS",
)

THRESHOLD_COLORS = {
    "≥0.1": "#1f77b4",
    "≥4": "#17becf",
    "≥10": "#2ca02c",
    "≥13": "#bcbd22",
    "≥25": "#ff7f0e",
    "≥50": "#d62728",
    "≥100": "#9467bd",
    "≥250": "#8c564b",
}

_FALLBACK_COLORS = ("#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b")

#: 多模型图的配色，按"模型在字典里的次序"取。箱线图和差值图共用一份——
#: 两张图吃的都是 ``{主模型: data, **baselines}``，所以同一个模型在两处颜色一致。
MODEL_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b")


def setup_chinese_font() -> Optional[str]:
    """选择可用的中文字体；返回选中的字体名（没有则 None）。"""
    available = {font.name for font in fm.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def prepare_ts_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """校验并规范化 TS 结果表。

    Raises:
        ValueError: 缺少必需列，或数据为空。
    """
    data = df.copy()
    data.columns = [str(column).strip() for column in data.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            f"TS 结果表缺少必需列 {missing}；现有列: {list(data.columns)}"
        )
    if data.empty:
        raise ValueError("TS 结果表是空的")

    if "threshold_mm" not in data.columns:
        data["threshold_mm"] = data["grade"].astype(str).str.replace("≥", "", regex=False)
        data["threshold_mm"] = pd.to_numeric(data["threshold_mm"], errors="coerce")
    if "漏报率" not in data.columns:
        if "miss_rate" in data.columns:
            data["漏报率"] = pd.to_numeric(data["miss_rate"], errors="coerce")
        elif "POD" in data.columns:
            data["漏报率"] = 1.0 - pd.to_numeric(data["POD"], errors="coerce")

    for column in OPTIONAL_COLUMNS + ("漏报率",):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["lead_h"] = pd.to_numeric(data["lead_h"], errors="coerce")
    data = data.dropna(subset=["lead_h", "TS"])
    if data.empty:
        raise ValueError("TS 结果表里没有可用的 lead_h/TS 数值")

    data["grade"] = data["grade"].astype(str)
    if "threshold_mm" in data.columns and data["threshold_mm"].notna().any():
        order = data.groupby("grade")["threshold_mm"].first().sort_values().index
        data["grade"] = pd.Categorical(data["grade"], categories=list(order), ordered=True)
    return data.sort_values(["grade", "lead_h"])


def split_regions(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """宽表 → ``(全球行, {带名: 该带的行})``。

    分纬度带评测时同一张宽表里全球行与各带行并存（全球行的 ``region`` 空），
    **两者绝不能混着算**：各带平均再平均 ≠ 全球平均，混进去会把带间差异整个
    抹平。所以图与正文一律只看全球行，带行走 :func:`ts_report.build_ts_report`
    的「附 L」那一块。

    没有 ``region`` 列（这个能力出现之前的产物、或没配分带）时：全部算全球行、
    带集为空——**与加这个能力之前逐行一致**。函数本身是幂等的，传进来的已经是
    全球行时得到的还是它自己。
    """
    if "region" not in df.columns:
        return df, {}
    values = df["region"]
    empty = values.isna() | (values.astype(str).str.strip() == "")
    if bool(empty.all()):
        return df, {}
    bands = {
        name: df[~empty & (values.astype(str).str.strip() == name)]
        for name in sorted(set(values[~empty].astype(str).str.strip()))
    }
    return df[empty], bands


def available_leads(df: pd.DataFrame) -> List[float]:
    """数据里出现过的时效（升序）。"""
    return sorted(float(value) for value in pd.unique(df["lead_h"]))


def _grade_colors(grades: Sequence[str]) -> Dict[str, str]:
    colors: Dict[str, str] = {}
    for index, grade in enumerate(grades):
        colors[str(grade)] = THRESHOLD_COLORS.get(
            str(grade), _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]
        )
    return colors


class PrecipitationPlotter:
    """降水分类检验结果的绘图器。"""

    def __init__(self, style: Optional[str] = None):
        """
        Args:
            style: 可选 matplotlib 样式名；默认不套样式，
                避免样式把中文字体设置覆盖掉（旧实现就栽在这）。
        """
        self.font = setup_chinese_font()
        if style:
            plt.style.use(style)
            setup_chinese_font()
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 150
        plt.rcParams["savefig.bbox"] = "tight"

    def _save(self, fig, save_path: Optional[Path]) -> None:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"已保存: {save_path}")

    def plot_metric_vs_lead(
        self,
        df: pd.DataFrame,
        metric: str = "TS",
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """指标随时效变化（按阈值分线）。"""
        if metric not in df.columns:
            raise ValueError(f"结果表里没有指标列 '{metric}'")

        data = df.dropna(subset=[metric])
        colors = _grade_colors(list(data["grade"].cat.categories) if hasattr(data["grade"], "cat") else sorted(data["grade"].unique()))
        # 事件数过少的等级用虚线并标注，避免"看着像结论"（样本少时方差极大）
        events = {}
        if {"hits", "misses"}.issubset(data.columns):
            totals = data.fillna({"hits": 0, "misses": 0}).groupby("grade", observed=True)[
                ["hits", "misses"]
            ].sum()
            events = {str(index): float(row["hits"] + row["misses"]) for index, row in totals.iterrows()}

        fig, ax = plt.subplots(figsize=figsize)
        for grade, subset in data.groupby("grade", observed=True):
            subset = subset.sort_values("lead_h")
            sparse = events.get(str(grade), float("inf")) < 50
            ax.plot(
                subset["lead_h"],
                subset[metric],
                marker="o",
                markersize=5,
                linewidth=2,
                linestyle="--" if sparse else "-",
                label=f"{grade}（样本少，仅参考）" if sparse else str(grade),
                color=colors.get(str(grade), "gray"),
            )

        ax.set_xlabel("预报时效 (小时)")
        ax.set_ylabel(metric)
        ax.set_title(title or f"{metric} 随预报时效变化")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(left=0)
        if metric in ("TS", "POD", "FAR", "漏报率"):
            ax.set_ylim(0, 1)
        if metric == "BIAS":
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
            ax.text(
                0.01, 1.0, " BIAS=1 (无偏差)",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=9, alpha=0.7,
            )
        ax.legend(title="降水等级", loc="best", frameon=True)
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_metric_by_threshold(
        self,
        df: pd.DataFrame,
        lead_h: Optional[float] = None,
        metrics: Sequence[str] = ("TS", "POD", "FAR"),
        figsize: Tuple[int, int] = (11, 6),
        save_path: Optional[Path] = None,
    ):
        """指定时效下、各阈值的指标柱状对比（每个指标一个子图）。"""
        leads = available_leads(df)
        lead = float(lead_h) if lead_h is not None else leads[0]
        subset = df[df["lead_h"] == lead].sort_values("threshold_mm")
        if subset.empty:
            raise ValueError(f"结果表里没有 {lead}h 时效的数据（可用: {leads}）")

        metrics = [metric for metric in metrics if metric in subset.columns]
        if not metrics:
            raise ValueError("没有可绘制的指标列")

        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        axes = np.atleast_1d(axes)
        x = np.arange(len(subset))
        for ax, metric in zip(axes, metrics):
            ax.bar(x, subset[metric], alpha=0.85, color="#4c72b0")
            ax.set_title(f"{metric} @ {lead:g}h")
            ax.set_xticks(x)
            ax.set_xticklabels([str(value) for value in subset["grade"]])
            ax.set_xlabel("降水等级")
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")
            if metric in ("TS", "POD", "FAR", "漏报率"):
                ax.set_ylim(0, 1)
            elif metric == "BIAS":
                ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
            for index, value in enumerate(subset[metric]):
                if pd.notna(value):
                    ax.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        fig.suptitle(f"预报时效 {lead:g}h 各降水等级的指标对比")
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        metric: str = "TS",
        figsize: Tuple[int, int] = (12, 7),
        cmap: str = "YlGnBu",
        save_path: Optional[Path] = None,
    ):
        """阈值 × 时效 热力图。"""
        pivot = df.pivot_table(index="grade", columns="lead_h", values=metric, observed=True)
        fig, ax = plt.subplots(figsize=figsize)
        upper = 1.0 if metric != "BIAS" else max(2.0, float(np.nanmax(pivot.values)))
        if sns is not None:
            sns.heatmap(
                pivot, annot=True, fmt=".3f", cmap=cmap, linewidths=0.5,
                linecolor="gray", ax=ax, vmin=0, vmax=upper,
                cbar_kws={"label": metric},
            )
            # 数字颜色随底色深浅自动切换（浅底黑字、深底白字）
            values = np.asarray(pivot.values, dtype="f8")
            for text in ax.texts:
                x, y = text.get_position()
                row, column = int(round(float(y) - 0.5)), int(round(float(x) - 0.5))
                if 0 <= row < values.shape[0] and 0 <= column < values.shape[1]:
                    value = values[row, column]
                    if np.isfinite(value):
                        text.set_color("white" if value / upper > 0.6 else "black")
        else:  # pragma: no cover
            mesh = ax.pcolormesh(pivot.values, cmap=cmap, vmin=0, vmax=upper)
            fig.colorbar(mesh, ax=ax, label=metric)
            ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
            ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
            ax.set_yticklabels([str(value) for value in pivot.index])
        ax.set_xlabel("预报时效 (小时)")
        ax.set_ylabel("降水等级")
        ax.set_title(f"{metric} 热力图（时效 × 降水等级）")
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_performance_diagram(
        self,
        df: pd.DataFrame,
        lead_h: Optional[float] = None,
        figsize: Tuple[int, int] = (9, 9),
        save_path: Optional[Path] = None,
    ):
        """性能图：横轴 1-FAR，纵轴 POD，灰色为 TS 等值线。"""
        leads = available_leads(df)
        lead = float(lead_h) if lead_h is not None else leads[0]
        subset = df[(df["lead_h"] == lead) & df["POD"].notna() & df["FAR"].notna()]
        if subset.empty:
            raise ValueError(f"结果表里没有 {lead}h 的 POD/FAR 数据")

        fig, ax = plt.subplots(figsize=figsize)
        grid = np.linspace(0.001, 1, 200)
        sr, pod = np.meshgrid(grid, grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)
        contour = ax.contour(sr, pod, ts, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                             colors="gray", linewidths=1, linestyles="--", alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8, fmt="TS=%.1f")
        for bias in (0.5, 1.0, 1.5, 2.0):
            ax.plot(grid, grid * bias, color="steelblue", linestyle=":" , linewidth=1, alpha=0.5)
            ax.text(grid[-1], min(grid[-1] * bias, 1.0), f" BIAS={bias:g}", fontsize=8, color="steelblue")

        colors = _grade_colors([str(value) for value in subset["grade"].unique()])
        for _, row in subset.iterrows():
            ax.scatter(1 - row["FAR"], row["POD"], s=130, zorder=10,
                       color=colors.get(str(row["grade"]), "gray"),
                       edgecolors="black", linewidths=1.2, label=str(row["grade"]))
        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        ax.legend(seen.values(), seen.keys(), title="降水等级", loc="lower left", frameon=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("成功比 1-FAR")
        ax.set_ylabel("命中率 POD")
        ax.set_title(f"性能图（预报时效 {lead:g}h）")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_metric_boxplot(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        metric: str = "TS",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None,
    ):
        """各模型、各降水等级的**跨时效分布**（箱线图）。

        一个箱体 = 同一模型、同一量级在**全部时效**上的取值组成的样本：
        箱体是四分位距（IQR），箱中横线是中位数，须线延伸到 1.5×IQR 内最远的点，
        空心圆是离群时效。横坐标是降水等级，每个量级下并排各模型一个箱体。
        """
        prepared = {name: prepare_ts_dataframe(frame) for name, frame in model_dfs.items()}
        names = list(prepared)
        grades = ordered_grades(list(prepared.values()))
        if not names or not grades:
            raise ValueError("箱线图没有可画的数据（模型列表或降水等级为空）")

        width = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=figsize)

        for grade_index, grade in enumerate(grades):
            for model_index, name in enumerate(names):
                frame = prepared[name]
                subset = frame[frame["grade"].astype(str) == grade]
                stats = box_stats(subset, metric=metric) if len(subset) else {"n": 0}
                position = grade_index + (model_index - len(names) / 2 + 0.5) * width
                color = MODEL_COLORS[model_index % len(MODEL_COLORS)]
                if stats["n"] == 0:
                    # 该模型没有这个量级（或全无有效值）。**不能把空数组喂给 boxplot**：
                    # 它随 matplotlib 版本时好时坏，还会吐 RuntimeWarning。画个字说明空缺。
                    ax.text(
                        position, 0.02, "无有效值", rotation=90,
                        ha="center", va="bottom", fontsize=8, color="gray",
                    )
                    continue
                # 一次调用只画一个箱体：这样每个箱子能单独上色，缺值也能按格跳过。
                ax.boxplot(
                    subset[metric].dropna().to_numpy(dtype=float),
                    positions=[position],
                    widths=width * 0.85,
                    patch_artist=True,
                    showfliers=True,
                    whis=1.5,
                    manage_ticks=False,  # 否则每次调用都重设刻度定位器，最后一次说了算
                    boxprops=dict(facecolor=color, edgecolor="#333333", linewidth=1.0, alpha=0.85),
                    medianprops=dict(color="black", linewidth=1.6),
                    whiskerprops=dict(color="#333333", linewidth=1.2),
                    capprops=dict(color="#333333", linewidth=1.2),
                    flierprops=dict(
                        marker="o", markerfacecolor="none", markeredgecolor=color,
                        markersize=5, markeredgewidth=1.0, linestyle="none",
                    ),
                )

        ax.set_xticks(range(len(grades)))
        ax.set_xticklabels(grades)
        ax.set_xlim(-0.5, len(grades) - 0.5)
        ax.set_xlabel("降水等级")
        ax.set_ylabel(metric)
        ax.set_title(f"各模型 {metric} 跨时效分布")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        if metric in ("TS", "POD", "FAR", "漏报率"):
            ax.set_ylim(0, 1)  # 比率型指标锁 0–1，见 references/precipitation-evaluation.md §6.1
        elif metric == "BIAS":
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
        if len(names) > 1:
            # 手工拼图例：直接 ax.legend() 会给每次 boxplot 调用各留一个条目
            ax.legend(
                handles=[
                    Patch(facecolor=MODEL_COLORS[i % len(MODEL_COLORS)],
                          edgecolor="#333333", label=name)
                    for i, name in enumerate(names)
                ],
                title="模型", loc="best", frameon=True,
            )
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def create_report(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        model_name: str = "model",
        lead_h: Optional[float] = None,
        baseline_df: Optional[pd.DataFrame] = None,
        baseline_name: Optional[str] = None,
        baselines: Optional[Dict[str, pd.DataFrame]] = None,
        sources: Optional[Dict[str, str]] = None,
        change_description: Optional[str] = None,
        region_labels: Optional[Dict[str, str]] = None,
        probability: Optional[Dict[str, pd.DataFrame]] = None,
        members: Optional[int] = None,
        coverage: Optional[Dict[str, int]] = None,
        ensemble: bool = False,
    ) -> Dict[str, Path]:
        """出图 + 写 Markdown 报告，返回 {名称: 路径}。

        ``probability`` / ``members`` / ``coverage`` / ``ensemble`` 是集合产物
        专属的，原样透传给 :func:`ts_report.build_ts_report`；确定性产物一个都
        不用给（全为默认值时报告的每一个字节与以前一致）。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = prepare_ts_dataframe(df)
        # 分纬度带：图和正文一律只看全球行，带行单独进报告的「附 L」（见
        # ``split_regions``）。对比表同样只留全球行——各带平均再平均不等于全球
        # 平均，把带行当全球行拿去比是另一种口径，不是同一件事。
        data, region_frames = split_regions(data)
        # 对比表统一在这里规范化一次：差值图、箱线图、报告表格读的是同一份，
        # spec 有问题的对比表也会在这里就报出清晰的缺列错误。
        baselines = {
            name: split_regions(prepare_ts_dataframe(frame))[0]
            for name, frame in (baselines or {}).items()
        }
        lead = float(lead_h) if lead_h is not None else available_leads(data)[0]
        window = None
        if "window_h" in data.columns and data["window_h"].notna().any():
            window = float(data["window_h"].dropna().iloc[0])
        prefix = f"{window:g}h_" if window else ""

        print(f"\n生成 {model_name} 评测报告（基准时效 {lead:g}h，输出 {output_dir}）")
        artifacts: Dict[str, Path] = {}

        for metric in ("TS", "POD", "FAR", "BIAS"):
            if metric not in data.columns:
                continue
            path = output_dir / f"{prefix}{metric}_vs_lead.png"
            self.plot_metric_vs_lead(
                data, metric=metric,
                title=f"{model_name} {metric} 随预报时效变化",
                save_path=path,
            )
            plt.close()
            artifacts[f"{metric}_vs_lead"] = path

        threshold_path = output_dir / f"{prefix}metrics_by_threshold_{lead:g}h.png"
        self.plot_metric_by_threshold(data, lead_h=lead, save_path=threshold_path)
        plt.close()
        artifacts["metrics_by_threshold"] = threshold_path

        heatmap_path = output_dir / f"{prefix}TS_heatmap.png"
        self.plot_heatmap(data, metric="TS", save_path=heatmap_path)
        plt.close()
        artifacts["TS_heatmap"] = heatmap_path

        # 箱线图的模型集合与报告第五节表格**必须是同一个 dict**，
        # 否则"表里的中位数和图上的线对不上"会从后门溜回来。
        box_models = collect_box_models(
            model_name,
            data,
            baseline_df=baseline_df,
            baseline_name=baseline_name,
            baselines=baselines,
        )
        box_path = output_dir / f"{prefix}TS_boxplot.png"
        self.plot_metric_boxplot(box_models, metric="TS", save_path=box_path)
        plt.close()
        artifacts["TS_boxplot"] = box_path

        if {"POD", "FAR"}.issubset(data.columns) and data[["POD", "FAR"]].notna().any().all():
            perf_path = output_dir / f"{prefix}performance_diagram_{lead:g}h.png"
            self.plot_performance_diagram(data, lead_h=lead, save_path=perf_path)
            plt.close()
            artifacts["performance_diagram"] = perf_path

        # 概率评分那张图：窗口取**概率表自己的** window_h（本仓库是 6h），不能复用
        # 上面的 ``prefix``——那个是 TS 的 24h 窗口，文件名会和骨架里写的对不上。
        probability = {
            str(name): loaded
            for name, frame in (probability or {}).items()
            if (loaded := load_probability(frame)) is not None
        }
        if probability:
            prob_window = None
            for frame in probability.values():
                if "window_h" not in frame.columns:
                    continue
                values = pd.to_numeric(frame["window_h"], errors="coerce").dropna()
                if len(values):
                    prob_window = float(values.iloc[0])
                    break
            prob_prefix = f"{prob_window:g}h_" if prob_window else ""
            prob_path = output_dir / f"{prob_prefix}AROC_BSS_vs_lead.png"
            self.plot_aroc_bss_vs_lead(
                probability, window=prob_window, save_path=prob_path
            )
            plt.close()
            artifacts["AROC_BSS_vs_lead"] = prob_path

        if baselines:
            decay_path = output_dir / f"{prefix}multi_model_TS_delta_vs_{model_name}.png"
            self.plot_multi_model_vs_lead(
                box_models,
                metric="TS",
                mode="delta",
                reference=model_name,
                save_path=decay_path,
            )
            plt.close()
            artifacts["multi_model_TS_delta"] = decay_path

        report_path = output_dir / "REPORT.md"
        build_ts_report(
            data,
            report_path,
            model_name=model_name,
            baseline_df=baseline_df,
            baseline_name=baseline_name,
            baselines=baselines,
            sources=sources,
            change_description=change_description,
            artifacts=artifacts,
            lead_h=lead,
            box_models=box_models,
            region_frames=region_frames,
            region_labels=region_labels,
            probability=probability,
            members=members,
            coverage=coverage,
            ensemble=ensemble,
        )
        artifacts["report"] = report_path
        # 别在这两行用 ✓ / ⚠：Windows 控制台默认 GBK，编码不了这两个字符。报告和图
        # 这时**已经写完了**，却会在最后一行抛 UnicodeEncodeError、进程非 0 退出。
        print(f"报告完成：{report_path}（{len(artifacts) - 1} 张图 + 1 份报告）")
        if self.font is None:
            print("未找到中文字体，图中的中文可能显示为方块；请安装 Noto Sans CJK 或 SimHei")
        return artifacts

    def plot_multi_model_vs_lead(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        metric: str = "TS",
        mode: str = "delta",
        reference: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 7.5),
        save_path: Optional[Path] = None,
        legend_title: str = "模型",
        suptitle: Optional[str] = None,
    ):
        """各模型指标随时效变化的对比（每个降水等级一个子图）。

        Args:
            mode: ``"delta"``（默认）画相对主模型的**差值**——几个模型水平接近时
                绝对曲线会重合成一团，差值才看得出谁强谁弱；``"absolute"`` 画绝对值。
            reference: 主模型名（差值基准）；默认取 ``model_dfs`` 的第一个。
            legend_title / suptitle: 图例标题与总标题。分纬度带那张图借这个函数画
                （每个带一条线，与"每个模型一条线"结构完全一样），不改标题就会
                写成「各模型」却列着带名。
        """
        if mode not in ("delta", "absolute"):
            raise ValueError(f"未知 mode: {mode}（可选 delta/absolute）")
        prepared = {name: prepare_ts_dataframe(frame) for name, frame in model_dfs.items()}
        names = list(prepared)
        reference = reference or (names[0] if names else None)
        if mode == "delta" and reference not in prepared:
            raise ValueError(f"差值基准 '{reference}' 不在模型列表 {names} 里")
        grades = ordered_grades(list(prepared.values()))

        columns = min(len(grades), 3)
        rows = int(np.ceil(len(grades) / columns)) if grades else 1
        fig, axes = plt.subplots(rows, columns, figsize=figsize, sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        palette = MODEL_COLORS
        styles = ("-", "--", "-.", ":")

        def _curve(frame: pd.DataFrame, grade: str) -> pd.DataFrame:
            subset = frame[frame["grade"].astype(str) == grade].sort_values("lead_h")
            return subset if metric in subset.columns else subset.iloc[0:0]

        for index, grade in enumerate(grades):
            ax = axes[index]
            base = _curve(prepared[reference], grade) if reference else None
            if mode == "absolute":
                for model_index, (name, frame) in enumerate(prepared.items()):
                    subset = _curve(frame, grade)
                    if subset.empty:
                        continue
                    ax.plot(
                        subset["lead_h"], subset[metric],
                        marker="o", markersize=4, markevery=2, linewidth=1.8,
                        linestyle=styles[model_index % len(styles)],
                        color=palette[model_index % len(palette)], label=name,
                    )
            else:
                ax.axhline(0.0, color="k", linewidth=1, alpha=0.5)
                for model_index, (name, frame) in enumerate(prepared.items()):
                    if name == reference:
                        continue
                    subset = _curve(frame, grade)
                    if subset.empty or base is None or base.empty:
                        continue
                    merged = subset.merge(
                        base[["lead_h", metric]], on="lead_h", suffixes=("", "_base")
                    )
                    if merged.empty:
                        continue
                    ax.plot(
                        merged["lead_h"], merged[metric] - merged[f"{metric}_base"],
                        marker="o", markersize=4, markevery=2, linewidth=1.8,
                        linestyle=styles[model_index % len(styles)],
                        color=palette[model_index % len(palette)],
                        label=f"{name} − {reference}",
                    )
            ax.set_title(f"{grade} mm")
            ax.grid(True, alpha=0.3, linestyle="--")
            if mode == "absolute" and metric in ("TS", "POD", "FAR", "漏报率"):
                ax.set_ylim(0, 1)
        for ax in axes[len(grades):]:
            ax.axis("off")
        axes[0].set_ylabel(metric if mode == "absolute" else f"Δ{metric}（相对 {reference}）")
        for ax in axes[:len(grades)]:
            ax.set_xlabel("预报时效 (小时)")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3),
                       frameon=True, title=legend_title)
        fig.suptitle(
            suptitle
            or (
                f"各模型 {metric} 与 {reference} 的差值随时效变化（>0 表示优于 {reference}）"
                if mode == "delta"
                else f"各模型 {metric} 随时效变化"
            )
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        self._save(fig, save_path)
        return fig

    def plot_aroc_bss_vs_lead(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        *,
        window: Optional[float] = None,
        figsize: Tuple[int, int] = (16, 7.5),
        save_path: Optional[Path] = None,
        legend_title: str = "模型",
    ):
        """概率评分（AROC / BSS）随时效变化——上排 AROC、下排 BSS，每列一个阈值。

        吃的是概率宽表（``diagnostics/probability_wide.csv``），**不是** TS 宽表：
        列的集合完全不同，所以走 ``prepare_ts_dataframe`` 会在第一步就报缺列。
        这里只按 ``region`` 空值筛全球行——与 ``split_regions`` 对"全球行"的定义
        一致，免得带行混进来把曲线拉平。

        两排**不共享纵轴**：AROC 以 0.5 为无技巧、BSS 以 0 为无正技巧，量纲差得
        远（BSS 能到 −1），共享会把 BSS 的零线压到 AROC 中段去。AROC 锁 0–1
        （它的天然上界就是 1），BSS 一律自适应——它是"相对气候基准的技巧"，
        负多少都有可能，截断就是在藏坏消息。

        Args:
            model_dfs: ``{模型名: 概率宽表}``，主模型在前。颜色按这个次序取
                ``MODEL_COLORS``，与箱线图、差值图是同一份配色。
            window: 概率累积窗口（小时），只进总标题。省略则不写窗口。
        """
        prepared: Dict[str, pd.DataFrame] = {}
        for name, frame in model_dfs.items():
            subset = frame
            if "region" in frame.columns:
                region = frame["region"]
                subset = frame[region.isna() | (region.astype(str).str.strip() == "")]
            prepared[name] = subset

        grades = ordered_grades(list(prepared.values()))
        if not grades:
            return None
        fig, axes = plt.subplots(
            2, len(grades), figsize=figsize, sharex=True, squeeze=False
        )
        palette = MODEL_COLORS
        styles = ("-", "--", "-.", ":")

        for column, grade in enumerate(grades):
            for row, (metric, reference) in enumerate((("AROC", 0.5), ("BSS", 0.0))):
                ax = axes[row][column]
                ax.axhline(reference, color="k", linestyle="--", linewidth=1, alpha=0.5)
                for model_index, (name, frame) in enumerate(prepared.items()):
                    if metric not in frame.columns:
                        continue
                    subset = frame[frame["grade"].astype(str) == grade].sort_values("lead_h")
                    values = pd.to_numeric(subset[metric], errors="coerce")
                    if subset.empty or not values.notna().any():
                        continue
                    ax.plot(
                        subset["lead_h"], values,
                        marker="o", markersize=4, markevery=2, linewidth=1.8,
                        linestyle=styles[model_index % len(styles)],
                        color=palette[model_index % len(palette)], label=name,
                    )
                ax.set_title(f"{grade} mm")
                ax.set_xlabel("预报时效 (小时)")
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3, linestyle="--")
                if metric == "AROC":
                    ax.set_ylim(0, 1)

        handles, labels = axes[0][0].get_legend_handles_labels()
        if not handles:  # 上排全空时图例得从别处捞，否则整张图没有模型名
            for ax in axes.ravel():
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    break
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
                       frameon=True, title=legend_title)
        window_text = f"{window:g}h 累积窗口" if window else "概率口径"
        fig.suptitle(
            f"概率评分的 AROC（上排）与 BSS（下排）随时效变化（{window_text}；"
            f"虚线为 AROC=0.5 / BSS=0 的无技巧线）"
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        self._save(fig, save_path)
        return fig
