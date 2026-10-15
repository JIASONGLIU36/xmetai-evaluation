"""Figures from the same paired summary used to render the report."""
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xmetai_evaluation.visualization.precipitation_plots import setup_chinese_font


def plot_typhoon(summary: dict, out_dir: Path) -> Dict[str, str]:
    font = setup_chinese_font()
    names = summary["models"]
    metrics = summary["metrics"]
    components = [
        key for key, info in metrics.items()
        if key != "track" and any(row["n_case"] for row in info["by_lead"])
    ]
    titles = {
        "track": "Track error", "along_track": "Along-track error",
        "cross_track": "Cross-track error", "minimum_pressure": "Minimum pressure error",
        "maximum_wind": "Maximum wind error",
    }
    figures = {}
    palette = ("#e76f2e", "#2878b5", "#2a9d5b", "#b23a48")
    presentation = summary["presentation"]
    for suffix, panels in [("track", ["track"]), ("mae", components), ("me", components)]:
        if not panels:
            continue
        figure, axes = plt.subplots(
            len(panels), 1, figsize=(11, 4 * len(panels)), squeeze=False
        )
        try:
            for axis, metric in zip(axes[:, 0], panels):
                info = metrics[metric]
                rows = info["by_lead"]
                leads = np.asarray([row["lead_h"] for row in rows])
                for group, shade in (("exploratory_leads", "#eeeeee"),
                                     ("reference_leads", "#dddddd")):
                    selected = set(presentation[group])
                    for index, lead in enumerate(leads):
                        if lead in selected:
                            left = (leads[index - 1] + lead) / 2 if index else lead - 6
                            right = (lead + leads[index + 1]) / 2 if index + 1 < len(leads) else lead + 6
                            axis.axvspan(left, right, color=shade, alpha=0.6, zorder=0)
                value_key, interval_key = ("me", "me_ci95") if suffix == "me" else (
                    "mean", "ci95"
                )
                for index, name in enumerate(names):
                    values = [row["models"][name] for row in rows]
                    color = palette[index] if index < len(palette) else plt.get_cmap("tab10")(index % 10)
                    axis.plot(
                        leads, [np.nan if value[value_key] is None else value[value_key]
                                for value in values],
                        marker="o", label=name, color=color,
                    )
                    axis.fill_between(
                        leads,
                        [np.nan if value[interval_key][0] is None else value[interval_key][0]
                         for value in values],
                        [np.nan if value[interval_key][1] is None else value[interval_key][1]
                         for value in values],
                        color=color, alpha=0.12,
                    )
                axis.set_title(
                    f"{info['title'] if font else titles[metric]} "
                    f"({'ME' if suffix == 'me' else 'Mean' if metric == 'track' else 'MAE'})"
                )
                axis.set_ylabel(info["unit"])
                axis.set_xlabel("Forecast lead time (h)")
                axis.set_xticks(leads)
                axis.set_xticklabels([
                    f"{row['lead_h']:g}\nN={row['n_case']}, storms={row['n_storm']}"
                    + (" *" if row["small_sample"] else "") for row in rows
                ], fontsize=8)
                if suffix == "me":
                    axis.axhline(0, color="gray", linestyle="--", linewidth=1)
                else:
                    axis.set_ylim(bottom=0)
                axis.grid(axis="y", linestyle=":", alpha=0.4)
                if axis is axes[0, 0]:
                    axis.legend(ncol=min(4, len(names)), frameon=False, loc="upper left")
            figure.suptitle(
                "Strictly paired cases | 95% storm-block bootstrap CI\n"
                "Light gray: exploratory; gray: case reference; * small sample",
                fontsize=11,
            )
            figure.tight_layout()
            filename = f"{suffix}.png"
            figure.savefig(out_dir / filename, dpi=180)
            figures[suffix] = filename
        finally:
            plt.close(figure)
    return figures
