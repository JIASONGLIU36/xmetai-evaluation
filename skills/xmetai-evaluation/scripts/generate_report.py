#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测产物目录 → 图表 + Markdown 报告（skill 的唯一入口）。

用法::

    python skills/xmetai-evaluation/scripts/generate_report.py <产物目录> [选项]

    --result DIR       报告输出目录（默认 reports/<run_id>）
    --model NAME    报告标题里的模型名（默认取 manifest 的 run_id）
    --change TEXT   本次改动说明，写进报告第一节
    --compare 名字=CSV路径
                    对比模型的宽表，可重复；只在 TS 系列上有意义。
                    对比模型给的是**宽表**不是产物目录——它们没有 manifest。
                    evaluation_results/ 下一个模型一个目录，宽表都在
                    diagnostics/categorical_wide.csv。三个模式一起比：
                      --compare "FuXi=evaluation_results/weather_ts_det_fuxi/diagnostics/categorical_wide.csv" \\
                      --compare "AIFS=evaluation_results/weather_ts_det_aifs/diagnostics/categorical_wide.csv"
    --members N     集合成员数，只在集合产物上有意义（产物里没记这个字段）。
                    给出来后报告第一节会写明「集合每例 N 个成员」

**本脚本只做入口**：把仓库根加进 ``sys.path`` → 调
``xmetai_evaluation.visualization.report.build_report`` → 打印产物清单。

认能力、选模板、出图、写报告的逻辑**全在主仓库**
（``xmetai_evaluation/visualization/``），因为那里能被 ``pytest`` 覆盖。
脚本里一旦开始有判断逻辑，就又变成一份会主仓库漂移的副本。

之所以不直接 ``python -m xmetai_evaluation.visualization.report``：那要求包已经
``pip install -e .``。这里把仓库根插进 ``sys.path``，所以在没装过的机器上也能直接跑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: scripts/ -> xmetai-evaluation/ -> skills/ -> 仓库根
REPO_ROOT = Path(__file__).resolve().parents[3]


def main(argv=None) -> int:
    # Windows 控制台默认 GBK，报告的正文和日志里有大量中文。控制台代码页不是
    # UTF-8 时（GBK 有中文，但 cp437 / cp1252 这类没有），print 会抛
    # UnicodeEncodeError——报告和图其实都已经写完了，却在最后一步非 0 退出、
    # artifacts 也拿不到。在入口把标准输出切成 UTF-8，避免这种"功亏一篑"。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="按评测产物目录自动选模板，出图并写 Markdown 报告",
        epilog="示例: python skills/xmetai-evaluation/scripts/generate_report.py "
        "evaluation_results/weather_ts_single_fgvp --result reports/weather_ts_single_fgvp",
    )
    parser.add_argument("output_dir", type=Path, help="评测产物目录（含 manifest.json 与结果表）")
    parser.add_argument("--result", type=Path, default=None, help="报告输出目录（默认 reports/<run_id>）")
    parser.add_argument("--model", default=None, help="模型名（默认取 manifest 的 run_id）")
    parser.add_argument("--change", default=None, help="本次改动说明，写进报告第一节")
    parser.add_argument(
        "--compare", action="append", default=None, metavar="名字=CSV路径",
        help="对比模型的宽表（可重复），只在 TS 系列上有意义",
    )
    parser.add_argument(
        "--spectrum-variable", default=None,
        help="要点名看某一个起报的谱曲线的变量（读 diagnostics/spectrum_by_init.csv），"
        "只在连续场链路上有意义；该变量得真算过纬向谱",
    )
    parser.add_argument(
        "--spectrum-init", default=None,
        help="谱曲线的起报时刻，按子串匹配 init_time（如 20250102），省略取最后一个起报",
    )
    parser.add_argument(
        "--members", type=int, default=None,
        help="集合成员数（只在 weather_ts_ens 上有意义）。产物里没有这个字段，"
        "报告里凡要写成员数的地方都靠它；不给就省略数字",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT))
    from xmetai_evaluation.visualization.report import build_report

    artifacts = build_report(
        args.output_dir,
        out_dir=args.result,
        model_name=args.model,
        spectrum_variable=args.spectrum_variable,
        spectrum_init=args.spectrum_init,
        change_description=args.change,
        compare=args.compare,
        members=args.members,
    )
    print(f"\n产出 {len(artifacts)} 个文件：")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
