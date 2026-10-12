#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多个暴雨过程个例产物根 → 多模型对比图 + Markdown 报告（降水个例分析）。

用法::

    python skills/xmetai-evaluation/scripts/generate_rainstorm_report.py \\
      --model FuXi=/workspace/szwCode/evaluation_results/rainstorm_ts_single_fuxi \\
      --model FGVP-ctrl=/workspace/szwCode/evaluation_results/rainstorm_ts_single_fgvp_ctrl \\
      --process 202503,202521,202531 \\
      --result /workspace/szwCode/evaluation_results/rainstorm_report

``--model NAME=目录`` 可重复，至少两个。目录是
``configs/rainstorm_ts_single_*`` 的 ``OUTPUT_ROOT``（一个产物根底下有
``{编号}_24h/{起报日}/`` 与 ``{编号}_total/`` 两套段），**不是**单个过程的产物目录。

``NAME`` 是**报告里显示的模型名**，由调用方给——不从目录名、也不从
``manifest.run_id`` 猜。第一个 ``--model`` 是**主模型**，Δ 一律是「主模型 − 第二个」。

``--process`` **必给**，逗号分隔的过程编号（``202501``–``202543``）。编目里 43 个过程
全出会让报告变成一本册子，所以这里要求显式点名：先跑
``rainstorm_summary.py`` 拿到 ``screening.md``，照着挑。

其余选项：

    --result DIR    报告输出目录（必给，不存在会建）
    --title TEXT    覆盖报告标题那一行
    --change TEXT   本次改动说明，写进第一节与「改进建议」
    --archive TEXT  归档名，写进报告抬头的「归档：」

**两种口径互不可比**：口径 A（逐日 24h）一天一个 08–08 窗、基准率随每日实况浮动，
口径 B（过程累积）一根预报累加整段、基准率显著更高。报告里两套数分节呈现，
只在同一口径内跨模型、跨过程比。

**报告不出降水空间分布**——产物里只有站点列联表的聚合统计，没有逐站或逐格降水量，
画不出实况场、也画不出预报−实况差值场（原因写在报告第六节）。

**本脚本只做入口**：解析参数 → 把仓库根加进 ``sys.path`` → 调
``visualization.rainstorm_report.build_rainstorm_report`` → 打印产物清单。
加载、统计、渲染的逻辑**全在** ``visualization/`` 里。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: scripts/ -> xmetai-evaluation/ -> skills/ -> 仓库根
REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_model(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--model 要写成 NAME=目录，收到的是 {value!r}")
    name, directory = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--model 的模型名不能是空的: {value!r}")
    return name, Path(directory.strip())


def main(argv=None) -> int:
    # Windows 控制台默认 GBK，报告正文和日志里有大量中文。控制台代码页不是
    # UTF-8 时（GBK 有中文，但 cp437 / cp1252 这类没有），print 会抛
    # UnicodeEncodeError——报告和图其实都已经写完了，却在最后一步非 0 退出。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="按多个暴雨过程个例产物根出图并写多模型对比报告",
        epilog="示例: python skills/xmetai-evaluation/scripts/generate_rainstorm_report.py "
        "--model FuXi=evaluation_results/rainstorm_ts_single_fuxi "
        "--model FGVP-ctrl=evaluation_results/rainstorm_ts_single_fgvp_ctrl "
        "--process 202503,202521 --result reports/rainstorm",
    )
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME=目录", required=True,
        help="rainstorm_ts_single_* 的输出根（可重复，至少两个，第一个是主模型）",
    )
    parser.add_argument(
        "--process", default=None, metavar="编号[,编号...]", required=True,
        help="要详析的过程编号，逗号分隔（202501–202543）；编目全表见 rainstorm_catalog.py",
    )
    parser.add_argument("--result", type=Path, required=True, help="报告输出目录")
    parser.add_argument("--title", default=None, help="覆盖报告标题那一行")
    parser.add_argument("--change", default=None, help="本次改动说明，写进报告第一节")
    parser.add_argument("--archive", default=None, help="归档名，写进报告抬头的「归档：」")
    args = parser.parse_args(argv)

    models = []
    for raw in args.model:
        try:
            models.append(parse_model(raw))
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
    if len(models) < 2:
        parser.error(f"多模型报告至少要两个 --model，现在只有 {len(models)} 个")
    names = [name for name, _ in models]
    if len(set(names)) != len(names):
        parser.error(f"模型名重复了: {names}")

    processes = [item.strip() for item in args.process.split(",") if item.strip()]
    if not processes:
        parser.error("--process 不能是空的：给至少一个过程编号，如 --process 202503,202521")

    sys.path.insert(0, str(REPO_ROOT))
    from xmetai_evaluation.visualization.rainstorm_report import (
        build_rainstorm_report,
        load_archive,
    )

    try:
        archives = [load_archive(name, directory) for name, directory in models]
        artifacts = build_rainstorm_report(
            archives,
            args.result,
            processes=processes,
            title=args.title or "2025 年暴雨过程个例检验报告",
            archive=args.archive,
            change=args.change,
        )
    except ValueError as error:
        parser.error(str(error))

    print(f"\n产出 {len(artifacts)} 个文件：")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
