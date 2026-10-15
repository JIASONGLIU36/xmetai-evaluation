#!/usr/bin/env python3
"""台风场次 CSV -> 统计、图表与报告；计算逻辑在主仓库。"""
import sys
from pathlib import Path


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from xmetai_evaluation.visualization.typhoon_report import main as report_main

    return report_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
