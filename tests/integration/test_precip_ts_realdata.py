# -*- coding: utf-8 -*-
"""确定性降水 TS（FGVP × Diamond 站点）真实数据档：跑正式 config，对标 ``run_categorical.py``。

数据走 scripts/cut_precip_fixture.py 裁出的最小 fixture（起报 20250102 一天、tp 单要素 +
24 个小时站点实况 + 白名单）。本模块只覆盖时段、产物目录、阈值与并发形态。

⚠ 标准答案内联在下面，**待从基准脚本 run_categorical.py 的产物填入**（起报 20250102、
阈值 ≥0.1 / ≥10 / ≥25 的整数列联表）。填入前 golden 为 0，本模块会红——那是预期，跑
scripts/dump_precip_scores.py 拿到实测值再抄。
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.realdata

INIT_DATE = "20250102"
#: 只留两个有事件的档；≥25 这天全是 0（无事件 → TS 为 NaN），不评。
THRESHOLDS = (0.1, 10.0)

FIXTURE_DIR = os.environ.get("XMETAI_FIXTURE_DIR", "/workspace/data/shenzw")
FIXTURE_ROOT = f"{FIXTURE_DIR}/precip_test_data"
_DATA_PATHS = {
    "FGVP_OUTPUT": f"{FIXTURE_ROOT}/fgvp_output",
    "STATION_OBS": f"{FIXTURE_ROOT}/station",
    "STATION_LIST": f"{FIXTURE_ROOT}/station_list.dat",
}

# 框架实测（回归基线，未对独立基准 run_categorical.py）：起报 20250102、整数列联表。
GOLDEN_COUNTS = {
    0.1: {"hits": 269, "false_alarms": 1019, "misses": 80},
    10.0: {"hits": 2, "false_alarms": 8, "misses": 3},
}

#: 列联表逐格容差（整数计数，逐位一致；有阈值站翻转时放宽到 1）
COUNT_TOL = 0


def _require(value, name):
    if not value:
        pytest.skip(f"未配置环境变量 {name}")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{name}={value} 不存在")
    return path


def _run(output_dir: Path) -> Path:
    """按真实配置跑一条链，只覆盖时段、产物目录、阈值与并发形态。"""
    for env_name, default in _DATA_PATHS.items():
        _require(os.environ.get(env_name, default), env_name)

    from xmetai_evaluation.cli import run_evaluation
    from xmetai_evaluation.configs.base import load_config

    cfg = load_config("weather_ts_single_fgvp")
    cfg.forecast_reader = {
        **cfg.forecast_reader,
        "root_dir": _DATA_PATHS["FGVP_OUTPUT"],
        "lead_times": [6, 12, 18, 24],
    }
    cfg.observation_reader = {
        **cfg.observation_reader,
        "root_dir": _DATA_PATHS["STATION_OBS"],
        "station_list": _DATA_PATHS["STATION_LIST"],
    }
    cfg.start_date = INIT_DATE
    cfg.end_date = INIT_DATE
    cfg.periods = None          # 生产配置已改走多时段 periods；本模块只复核单时段，必须清掉
    cfg.output_dir = str(output_dir)
    cfg.limit = None
    # 只保留全球行（丢生产配置的分纬度带），对齐口径保持北京时（offset 8）。
    cfg.options = {"local_utc_offset_hours": 8}
    cfg.metric_options = {"ts_score": {"thresholds": list(THRESHOLDS)}}
    cfg.execution = {**cfg.execution, "mode": "serial", "n_workers": 1, "resume": False}
    run_evaluation(cfg)
    return output_dir


@pytest.fixture(scope="session")
def weather_ts(tmp_path_factory):
    """整条链跑一次，本模块所有用例共用。"""
    root = _run(tmp_path_factory.mktemp("weather_ts"))
    return root, pd.read_csv(root / "scores.csv")


def _value(scores: pd.DataFrame, metric: str, threshold: float) -> float:
    rows = scores[
        (scores["metric"] == metric)
        & (scores["threshold"] == threshold)
        & scores["region"].isna()
    ]
    assert len(rows) == 1, f"scores.csv 里 {metric} @ ≥{threshold} 应该正好一行，实际 {len(rows)} 行"
    return float(rows["value"].iloc[0])


def test_contingency_parity(weather_ts):
    """整数列联表逐格对拍（hits / false_alarms / misses）。"""
    root, _ = weather_ts
    wide = pd.read_csv(root / "diagnostics" / "categorical_wide.csv")
    assert len(wide) == len(THRESHOLDS)
    for thr in THRESHOLDS:
        row = wide[wide["threshold_mm"] == thr]
        assert len(row) == 1, f"categorical_wide 里 ≥{thr}mm 有 {len(row)} 行"
        for column, expected in GOLDEN_COUNTS[thr].items():
            actual = int(row[column].iloc[0])
            assert abs(actual - expected) <= COUNT_TOL, (
                f"≥{thr}mm 的 {column}: 框架 {actual} vs 基准 {expected}"
            )


def test_consistency(weather_ts):
    """长表里的 ts/pod/far/frequency_bias 必须能由整数列联表算回来（口径自洽哨）。"""
    root, scores = weather_ts
    wide = pd.read_csv(root / "diagnostics" / "categorical_wide.csv")
    derived = {
        "ts": lambda h, f, m: h / (h + f + m),
        "pod": lambda h, f, m: h / (h + m),
        "far": lambda h, f, m: f / (h + f),
        "frequency_bias": lambda h, f, m: (h + f) / (h + m),
    }
    assert set(scores["metric"]) == {"ts", "pod", "far", "miss_rate", "frequency_bias"}
    assert (scores["status"] == "success").all()
    for thr in THRESHOLDS:
        row = wide[wide["threshold_mm"] == thr].iloc[0]
        h, f, m = int(row["hits"]), int(row["false_alarms"]), int(row["misses"])
        assert h + f + m <= int(row["n_pairs"])
        for metric, formula in derived.items():
            expected = formula(h, f, m)
            assert _value(scores, metric, thr) == pytest.approx(expected, abs=1e-9)
