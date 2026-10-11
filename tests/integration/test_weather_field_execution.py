# -*- coding: utf-8 -*-
"""执行层 synthetic 测试：多进程 / 窗缓存 / 驻留缓存 与串行逐位一致，resume 可复现。

生产配置（``weather_rmse_single_fuxi``）走 ``processes`` + ``observation=window:16`` +
``reference=resident`` + ``resume=True``，而其余所有测试都跑 ``serial + resume=False``，
这几条路径一直没被测到。本模块用一个**多起报日**的合成 fixture（2 个起报 → 2 个块），
验证：

- ``processes``（跨进程 pickle + 多块 merge）与 ``serial`` 结果一致；
- ``resume=True`` 会落 ``.states/`` 状态，重跑结果可复现。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests import runner_weather_field as runner
from xmetai_evaluation.cli import run_evaluation
from xmetai_evaluation.configs.base import load_config

#: 两个起报日 → ``chunk_days=1`` 下切成 2 个块，``processes`` 才会真正开多个 worker 并 merge。
INIT_TIMES = (datetime(2025, 1, 2), datetime(2025, 1, 3))


def _build_config(
    forecast_root, stores, climatology, output_dir, *, mode, n_workers, resume
):
    cfg = load_config("weather_rmse_single_fuxi")
    cfg.forecast_reader = {
        **cfg.forecast_reader,
        "root_dir": str(forecast_root),
        "variables": ["z500"],
        "lead_times": [6, 12],
    }
    cfg.observation_reader = {
        **cfg.observation_reader,
        "stores": {name: str(path) for name, path in stores.items()},
        "variables": ["z500"],
    }
    cfg.reference_reader = {
        **cfg.reference_reader,
        "root_dir": str(climatology),
        "smooth_days": 15,
        "units": {"z500": "m^2/s^2"},
    }
    cfg.start_date = "20250102"
    cfg.end_date = "20250103"
    cfg.output_dir = str(output_dir)
    cfg.limit = None
    cfg.options = {"sample_by": "init_lead"}
    cfg.metric_options = {
        "rmse": {"variables": ["z500"]},
        "acc": {"variables": ["z500"], "centered": False},
        "activity": {"variables": ["z500"]},
        "zonal_spectrum": {"variables": ["z500"], "max_wavenumber": 3},
        "spherical_bands": {"variables": ["z500"], "bands": [(1, 1), (2, 3)], "block": 1},
    }
    cfg.writers = ["csv_long", "spectrum"]
    # 保留生产配置的 loads（window:16 + resident），只改并发形态
    cfg.execution = {**cfg.execution, "mode": mode, "n_workers": n_workers, "resume": resume}
    return cfg


def _scores(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / "scores.csv")


_SORT_KEY = ["metric", "variable", "lead_h", "init_time", "region", "group"]


def _assert_same(scores_a: pd.DataFrame, scores_b: pd.DataFrame) -> None:
    """两块结果应逐位一致；value 列允许 float 求和次序的极小差（并行 merge 次序不同）。"""
    a = scores_a.sort_values(_SORT_KEY, na_position="last").reset_index(drop=True)
    b = scores_b.sort_values(_SORT_KEY, na_position="last").reset_index(drop=True)
    assert list(a.columns) == list(b.columns)
    assert a.shape == b.shape
    pd.testing.assert_frame_equal(a.drop(columns=["value"]), b.drop(columns=["value"]))
    assert np.allclose(
        a["value"].to_numpy(dtype=float),
        b["value"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=0.0,
        equal_nan=True,
    )


def test_processes_matches_serial(tmp_path):
    forecast_root = runner.write_synthetic_fuxi_phys_multi(tmp_path / "forecast", INIT_TIMES)
    stores = runner.write_synthetic_era5_zarr_multi(tmp_path / "era5", INIT_TIMES)
    climatology = runner.write_synthetic_daily_climatology(tmp_path / "climatology.nc")

    serial = _build_config(
        forecast_root, stores, climatology, tmp_path / "out_serial",
        mode="serial", n_workers=1, resume=False,
    )
    processes = _build_config(
        forecast_root, stores, climatology, tmp_path / "out_processes",
        mode="processes", n_workers=2, resume=False,
    )
    run_evaluation(serial)
    run_evaluation(processes)

    _assert_same(_scores(tmp_path / "out_serial"), _scores(tmp_path / "out_processes"))


def test_resume_reproduces_result(tmp_path):
    forecast_root = runner.write_synthetic_fuxi_phys_multi(tmp_path / "forecast", INIT_TIMES)
    stores = runner.write_synthetic_era5_zarr_multi(tmp_path / "era5", INIT_TIMES)
    climatology = runner.write_synthetic_daily_climatology(tmp_path / "climatology.nc")

    cfg = _build_config(
        forecast_root, stores, climatology, tmp_path / "out_resume",
        mode="serial", n_workers=1, resume=True,
    )
    run_evaluation(cfg)
    first = _scores(tmp_path / "out_resume")
    assert (tmp_path / "out_resume" / ".states").exists(), "resume 应落 .states/ 状态目录"

    # 同一 output_dir 再跑一次（resume=True），结果应完全复现
    run_evaluation(cfg)
    second = _scores(tmp_path / "out_resume")
    _assert_same(first, second)
