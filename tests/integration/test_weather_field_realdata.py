# -*- coding: utf-8 -*-
"""确定性连续量检验（FuXi x ERA5）真实数据档：跑正式 config，对标既有基准。

和 ``test_fdp_fengqing_realdata.py`` 同一个思路：不再手搓 ``EvaluationBatch``
绕开 Runner，而是真跑 ``weather_rmse_single_fuxi`` 这条正式 config
（``load_config`` -> ``run_evaluation``），读 ``scores.csv``/``diagnostics/`` 对拍。

数据**不是**专门切的小 fixture——直接复用正式 config 自己的环境变量
（``FUXI_OUTPUT``/``ERA5_PL_STORE``/``ERA5_SFC_STORE``/``ERA5_CLIMO``，没配就落回
config 里写的生产默认路径）。本模块只覆盖 ``start_date``/``end_date``（钉死在
2025-01-02 这一个起报日，对应基准值的来源）、``output_dir`` 与并发形态。

标准答案内联（只 z500，和现有几份 metric 级真实数据测试的基准值逐位相同，
来自 ``multi_model_verifier_fix.py``/``activity_spectrum_verifier.py`` 独立计算）：
``rmse``/``acc``——4 个时效各一个标量；``acc`` 只取 uncentered，因为正式
config 的 ``metric_options["acc"]["centered"]`` 就是 False，这条流程根本不产出
centered 口径。

谱（``zonal_spectrum``）不在这里做逐位对拍：诊断文件 ``diagnostics/spectrum_z500.csv``
是**全部样本的均值曲线**（见 ``components.py::_writer_spectrum`` 的 docstring），
而既有基准是**逐时效**的值——要对拍得先把 4 个时效的功率值在线性空间里平均、
再取 log10，容易在换算里引入新的误差来源。这里只做覆盖性检查（文件存在、
波数对、状态成功），逐时效的精确谱对拍留在 metric 级测试里。
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.realdata
pytest.importorskip("netCDF4")
pytest.importorskip("zarr")

INIT_DATE = "20250102"
LEADS = (6, 12, 18, 24)

#: 最小 fixture（scripts/cut_weather_field_fixture.py 裁出，起报 20250102 一天、只 z500）。
#: 环境变量仍可覆盖这些默认值（换位置直接改这一行或设环境变量）。
FIXTURE_DIR = os.environ.get("XMETAI_FIXTURE_DIR", "/workspace/data/shenzw")
FIXTURE_ROOT = f"{FIXTURE_DIR}/weather_field_test_data"
_DATA_PATHS = {
    "FUXI_OUTPUT": f"{FIXTURE_ROOT}/fuxi_single",
    "ERA5_PL_STORE": f"{FIXTURE_ROOT}/era5_pl.zarr",
    "ERA5_CLIMO": f"{FIXTURE_ROOT}/era5_clim_phys_z500.nc",
}

# 基准值来源见模块 docstring；ACC 使用正式配置的 uncentered 口径。
GOLDEN_RMSE = {6: 18.22720349001401, 12: 27.349550366035462,
               18: 29.06549381901426, 24: 36.496437989720626}
# 只有 uncentered：正式 config 的 acc 口径就是 centered=False。
GOLDEN_ACC = {6: 0.9997611245211051, 12: 0.9994583138978743,
              18: 0.9993916159869105, 24: 0.9990353251628522}
GOLDEN_ACTIVITY = {
    6: (1.0003266386465477, 826.7556352457763, 826.4856730841291),
    12: (1.0000233375318883, 822.6245243711462, 822.6053267931007),
    18: (1.000860821854183, 826.0358190111725, 825.3253609036951),
    24: (0.9995893896831011, 823.6695357822448, 824.0078819197671),
}
#: 实测前还不知道真实误差量级，先给一个保守值；跑出来如果长期稳定在
#: 某个更小的量级，收紧这个数，别让它长期停在一个没验证过的保守值上。
REL_TOL = 1e-6


def _require(value, name):
    if not value:
        pytest.skip(f"未配置环境变量 {name}")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{name}={value} 不存在")
    return path


def _run(output_dir: Path) -> Path:
    """按真实配置跑一条链，只覆盖时段、产物目录与并发形态。

    数据路径不在模块级检查——``pytest.skip`` 只能在函数体内裸调用，模块级要
    显式传 ``allow_module_level=True``；放进这个函数（由 ``weather_field``
    fixture 调用）就是普通的函数内 skip，不用特殊处理。
    """
    for env_name, default in _DATA_PATHS.items():
        _require(os.environ.get(env_name, default), env_name)

    from xmetai_evaluation.cli import run_evaluation
    from xmetai_evaluation.configs.base import load_config

    cfg = load_config("weather_rmse_single_fuxi")
    # 指向最小 fixture（只 z500），并只评 z500：预报/实况/气候态都裁到单要素。
    cfg.forecast_reader = {
        **cfg.forecast_reader,
        "root_dir": _DATA_PATHS["FUXI_OUTPUT"],
        "variables": ["z500"],
        "lead_times": [6, 12, 18, 24],  # 显式声明，避免框架「从 0 铺起」多要 lead 0 的实况
    }
    cfg.observation_reader = {
        **cfg.observation_reader,
        "stores": {"pl": _DATA_PATHS["ERA5_PL_STORE"]},
        "variables": ["z500"],
    }
    cfg.reference_reader = {
        **cfg.reference_reader,
        "root_dir": _DATA_PATHS["ERA5_CLIMO"],
    }
    cfg.metric_options = {
        "rmse": {"variables": ["z500"]},
        "acc": {"variables": ["z500"], "centered": False},
        "activity": {"variables": ["z500"]},
        "zonal_spectrum": {"variables": ["z500"], "max_wavenumber": 720},
        "spherical_bands": {"variables": ["z500"]},
    }
    cfg.start_date = INIT_DATE
    cfg.end_date = INIT_DATE
    cfg.output_dir = str(output_dir)
    cfg.limit = None
    cfg.execution = {**cfg.execution, "mode": "serial", "n_workers": 1, "resume": False}
    run_evaluation(cfg)
    return output_dir


@pytest.fixture(scope="session")
def weather_field(tmp_path_factory):
    """整条链跑一次，本模块所有用例共用，避免重复跑一整段流程。"""
    root = _run(tmp_path_factory.mktemp("weather_field"))
    return root, pd.read_csv(root / "scores.csv")


def _value(scores: pd.DataFrame, metric: str, lead: int) -> float:
    """取 ``(metric, z500, lead)`` 这一格的 value；``region`` 必须为空（全球行）。"""
    rows = scores[
        (scores["metric"] == metric)
        & (scores["variable"] == "z500")
        & (scores["lead_h"] == float(lead))
        & scores["region"].isna()
    ]
    assert len(rows) == 1, f"scores.csv 里 {metric} @ {lead}h 应该正好一行，实际 {len(rows)} 行"
    return float(rows["value"].iloc[0])


def _assert_rel(actual: float, expected: float, label: str) -> None:
    relative = abs(actual - expected) / abs(expected)
    assert relative <= REL_TOL, (
        f"{label}: 框架 {actual!r} vs 基准 {expected!r}，相对差 {relative:.3e}"
    )


def test_rmse_parity(weather_field):
    _, scores = weather_field
    for lead in LEADS:
        _assert_rel(_value(scores, "rmse", lead), GOLDEN_RMSE[lead], f"rmse @ {lead}h")


def test_acc_parity(weather_field):
    """正式 config 的 acc 只产出 uncentered 口径（见模块 docstring）。"""
    _, scores = weather_field
    for lead in LEADS:
        _assert_rel(_value(scores, "acc", lead), GOLDEN_ACC[lead], f"acc @ {lead}h")


def test_activity_parity(weather_field):
    _, scores = weather_field
    for lead in LEADS:
        ratio, forecast_activity, observation_activity = GOLDEN_ACTIVITY[lead]
        _assert_rel(
            _value(scores, "activity_ratio", lead), ratio, f"activity_ratio @ {lead}h"
        )
        _assert_rel(
            _value(scores, "activity_forecast", lead),
            forecast_activity,
            f"activity_forecast @ {lead}h",
        )
        _assert_rel(
            _value(scores, "activity_observation", lead),
            observation_activity,
            f"activity_observation @ {lead}h",
        )


def test_spectrum_coverage(weather_field):
    """谱只做覆盖性检查，精确对拍留给 metric 级测试（原因见模块 docstring）。"""
    out, scores = weather_field
    spectrum = pd.read_csv(out / "diagnostics" / "spectrum_z500.csv")
    assert list(spectrum.columns) == ["wavenumber", "wavelength_km", "pred_mean", "obs_mean"]
    assert spectrum["wavenumber"].min() == 1
    assert spectrum["wavenumber"].max() == 720
    ratio_rows = scores[(scores["metric"] == "spectrum_power_ratio") & (scores["variable"] == "z500")]
    assert (ratio_rows["status"] == "success").all()


def test_scores_coverage(weather_field):
    """兜底：z500 该有的几族指标都出来了，且状态都是 success。"""
    _, scores = weather_field
    z500 = scores[(scores["variable"] == "z500") & scores["region"].isna()]
    assert set(z500["lead_h"]) == {float(lead) for lead in LEADS}
    expected_metrics = {
        "rmse", "acc", "activity_ratio", "activity_forecast",
        "activity_observation", "activity_bias", "spectrum_power_ratio",
    }
    assert expected_metrics <= set(z500["metric"])
    assert (z500["status"] == "success").all()
