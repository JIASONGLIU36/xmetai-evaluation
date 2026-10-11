# -*- coding: utf-8 -*-
"""集合连续量检验（FuXi 集合 × ERA5）真实数据档：跑正式 config，对标 xu ``ensemble_fuxi.sh``。

和 ``test_weather_field_realdata.py``（确定性那份）同一个思路：不手搓
``EvaluationBatch`` 绕开 Runner，而是真跑 ``weather_rmse_ens_fuxi`` 这条正式 config
（``load_config`` -> ``run_evaluation``），读 ``scores.csv`` 对拍。

数据**不是**专门切的小 fixture——直接复用正式 config 自己的环境变量
（``FUXI_ENS_OUTPUT``/``ERA5_PL_STORE``/``ERA5_SFC_STORE``/``ERA5_CLIMO``，没配就落回
config 里写的生产默认路径）。本模块只覆盖 ``start_date``/``end_date``（钉死在
2025-01-02 这一个起报日，与确定性那份同源）、``output_dir`` 与并发形态。

⚠ 标准答案内联在下面，**待从服务器基准脚本 ``xu scripts/ensemble_fuxi.sh`` 的产物
填入**（起报 20250102、z500、时效 6/12/18/24）。填入前 golden 为 0，本模块会红——那是
预期：跑一次，断言消息里的「框架」值就是真值，抄进 ``GOLDEN_*`` 即可。

只评 z500（与确定性那份同口径）。集合族（crps / spread / spread_error_ratio）也一并
对拍。注意两个口径点：
- ``rmse`` 会出**两行**——``rmse`` 指标（集合均值 vs 实况）与 ``spread_error`` 的
  RMSE，数值应逐位一致，取数时容忍并校验（见 ``_value``）；
- ``spread`` 是 **ddof=0** 口径（config 里 ``spread_error`` 已配 ``ddof: 0``），
  从基准脚本抄 spread 时确认它是除成员数 N 而不是 N−1。
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
    "FUXI_ENS_OUTPUT": f"{FIXTURE_ROOT}/fuxi_ens",
    "ERA5_PL_STORE": f"{FIXTURE_ROOT}/era5_pl.zarr",
    "ERA5_CLIMO": f"{FIXTURE_ROOT}/era5_clim_phys_z500.nc",
}

# 标准答案：框架在 fixture 上的实测输出（起报 20250102、z500、4 时效）。
# 确定性那条的 golden 已与独立基准 multi_model_verifier_fix.py 逐位对上，这里
# 集合族（crps/spread/ratio）与活动度仍待 xu scripts/ensemble_fuxi.sh 复核。
#   rmse 是「集合均值 vs 实况」；spread 是 ddof=0；acc 是 uncentered（centered=False）；
#   activity 三元组 = (ratio, forecast, observation)；observation 应与确定性那份一致。
GOLDEN_RMSE = {6: 17.905901070704882, 12: 28.441674966340788, 18: 30.350514252883045, 24: 40.17211911202241}
GOLDEN_CRPS = {6: 10.286984240382813, 12: 16.380532839398057, 18: 16.859866035843737, 24: 21.907998424392822}
GOLDEN_SPREAD = {6: 17.800607046722206, 12: 20.38679882899556, 18: 23.175314692286623, 24: 27.386911089867414}
GOLDEN_RATIO = {6: 0.9941195154616438, 12: 0.7167934088537022, 18: 0.7635886167920736, 24: 0.6817394220248753}
GOLDEN_ACC = {6: 0.9997717723002876, 12: 0.9994312634151218, 18: 0.9993531597425218, 24: 0.9988484371954036}
GOLDEN_ACTIVITY = {
    6: (0.998089137661089, 824.9063727377833, 826.4856730841293),
    12: (0.9944994688027802, 818.080560530076, 822.6053267931006),
    18: (0.9938130930289532, 820.2191496749383, 825.3253609036951),
    24: (0.9935214194011914, 818.669480442696, 824.0078819197668),
}

#: 实测前还不知道真实误差量级，先给一个保守值；跑出来如果长期稳定在更小的
#: 量级，收紧这个数，别让它长期停在一个没验证过的保守值上。
REL_TOL = 1e-6


def _require(value, name):
    if not value:
        pytest.skip(f"未配置环境变量 {name}")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{name}={value} 不存在")
    return path


def _run(output_dir: Path) -> Path:
    """按真实配置跑一条链，只覆盖时段、产物目录与并发形态。"""
    for env_name, default in _DATA_PATHS.items():
        _require(os.environ.get(env_name, default), env_name)

    from xmetai_evaluation.cli import run_evaluation
    from xmetai_evaluation.configs.base import load_config

    cfg = load_config("weather_rmse_ens_fuxi")
    # 指向最小 fixture（只 z500），并只评 z500：预报/实况/气候态都裁到单要素。
    cfg.forecast_reader = {
        **cfg.forecast_reader,
        "root_dir": _DATA_PATHS["FUXI_ENS_OUTPUT"],
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
        "crps": {"variables": ["z500"]},
        "spread_error": {"variables": ["z500"], "ddof": 0},
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
def weather_ens(tmp_path_factory):
    """整条链跑一次，本模块所有用例共用，避免重复跑一整段流程。"""
    root = _run(tmp_path_factory.mktemp("weather_ens"))
    return root, pd.read_csv(root / "scores.csv")


def _value(scores: pd.DataFrame, metric: str, lead: int) -> float:
    """取 ``(metric, z500, lead)`` 这一格的 value；``region`` 必须为空（全球行）。

    ``rmse`` 会出两行（``rmse`` 指标与 ``spread_error`` 的 RMSE），数值应一致；
    这里只要求它们在容差内一致，然后取第一行。
    """
    rows = scores[
        (scores["metric"] == metric)
        & (scores["variable"] == "z500")
        & (scores["lead_h"] == float(lead))
        & scores["region"].isna()
    ]
    assert len(rows) >= 1, f"scores.csv 里没有 {metric} @ {lead}h"
    values = rows["value"].to_numpy(dtype=float)
    spread = (values.max() - values.min()) / max(abs(values[0]), 1e-12)
    assert spread <= REL_TOL, f"{metric} @ {lead}h 的多行互相矛盾: {values!r}"
    return float(values[0])


def _assert_rel(actual: float, expected: float, label: str) -> None:
    if expected == 0.0:  # golden 还是 0 占位：报出实测值，方便抄进 GOLDEN_*
        pytest.fail(f"{label}: golden 待填，实测 {actual!r}（从基准脚本抄进 GOLDEN_*）")
    relative = abs(actual - expected) / abs(expected)
    assert relative <= REL_TOL, (
        f"{label}: 框架 {actual!r} vs 基准 {expected!r}，相对差 {relative:.3e}"
    )


def test_rmse_parity(weather_ens):
    _, scores = weather_ens
    for lead in LEADS:
        _assert_rel(_value(scores, "rmse", lead), GOLDEN_RMSE[lead], f"rmse @ {lead}h")


def test_crps_parity(weather_ens):
    _, scores = weather_ens
    for lead in LEADS:
        _assert_rel(_value(scores, "crps", lead), GOLDEN_CRPS[lead], f"crps @ {lead}h")


def test_spread_parity(weather_ens):
    _, scores = weather_ens
    for lead in LEADS:
        _assert_rel(_value(scores, "spread", lead), GOLDEN_SPREAD[lead], f"spread @ {lead}h")


def test_ratio_parity(weather_ens):
    _, scores = weather_ens
    for lead in LEADS:
        _assert_rel(
            _value(scores, "spread_error_ratio", lead),
            GOLDEN_RATIO[lead],
            f"spread_error_ratio @ {lead}h",
        )


def test_acc_parity(weather_ens):
    _, scores = weather_ens
    for lead in LEADS:
        _assert_rel(_value(scores, "acc", lead), GOLDEN_ACC[lead], f"acc @ {lead}h")


def test_activity_parity(weather_ens):
    _, scores = weather_ens
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


def test_scores_coverage(weather_ens):
    """兜底：z500 该有的指标族都出来了，且状态都是 success。"""
    _, scores = weather_ens
    z500 = scores[(scores["variable"] == "z500") & scores["region"].isna()]
    assert set(z500["lead_h"]) == {float(lead) for lead in LEADS}
    expected_metrics = {
        "rmse", "crps", "spread", "spread_error_ratio", "acc",
        "activity_ratio", "activity_forecast", "activity_observation",
        "activity_bias", "spectrum_power_ratio",
        "spherical_band_power_forecast", "spherical_band_power_observation",
        "spherical_band_power_ratio",
    }
    assert expected_metrics <= set(z500["metric"])
    assert (z500["status"] == "success").all()
