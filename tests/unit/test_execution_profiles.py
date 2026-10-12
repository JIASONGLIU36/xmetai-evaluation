# -*- coding: utf-8 -*-
"""execution/profiles.py + execution/strategy.py：指标画像与执行形态推导。

对应台风套件里"协议/执行层纯函数"那一层（``test_typhoon_seed.py``）：钉的是
**注册名 → 画像（compute_class / needs_full_field）** 的对照表，以及
``mode="auto"`` 的落地结果。

这张表是"静默错"的典型：写错一个名字不会报错，只会让这一段悄悄少开进程。
实例名与注册名并不总是一致（``ts_score`` -> ``ts``、``ensemble_probability``
-> ``prob``、``acc_uncentered`` -> ``acc``），所以这里表驱动地遍历注册表，
新增指标忘了登记期望值、画像本身算错都会失败；而"按注册名登记、查表用实例名"
这一类的后果单独用一个 ``skip`` 钉住（``ensemble_probability`` 期望重指标却
回退成轻指标，``mode="auto"`` 因此少开进程）——**源码暂不改**，修好后把该 ``skip``
改回正常测试，并删掉标记与本文件里的对应常量。
"""
from types import SimpleNamespace

import pytest

from xmetai_evaluation.components import register_builtin_components
from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.core.registry import ComponentType, get_registry
from xmetai_evaluation.execution.profiles import (
    _FULL_FIELD_METRICS,
    _HEAVY_METRICS,
    _LIGHT_METRICS,
    ResourceProfile,
    _profile_for_name,
    needs_full_field,
    union_profile,
)
from xmetai_evaluation.execution.strategy import derive_strategy

#: 组件注册名 -> (期望 compute_class, 期望 needs_full_field)
EXPECTED_PROFILES = {
    "rmse": ("light", False),
    "bias": ("light", False),
    "acc": ("light", False),
    "acc_uncentered": ("light", False),
    "ts_score": ("light", False),
    "activity": ("light", False),
    "track_error": ("light", False),
    "track_error_ens": ("light", False),
    "crps": ("heavy", False),
    "spread_error": ("heavy", False),
    "ensemble_probability": ("heavy", False),
    "fss": ("heavy", True),
    "spectrum": ("heavy", True),
    "zonal_spectrum": ("heavy", True),
    "spherical_bands": ("heavy", True),
}

#: 已知缺陷（本次只动测试、不动源码）：画像表登记的是**组件注册名**，而
#: ``_profile_for_name`` 按**指标实例名**（``Metric.name``）查表，三个名字对不上：
#: ``acc_uncentered`` -> ``acc``、``ts_score`` -> ``ts``、``ensemble_probability``
#: -> ``prob``。前两个无害（``_LIGHT_METRICS`` 根本不参与查表，缺省即 light）；
#: 第三个是实打实的错——期望重指标却回退成轻指标，``mode="auto"`` 于是少开进程。
#: 修好 ``profiles.py`` 后把本文件的 ``skip`` 改回正常测试，请连同这两个常量一起删掉。
KNOWN_REGISTRATION_NAME_MISMATCH = "ensemble_probability"

#: 画像表里当前查不到（悬空）的名字：就是上面那三个。钉成快照，防止再欠新的。
KNOWN_DANGLING_TABLE_NAMES = frozenset({"acc_uncentered", "ts_score", "ensemble_probability"})


def _registry():
    register_builtin_components()
    return get_registry()


def _build(*registration_names):
    """按注册名构造指标实例（与运行期同一条路径：只给注册名和参数）。"""
    registry = _registry()
    return [registry.build(ComponentType.METRIC, name) for name in registration_names]


def _instance_name(registration_name):
    """注册名 -> 指标实例名：查画像用的是实例名，不是注册名。"""
    return _registry().build(ComponentType.METRIC, registration_name).name


def test_profile_expectation_table_covers_every_registered_metric():
    assert sorted(_registry().names(ComponentType.METRIC)) == sorted(
        EXPECTED_PROFILES
    ), "新增/改名指标时必须同时在这里登记它的画像期望值"


@pytest.mark.parametrize(
    "registration_name",
    [name for name in sorted(EXPECTED_PROFILES) if name != KNOWN_REGISTRATION_NAME_MISMATCH],
)
def test_registered_metric_resolves_to_its_declared_profile(registration_name):
    """按实例名查表的结果必须与 ``EXPECTED_PROFILES`` 一致（逐指标报出不符项）。"""
    metric = _registry().build(ComponentType.METRIC, registration_name)
    profile = _profile_for_name(metric.name)
    expected = EXPECTED_PROFILES[registration_name]
    assert (profile.compute_class, profile.needs_full_field) == expected, (
        f"{registration_name}（实例名 {metric.name!r}）画像不符："
        f"实际 {(profile.compute_class, profile.needs_full_field)}，期望 {expected}"
    )


@pytest.mark.skip(
    reason=(
        "源码冻结暂不修：_HEAVY_METRICS 登记的是注册名 ensemble_probability，"
        "查表用的却是实例名 prob，集合概率指标因此被当成轻量指标"
    ),
)
def test_ensemble_probability_metric_is_heavy():
    """集合概率指标必须是重指标，否则 ``mode="auto"`` 会退回线程并行。"""
    assert _profile_for_name(_instance_name("ensemble_probability")).compute_class == "heavy"
    assert union_profile(_build("ensemble_probability")).compute_class == "heavy"


def test_profile_name_tables_only_use_real_metric_instance_names():
    """画像表里的名字必须能被查表命中——查不到等于该条目静默失效。

    现状：有三个名字查不到（见 ``KNOWN_DANGLING_TABLE_NAMES``）。所以这里不直接
    断言"表 ⊆ 实例名"，而是钉住这三个**已知**名字：新指标若又写成注册名，照样失败。
    """
    instance_names = {_instance_name(name) for name in _registry().names(ComponentType.METRIC)}
    dangling = (_LIGHT_METRICS | _HEAVY_METRICS | _FULL_FIELD_METRICS) - instance_names
    assert dangling == KNOWN_DANGLING_TABLE_NAMES, (
        f"画像表出现新的悬空名字：{sorted(dangling - KNOWN_DANGLING_TABLE_NAMES)}；"
        f"已消失的：{sorted(KNOWN_DANGLING_TABLE_NAMES - dangling)}"
    )


def test_unknown_metric_name_falls_back_to_light_whole_field():
    assert _profile_for_name("mystery_metric") == ResourceProfile()
    assert _profile_for_name("") == ResourceProfile()

    stub = SimpleNamespace(name="mystery_metric")
    assert needs_full_field(stub) is False
    # 连 name 都没有（配置写错、传了裸对象）也不能炸，取最保守缺省
    assert needs_full_field(SimpleNamespace()) is False
    assert needs_full_field(SimpleNamespace(name=None)) is False


def test_union_profile_is_union_over_metrics():
    light, fss, probabilistic, active = _build("rmse", "fss", "ensemble_probability", "activity")
    # crps 是"按实例名命中的重指标"：要成员、但不吃完整场
    members = _build("crps")[0]

    assert union_profile([]) == ResourceProfile()

    light_only = union_profile([light])
    assert light_only.compute_class == "light"
    assert light_only.needs_full_field is False
    assert light_only.needs_members is False
    assert light_only.needs_reference is False

    mixed = union_profile([light, fss])
    assert mixed.compute_class == "heavy"
    assert mixed.needs_full_field is True
    assert mixed.needs_members is False

    # needs_members 来自指标自身（Metric.needs_members），与画像表无关——
    # 这正是"画像表写错名字"只影响并行度、不影响算不算成员的原因
    ensemble = union_profile([light, probabilistic])
    assert ensemble.needs_members is True
    assert ensemble.needs_full_field is False

    # activity 要气候态参考（与前两个都不沾成员）
    assert union_profile([active]).needs_reference is True
    assert union_profile([active]).needs_members is False

    assert union_profile([light, members, active]).as_dict() == {
        "needs_reference": True,
        "needs_members": True,
        "needs_full_field": False,
        "compute_class": "heavy",
    }


def test_auto_mode_lands_on_processes_for_heavy_metrics():
    """重指标必须走进程并行（``resolve_mode`` 按联合画像落地）。

    这里用 ``crps`` 走通这条路径：集合概率指标因上面那个已知缺陷暂时落在
    threads，见 ``test_ensemble_probability_metric_is_heavy``。
    """
    profile = union_profile(_build("crps"))
    assert profile.compute_class == "heavy"

    strategy = derive_strategy(
        profile,
        observation_reader="station",
        reference_reader=None,
        execution={"mode": "auto"},
        num_workers=4,
    )
    assert strategy.mode == "auto"  # 配置值原样保留，落地在 resolve_mode
    assert strategy.resolve_mode(n_chunks=8, profile=profile) == "processes"
    assert strategy.resolve_mode(n_chunks=1, profile=profile) == "serial"


def test_auto_mode_lands_on_threads_for_light_metrics():
    profile = union_profile(_build("rmse", "ts_score"))
    strategy = derive_strategy(profile, observation_reader="station", reference_reader=None)
    assert strategy.mode == "auto"
    assert strategy.resolve_mode(n_chunks=8, profile=profile) == "threads"
    assert strategy.resolve_mode(n_chunks=0, profile=profile) == "serial"


def test_explicit_mode_wins_over_profile():
    heavy = union_profile(_build("fss"))
    explicit = derive_strategy(heavy, "station", None, {"mode": "threads"})
    assert explicit.resolve_mode(n_chunks=8, profile=heavy) == "threads"


def test_resident_defaults_follow_reader_shape():
    profile = union_profile(_build("rmse"))
    station = derive_strategy(
        profile,
        observation_reader="diamond_station",
        reference_reader="daily_climatology",
    )
    assert station.load_policy("observation") == ("resident", 0)
    assert station.load_policy("reference") == ("resident", 0)

    big = derive_strategy(profile, observation_reader="era5_zarr", reference_reader=None)
    assert big.load_policy("observation") == ("slice", 0)
    # 没声明的角色也退回 slice，不是 KeyError
    assert big.load_policy("reference") == ("slice", 0)

    windowed = derive_strategy(
        profile,
        observation_reader="era5_zarr",
        reference_reader="climatology",
        execution={"loads": {"reference": "window:30"}},
    )
    assert windowed.load_policy("reference") == ("window", 30)
    assert windowed.load_policy("observation") == ("slice", 0)


def test_derive_strategy_rejects_bad_execution_config():
    profile = union_profile(_build("rmse"))
    with pytest.raises(ConfigError, match="未知键"):
        derive_strategy(profile, "station", None, {"mdoe": "auto"})
    with pytest.raises(ConfigError, match="无效的执行形态"):
        derive_strategy(profile, "station", None, {"mode": "parallel"})
    with pytest.raises(ConfigError, match="chunk_days"):
        derive_strategy(profile, "station", None, {"chunk_days": 0})
    with pytest.raises(ConfigError, match="lead_chunk_days"):
        derive_strategy(profile, "station", None, {"lead_chunk_days": -1})
    with pytest.raises(ConfigError, match="loads 必须是"):
        derive_strategy(profile, "station", None, {"loads": "resident"})
    with pytest.raises(ConfigError, match="角色 'forecast' 不存在"):
        derive_strategy(profile, "station", None, {"loads": {"forecast": "resident"}})
    # 逐样本直读概率的参考源没有"整包"可加载
    with pytest.raises(ConfigError, match="不提供整包加载"):
        derive_strategy(profile, "station", "ref_probability", {"loads": {"reference": "resident"}})
