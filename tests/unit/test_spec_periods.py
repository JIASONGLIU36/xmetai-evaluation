# -*- coding: utf-8 -*-
"""多时段（``EvalConfig.periods``）的解析与落点。

不依赖任何数据：被测的是纯声明层（``PipelineSpec.declared_periods`` /
``init_candidate_times`` / ``period``）以及计划层的那个**唯一接入点**
``_discover_init_times``——它决定拿哪些候选起报日去 catalog 探测。

口径要点：多段的空档期**不能出现在候选起报里**（那才是多时段相对"跑整段"
省下来的东西），所以这里用假 catalog 把请求原样截下来断言。
"""
import pytest

from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.execution.plan import _discover_init_times
from xmetai_evaluation.pipeline.spec import PipelineSpec, SourceSpec


def _spec(**kwargs) -> PipelineSpec:
    """最小可用声明：只有 name + 两个数据源，其余走缺省。"""
    base = dict(
        name="t_periods",
        forecast=SourceSpec(reader="fuxi", params={"variable": "tp"}),
        observation=SourceSpec(
            reader="station", params={"variable": "precipitation"}
        ),
    )
    base.update(kwargs)
    return PipelineSpec(**base)


def _dates(values):
    return [value.strftime("%Y%m%d") for value in values]


class _EchoForecast:
    """假预报句柄：catalog 把请求原样当索引传回，reader 原样吐回起报列表。

    于是 ``_discover_init_times`` 的返回值就等于它请求的候选集——正是要断言
    的东西（不用真数据，也就不会"看起来过了但其实没探测到"）。
    """

    source_id = "fake"

    class _Catalog:
        def discover(self, request):
            return request

    class _Reader:
        def available_init_times(self, index):
            return list(index.init_times)

    def __init__(self):
        self.catalog = self._Catalog()
        self.reader = self._Reader()


# ── 单段：与加这个能力之前逐位一致 ────────────────────────────────────


def test_single_period_from_start_end():
    spec = _spec(start_date="20250101", end_date="20250103")
    assert _dates(spec.init_candidate_times()) == [
        "20250101", "20250102", "20250103",
    ]


def test_missing_start_date_and_periods_raises():
    with pytest.raises(ValueError, match="必须提供"):
        _spec().init_candidate_times()


def test_end_date_defaults_to_start_date():
    """只给 start_date = 评那一天（加这个能力之前就是这口径）。"""
    spec = _spec(start_date="20250101")
    assert _dates(spec.init_candidate_times()) == ["20250101"]


# ── 多段 ──────────────────────────────────────────────────────────────


def test_multiple_periods_union_without_gap():
    """空档期不能出现在候选起报里——这是多时段唯一省下来的东西。"""
    spec = _spec(periods=[("20250101", "20250103"), ("20250201", "20250202")])
    assert _dates(spec.init_candidate_times()) == [
        "20250101", "20250102", "20250103", "20250201", "20250202",
    ]


def test_plan_layer_requests_the_same_union():
    """计划层（切块与观测跨度的来源）必须拿到同一份并集。"""
    spec = _spec(periods=[("20250101", "20250103"), ("20250201", "20250202")])
    discovered = _discover_init_times(spec, _EchoForecast())
    assert _dates(discovered) == _dates(spec.init_candidate_times())


def test_overlapping_periods_are_deduplicated():
    """两段重叠时同一天只算一次，否则那天会被切两次块、样本重复计数。"""
    spec = _spec(periods=[("20250101", "20250105"), ("20250103", "20250106")])
    assert _dates(spec.init_candidate_times()) == [
        "20250101", "20250102", "20250103", "20250104", "20250105", "20250106",
    ]


def test_period_is_the_envelope_of_all_segments():
    """``period()`` 只用于展示与报错，给的是包络区间。"""
    spec = _spec(periods=[("20250101", "20250103"), ("20250601", "20250602")])
    start, end = spec.period()
    assert _dates([start, end]) == ["20250101", "20250602"]


def test_periods_accepts_datetime_like_values():
    spec = _spec(periods=[("2025-01-01", "2025-01-02")])
    assert _dates(spec.init_candidate_times()) == ["20250101", "20250102"]


# ── 错误口径 ──────────────────────────────────────────────────────────


def test_periods_with_start_date_raises():
    """两套时段同时给 = 配置错，不许静默选一套。"""
    spec = _spec(
        periods=[("20250101", "20250103")], start_date="20250601", end_date="20250602"
    )
    with pytest.raises(ConfigError, match="不能同时给"):
        spec.init_candidate_times()


@pytest.mark.parametrize(
    "bad",
    [
        [("20250101",)],                    # 少一个
        [("20250101", "20250103", "20250105")],  # 多一个
        [("20250601", "20250101")],         # 起止颠倒
        [("2025-13-45", "20250101")],       # 解析不了
    ],
)
def test_malformed_periods_raise_config_error(bad):
    with pytest.raises(ConfigError):
        _spec(periods=bad).init_candidate_times()


# ── 配置 -> 声明 透传 ─────────────────────────────────────────────────


def test_specs_from_config_carries_periods():
    from xmetai_evaluation.configs.base import EvalConfig
    from xmetai_evaluation.pipeline.spec import specs_from_config

    cfg = EvalConfig(
        name="t",
        description="",
        pipeline="weather_ts_det",
        forecast_reader={"type": "fuxi", "root_dir": "/tmp/f", "variable": "tp"},
        observation_reader={
            "type": "station",
            "root_dir": "/tmp/o",
            "variable": "precipitation",
        },
        periods=[("20250101", "20250103"), ("20250201", "20250202")],
    )
    (spec,) = specs_from_config(cfg)
    assert spec.periods == [("20250101", "20250103"), ("20250201", "20250202")]
    assert _dates(spec.init_candidate_times()) == [
        "20250101", "20250102", "20250103", "20250201", "20250202",
    ]
