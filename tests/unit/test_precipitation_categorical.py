from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.base import ProductType
from xmetai_evaluation.metrics.categorical import TSScore


THRESHOLDS = [("rain", 1.0)]


def _batch(forecast, observation, *, valid_mask=None):
    forecast = xr.DataArray(np.asarray(forecast, dtype="f8"), dims=("station",))
    observation = xr.DataArray(np.asarray(observation, dtype="f8"), dims=("station",))
    if valid_mask is None:
        valid_mask = np.isfinite(forecast) & np.isfinite(observation)
    else:
        valid_mask = xr.DataArray(np.asarray(valid_mask, dtype=bool), dims=("station",))
    return EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=[{"lead_hours": 6}],
        valid_mask=valid_mask,
        alignment={"time": "matched", "space": "station"},
        protocol_id="station_valid_time",
    )


def test_ts_known_contingency_table_and_scores():
    result = TSScore(THRESHOLDS).compute(_batch([2, 2, 0, 0], [2, 0, 2, 0]))

    score = result.value["rain"]
    assert result.status is ResultStatus.SUCCESS
    assert score["hits"] == score["misses"] == score["false_alarms"] == 1
    assert score["correct_negatives"] == 1
    assert score["n_pairs"] == 4
    assert score["TS"] == pytest.approx(1 / 3)
    assert score["POD"] == pytest.approx(0.5)
    assert score["FAR"] == pytest.approx(0.5)
    assert score["miss_rate"] == pytest.approx(0.5)
    assert score["BIAS"] == pytest.approx(1.0)


def test_ts_threshold_is_inclusive_and_supports_multiple_levels():
    metric = TSScore([("one", 1.0), ("two", 2.0)])
    result = metric.compute(_batch([1.0, 2.0], [1.0, 1.5]))

    assert result.value["one"]["hits"] == 2
    assert result.value["two"]["false_alarms"] == 1
    assert result.value["two"]["hits"] == 0


def test_ts_respects_valid_mask():
    result = TSScore(THRESHOLDS).compute(
        _batch([2, 2, 0, 0], [2, 0, 2, 0], valid_mask=[True, False, True, False])
    )

    score = result.value["rain"]
    assert score["hits"] == 1
    assert score["misses"] == 1
    assert score["false_alarms"] == 0
    assert score["n_pairs"] == 2


def test_ts_no_events_is_partial_and_empty_mask_is_no_valid_data():
    no_events = TSScore(THRESHOLDS).compute(_batch([0, 0], [0, 0]))
    assert no_events.status is ResultStatus.PARTIAL
    assert np.isnan(no_events.value["rain"]["TS"])

    empty = TSScore(THRESHOLDS).compute(
        _batch([2, 0], [2, 0], valid_mask=[False, False])
    )
    assert empty.status is ResultStatus.NO_VALID_DATA
    assert empty.n_valid == 0


def test_ts_merged_chunks_equal_whole_batch():
    metric = TSScore(THRESHOLDS)
    whole = metric.compute(_batch([2, 2, 0, 0], [2, 0, 2, 0]))
    merged = metric.finalize(
        metric.merge(
            [
                metric.accumulate(_batch([2, 2], [2, 0])),
                metric.accumulate(_batch([0, 0], [2, 0])),
            ]
        )
    )

    assert merged.value == whole.value
    assert merged.n_valid == whole.n_valid


def test_ts_merge_rejects_empty_and_foreign_metric_states():
    metric = TSScore(THRESHOLDS)
    state = metric.accumulate(_batch([2], [2]))

    with pytest.raises(MetricError, match="empty states"):
        metric.merge([])
    with pytest.raises(MetricError, match="different metrics"):
        metric.merge([state, replace(state, metric_name="fss")])


def test_ts_requirements_and_deterministic_product_guard():
    metric = TSScore(THRESHOLDS)
    requirements = metric.requirements()
    assert requirements.product_type is ProductType.DETERMINISTIC_FIELD
    # TS 当前未声明 thresholds（fss 与概率评分已声明）——按现状记录；
    # 将来若补齐该声明，这里会失败，提示同步更新。
    assert requirements.thresholds is None

    batch = _batch([2, 0], [2, 0])
    batch.forecast = xr.DataArray(
        [[2.0, 0.0], [1.0, 0.0]], dims=("member", "station")
    )
    with pytest.raises(MetricError, match="要求确定性场"):
        metric.compute(batch)


# 下面这条钉住 TSScore.merge() 缺失的契约：目前只校验 states 非空与 metric_name，
# 不比 metric_version、也不比阈值参数。源码解冻修好后，把本测试从 skip 改回正常测试、
# 删掉标记，并把 pytest.raises 断言加回上面的 merge 测试。
@pytest.mark.skip(
    reason=(
        "源码冻结暂不修：TSScore.merge() 目前只校验 states 非空与 metric_name，"
        "不比 metric_version、也不比阈值参数"
    ),
)
def test_ts_merge_version_and_threshold_contract_pending_source_fix():
    metric = TSScore(THRESHOLDS)
    state = metric.accumulate(_batch([2], [2]))

    with pytest.raises(MetricError, match="different versions"):
        metric.merge([state, replace(state, metric_version="2.0.0")])

    other = TSScore([("rain", 5.0)]).accumulate(_batch([2], [2]))
    with pytest.raises(MetricError, match="different parameters"):
        metric.merge([state, other])
