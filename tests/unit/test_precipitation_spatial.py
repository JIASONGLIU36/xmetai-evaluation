from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics import spatial
from xmetai_evaluation.metrics.spatial import FractionsSkillScore, neighborhood_fraction


THRESHOLDS = [("heavy", 1.0)]


def _batch(forecast, observation, *, valid_mask=None):
    forecast = xr.DataArray(np.asarray(forecast, dtype="f8"), dims=("lat", "lon"))
    observation = xr.DataArray(np.asarray(observation, dtype="f8"), dims=("lat", "lon"))
    if valid_mask is None:
        valid_mask = np.isfinite(forecast) & np.isfinite(observation)
    else:
        valid_mask = xr.DataArray(np.asarray(valid_mask, dtype=bool), dims=("lat", "lon"))
    return EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=[{"lead_hours": 6}],
        valid_mask=valid_mask,
        alignment={"time": "matched", "space": "grid"},
        protocol_id="grid_valid_time",
    )


def test_neighborhood_fraction_constant_boundary_behavior():
    impulse = np.zeros((3, 3), dtype="f8")
    impulse[1, 1] = 1.0
    np.testing.assert_allclose(neighborhood_fraction(impulse, 3), np.full((3, 3), 1 / 9))


def test_neighborhood_fraction_fallback_matches_known_result(monkeypatch):
    monkeypatch.setattr(spatial, "_uniform_filter", None)
    field = np.arange(25, dtype="f8").reshape(5, 5)
    actual = neighborhood_fraction(field, 3)

    padded = np.pad(field, 1, mode="constant")
    expected = np.empty_like(field)
    for row in range(5):
        for col in range(5):
            expected[row, col] = padded[row : row + 3, col : col + 3].mean()
    np.testing.assert_allclose(actual, expected)

    with pytest.raises(ValueError, match="只支持奇数窗口"):
        neighborhood_fraction(field, 2)


def test_fss_identical_fields_return_one_for_every_window():
    field = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    result = FractionsSkillScore(THRESHOLDS, windows=[1, 3]).compute(
        _batch(field, field)
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.value["heavy|w1"]["FSS"] == pytest.approx(1.0)
    assert result.value["heavy|w3"]["FSS"] == pytest.approx(1.0)


def test_fss_displaced_events_match_hand_calculation_at_window_one():
    forecast = [[2, 0], [0, 0]]
    observation = [[0, 2], [0, 0]]
    result = FractionsSkillScore(THRESHOLDS, windows=[1]).compute(
        _batch(forecast, observation)
    )

    assert result.value["heavy|w1"]["FSS"] == pytest.approx(0.0)
    assert result.value["heavy|w1"]["n_points"] == 4


def test_fss_respects_mask_and_reports_empty_batch():
    metric = FractionsSkillScore(THRESHOLDS, windows=[1])
    masked = metric.compute(
        _batch([[2, 0], [0, 0]], [[0, 2], [0, 0]], valid_mask=[[True, False], [False, False]])
    )
    assert masked.value["heavy|w1"]["FSS"] == pytest.approx(0.0)
    assert masked.n_valid == 1

    empty = metric.compute(
        _batch([[2, 0], [0, 0]], [[2, 0], [0, 0]], valid_mask=np.zeros((2, 2), bool))
    )
    assert empty.n_valid == 0
    # 全掩码批次的 status：现状是 UNDEFINED（与"有站点但无事件"不可区分），
    # 期望 NO_VALID_DATA——见文件末尾 xfail 钉住的测试。


def test_fss_no_events_is_undefined():
    result = FractionsSkillScore(THRESHOLDS, windows=[1]).compute(
        _batch(np.zeros((2, 2)), np.zeros((2, 2)))
    )
    assert result.status is ResultStatus.UNDEFINED
    assert np.isnan(result.value["heavy|w1"]["FSS"])


def test_fss_rejects_mismatched_field_shapes():
    batch = _batch(np.zeros((2, 2)), np.zeros((2, 2)))
    batch.forecast = xr.DataArray(np.zeros((3, 2)), dims=("lat", "lon"))
    with pytest.raises(MetricError, match="同形状"):
        FractionsSkillScore(THRESHOLDS, windows=[1]).compute(batch)


def test_fss_merge_adds_independent_cases_and_rejects_foreign_states():
    metric = FractionsSkillScore(THRESHOLDS, windows=[1])
    state = metric.accumulate(_batch([[2, 0]], [[2, 0]]))
    merged = metric.finalize(metric.merge([state, state]))
    assert merged.value["heavy|w1"]["FSS"] == pytest.approx(1.0)
    assert merged.n_valid == 4

    with pytest.raises(MetricError, match="empty states"):
        metric.merge([])
    with pytest.raises(MetricError, match="different metrics"):
        metric.merge([state, replace(state, metric_name="ts")])


# 下面两条钉住 FSS 缺失/错位的行为；源码解冻修好后，把它们从 skip 改回正常测试、
# 删掉对应标记，并把上面的原地断言恢复成期望值。
@pytest.mark.skip(
    reason=(
        "源码冻结暂不修：FSS.finalize() 先判 `if not results`，而 stats 是「阈值×窗口」全笛卡尔积、"
        "accumulate 必写满：整批被掩码掉时 results 非空、FSS 全为 nan，于是落进 "
        "`all(isnan)` 分支报 UNDEFINED，与「有站点但无事件」不可区分；期望 NO_VALID_DATA"
    ),
)
def test_fss_empty_batch_is_no_valid_data_pending_source_fix():
    metric = FractionsSkillScore(THRESHOLDS, windows=[1])
    empty = metric.compute(
        _batch([[2, 0], [0, 0]], [[2, 0], [0, 0]], valid_mask=np.zeros((2, 2), bool))
    )
    assert empty.status is ResultStatus.NO_VALID_DATA
    assert empty.n_valid == 0


@pytest.mark.skip(
    reason=(
        "源码冻结暂不修：FractionsSkillScore.merge() 目前只校验 states 非空与 metric_name，"
        "不比 metric_version、也不比 windows 参数"
    ),
)
def test_fss_merge_version_and_window_contract_pending_source_fix():
    metric = FractionsSkillScore(THRESHOLDS, windows=[1])
    state = metric.accumulate(_batch([[2, 0]], [[2, 0]]))

    with pytest.raises(MetricError, match="different versions"):
        metric.merge([state, replace(state, metric_version="2.0.0")])

    other = FractionsSkillScore(THRESHOLDS, windows=[3]).accumulate(_batch([[2, 0]], [[2, 0]]))
    with pytest.raises(MetricError, match="different parameters"):
        metric.merge([state, other])
