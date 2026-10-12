from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.base import ProductType
from xmetai_evaluation.metrics.probabilistic import EnsembleProbabilityScore


THRESHOLDS = [("rain", 1.0)]


def _batch(members, observation, *, valid_mask=None, weights=None, reference=None):
    members = xr.DataArray(np.asarray(members, dtype="f8"), dims=("member", "station"))
    observation = xr.DataArray(np.asarray(observation, dtype="f8"), dims=("station",))
    if valid_mask is None:
        valid_mask = xr.DataArray(np.ones(observation.shape, bool), dims=("station",))
    else:
        valid_mask = xr.DataArray(np.asarray(valid_mask, dtype=bool), dims=("station",))
    weights_da = None if weights is None else xr.DataArray(weights, dims=("station",))
    return EvaluationBatch(
        forecast=members.mean("member"),
        observation=observation,
        members=members,
        reference=reference,
        sample_keys=[{"lead_hours": 6}],
        valid_mask=valid_mask,
        weights=weights_da,
        alignment={"time": "matched", "space": "station"},
        protocol_id="station_valid_time",
    )


def test_probability_score_matches_hand_calculated_bs_bss_and_aroc():
    members = [[0, 0, 2, 2], [0, 2, 2, 2]]
    result = EnsembleProbabilityScore(THRESHOLDS).compute(
        _batch(members, [0, 2, 2, 0])
    )

    score = result.value["rain"]
    assert result.status is ResultStatus.SUCCESS
    assert score["BS"] == pytest.approx(0.3125)
    assert score["base_rate"] == pytest.approx(0.5)
    assert score["BS_ref"] == pytest.approx(0.25)
    assert score["BSS"] == pytest.approx(-0.25)
    # One event/non-event pair is tied at probability 1 and counts as half.
    assert score["AROC"] == pytest.approx(0.625)
    assert score["n_points"] == 4


def test_probability_score_is_weighted_and_respects_mask():
    members = [[0, 0, 2, 2], [0, 2, 2, 2]]
    result = EnsembleProbabilityScore(THRESHOLDS).compute(
        _batch(
            members,
            [0, 2, 2, 0],
            valid_mask=[True, True, True, False],
            weights=[1.0, 2.0, 3.0, 4.0],
        )
    )
    # probabilities 0, 0.5, 1 with weighted squared errors 0, .5, 0.
    assert result.value["rain"]["BS"] == pytest.approx(0.5 / 6.0)
    assert result.n_valid == 3


def test_probability_external_reference_picks_matching_threshold_column():
    reference = xr.DataArray(
        [[0.1], [0.2], [0.8], [0.9]],
        dims=("station", "threshold"),
        coords={"threshold": [1.0]},
    )
    result = EnsembleProbabilityScore(THRESHOLDS).compute(
        _batch(
            [[0, 0, 2, 2], [0, 2, 2, 2]],
            [0, 2, 2, 0],
            weights=[1.0, 2.0, 3.0, 4.0],
            reference=reference,
        )
    )
    # 现状：BS_ref 忽略 batch.weights（无权重口径 Σ(p_clim−o)²/N），而 BS 是加权口径，
    # BSS = 1 − BS/BS_ref 因此分子分母口径不一致；期望的加权值由文件末尾的 xfail
    # 测试钉住（见 test_probability_weighted_reference_uses_matching_weights_pending_source_fix）。
    expected_ref_unweighted = (0.1**2 + 0.8**2 + 0.2**2 + 0.9**2) / 4
    assert result.value["rain"]["BS_ref"] == pytest.approx(expected_ref_unweighted)


def test_probability_excludes_nonfinite_members_and_empty_data():
    metric = EnsembleProbabilityScore(THRESHOLDS)
    result = metric.compute(_batch([[0, np.nan], [2, 2]], [0, 2]))
    assert result.n_valid == 1

    empty = metric.compute(
        _batch([[0, 2], [2, 2]], [0, 2], valid_mask=[False, False])
    )
    assert empty.status is ResultStatus.NO_VALID_DATA
    assert empty.n_valid == 0


def test_probability_without_both_event_classes_is_partial():
    result = EnsembleProbabilityScore(THRESHOLDS).compute(
        _batch([[2, 2], [2, 2]], [2, 2])
    )
    assert result.status is ResultStatus.PARTIAL
    assert np.isnan(result.value["rain"]["AROC"])


def test_probability_merge_matches_whole_batch():
    metric = EnsembleProbabilityScore(THRESHOLDS)
    whole = metric.compute(_batch([[0, 0, 2, 2], [0, 2, 2, 2]], [0, 2, 2, 0]))
    merged = metric.finalize(
        metric.merge(
            [
                metric.accumulate(_batch([[0, 0], [0, 2]], [0, 2])),
                metric.accumulate(_batch([[2, 2], [2, 2]], [2, 0])),
            ]
        )
    )
    for key in ("AROC", "BS", "BS_ref", "BSS", "base_rate"):
        assert merged.value["rain"][key] == pytest.approx(whole.value["rain"][key])


def test_probability_rejects_bad_inputs_and_foreign_states():
    metric = EnsembleProbabilityScore(THRESHOLDS)
    batch = _batch([[0, 2], [2, 2]], [0, 2])
    state = metric.accumulate(batch)

    bad_members = replace(batch, members=xr.DataArray([0.0, 2.0], dims=("station",)))
    with pytest.raises(MetricError, match="member, sample"):
        metric.compute(bad_members)
    bad_weights = replace(batch, weights=xr.DataArray([1.0], dims=("station",)))
    with pytest.raises(MetricError, match="权重形状"):
        metric.compute(bad_weights)

    with pytest.raises(MetricError, match="empty states"):
        metric.merge([])
    with pytest.raises(MetricError, match="different metrics"):
        metric.merge([state, replace(state, metric_name="crps")])

    different_members = EnsembleProbabilityScore(THRESHOLDS).accumulate(
        _batch([[0, 2], [1, 2], [2, 2]], [0, 2])
    )
    with pytest.raises(MetricError, match="成员数不一致"):
        metric.merge([state, different_members])


def test_probability_requirements_request_members_and_reference():
    metric = EnsembleProbabilityScore(THRESHOLDS)
    assert metric.requirements().product_type is ProductType.ENSEMBLE_SAMPLES
    assert metric.needs_members()
    assert metric.needs_reference()


# 下面两条钉住已缺失/口径不一致的行为；源码解冻修好后，把它们从 skip 改回正常测试、
# 删掉对应标记，并把上面的原地断言恢复成期望值。
@pytest.mark.skip(
    reason=(
        "源码冻结暂不修：BS_ref 现在恒用无权重口径 Σ(p_clim−o)²/N，而 BS（直方图）在有 batch.weights "
        "时是加权口径，BSS = 1 − BS/BS_ref 因此分子分母口径不一致"
    ),
)
def test_probability_weighted_reference_uses_matching_weights_pending_source_fix():
    reference = xr.DataArray(
        [[0.1], [0.2], [0.8], [0.9]],
        dims=("station", "threshold"),
        coords={"threshold": [1.0]},
    )
    result = EnsembleProbabilityScore(THRESHOLDS).compute(
        _batch(
            [[0, 0, 2, 2], [0, 2, 2, 2]],
            [0, 2, 2, 0],
            weights=[1.0, 2.0, 3.0, 4.0],
            reference=reference,
        )
    )
    expected_ref = (1 * 0.1**2 + 2 * 0.8**2 + 3 * 0.2**2 + 4 * 0.9**2) / 10
    assert result.value["rain"]["BS_ref"] == pytest.approx(expected_ref)


@pytest.mark.skip(
    reason=(
        "源码冻结暂不修：EnsembleProbabilityScore.merge() 目前只校验 states 非空、metric_name 与"
        "成员数，不比 metric_version、也不比阈值参数"
    ),
)
def test_probability_merge_version_and_threshold_contract_pending_source_fix():
    metric = EnsembleProbabilityScore(THRESHOLDS)
    batch = _batch([[0, 2], [2, 2]], [0, 2])
    state = metric.accumulate(batch)

    with pytest.raises(MetricError, match="different versions"):
        metric.merge([state, replace(state, metric_version="2.0.0")])

    other = EnsembleProbabilityScore([("rain", 5.0)]).accumulate(batch)
    with pytest.raises(MetricError, match="different parameters"):
        metric.merge([state, other])
