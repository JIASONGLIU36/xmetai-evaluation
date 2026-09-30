# -*- coding: utf-8 -*-
"""track_error / track_error_ens 指标单元测试（合成批次，不经文件系统）。

设计几何：实况每 6h 北移 0.5°，预报/成员沿正北偏移固定度数——
所有预期值都可以用 ``KM_PER_DEG`` 手算。
"""
import math

import numpy as np
import pytest
import xarray as xr

from tests.runner import KM_PER_DEG
from xmetai_evaluation.core.contracts import EvaluationBatch, ResultStatus
from xmetai_evaluation.core.errors import MetricError
from xmetai_evaluation.metrics.base import MetricState
from xmetai_evaluation.metrics.track_error import (
    CURVE_FIELDS,
    ENS_EXTRA_FIELDS,
    ENS_SUMMARY_FIELDS,
    SUMMARY_FIELDS,
    TrackError,
    TrackErrorEns,
)

INIT_ISO = "2025-06-10T00:00:00"
LEADS = [6.0, 12.0, 18.0]


def _dataset(fields, leads, *, members=None, storm="2501"):
    dims = ("lead_time", "storm", "member") if members is not None else ("lead_time", "storm")
    coords = {"lead_time": list(leads), "storm": [storm]}
    if members is not None:
        coords["member"] = list(members)
    data = {name: (dims, np.asarray(values, dtype="f8")) for name, values in fields.items()}
    return xr.Dataset(data, coords=coords)


def _batch(forecast, observation, *, init_obs=None, init_pos=(20.0, 121.0), seed_offset=6.0):
    storms = [str(value) for value in np.asarray(forecast["storm"].values)]
    leads = [float(value) for value in np.asarray(forecast["lead_time"].values)]
    valid = xr.DataArray(
        np.isfinite(np.asarray(observation["lat"].values, dtype="f8")),
        dims=("lead_time", "storm"),
        coords={"lead_time": leads, "storm": storms},
        name="valid_mask",
    )
    alignment = {
        "method": "storm_track",
        "tcname": "WUTIP",
        "tz_shift_hours": 8.0,
        "init_bjt": "2025-06-10 08:00:00",
        "init_position": list(init_pos),
        "seed_offset_hours": seed_offset,
        "search": {"center_half_deg": 3.0},
    }
    if init_obs is not None:
        alignment["init_obs_position"] = list(init_obs)
    return EvaluationBatch(
        forecast=forecast,
        observation=observation,
        sample_keys=[{"init": INIT_ISO, "storm": storms[0]}],
        valid_mask=valid,
        sample_dim="storm",
        alignment=alignment,
        protocol_id="typhoon_track",
    )


def _det_case(*, obs_lat=None, fcst_offset=0.25, leads=LEADS):
    obs_lat = list(obs_lat if obs_lat is not None else [20.5, 21.0, 21.5])
    n = len(leads)
    observation = _dataset(
        {
            "lat": np.array(obs_lat)[:, None],
            "lon": np.full((n, 1), 121.0),
            "pmin": np.full((n, 1), 960.0),
            "vmax": np.full((n, 1), 25.0),
        },
        leads,
    )
    forecast = _dataset(
        {
            "lat": (np.array(obs_lat) + fcst_offset)[:, None],
            "lon": np.full((n, 1), 121.0),
            "pmin": np.full((n, 1), 950.0),
            "vmax": np.full((n, 1), 30.0),
        },
        leads,
    )
    return forecast, observation


def _ens_case(
    *,
    offsets=(0.25, -0.75),
    leads=(6.0, 12.0),
    obs_start=20.0,
    step=0.5,
    nan_lead=None,
    nan_member=None,
):
    obs_lat = [obs_start + step * (k + 1) for k in range(len(leads))]
    n = len(leads)
    members = [f"m{i}" for i in range(len(offsets))]
    f_lat = np.array([[lat + off for off in offsets] for lat in obs_lat])
    forecast = _dataset(
        {
            "lat": f_lat[:, None, :],
            "lon": np.full((n, 1, len(members)), 121.0),
            "pmin": np.full((n, 1, len(members)), 950.0),
            "vmax": np.full((n, 1, len(members)), 30.0),
        },
        leads,
        members=members,
    )
    if nan_lead is not None and nan_member is not None:
        forecast["lat"].values[nan_lead, 0, nan_member] = np.nan
    observation = _dataset(
        {
            "lat": np.array(obs_lat)[:, None],
            "lon": np.full((n, 1), 121.0),
            "pmin": np.full((n, 1), 960.0),
            "vmax": np.full((n, 1), 25.0),
        },
        leads,
    )
    return forecast, observation


# ---------------------------------------------------------------- TrackError

def test_track_error_values_and_contract():
    forecast, observation = _det_case()
    metric = TrackError()
    batch = _batch(forecast, observation)
    metric.validate(batch)
    result = metric.finalize(metric.accumulate(batch))
    curve = result.curve
    for field in CURVE_FIELDS:
        assert field in curve
    assert set(SUMMARY_FIELDS) <= set(result.value)
    assert curve["track_err_km"] == pytest.approx([0.25 * KM_PER_DEG] * 3, rel=1e-9)
    # 首个配对时次 at/ct 为空；其后按实况路径段分解
    assert math.isnan(curve["at_km"][0]) and math.isnan(curve["ct_km"][0])
    assert curve["at_km"][1:] == pytest.approx([0.25 * KM_PER_DEG] * 2, rel=1e-9)
    assert curve["ct_km"][1:] == pytest.approx([0.0] * 2, abs=1e-9)
    assert curve["wind_err_ms"] == pytest.approx([5.0] * 3)
    assert curve["pmin_err_hpa"] == pytest.approx([-10.0] * 3)
    assert curve["valid_bjt"] == [
        "2025-06-10 14:00:00",
        "2025-06-10 20:00:00",
        "2025-06-11 02:00:00",
    ]
    assert result.status is ResultStatus.SUCCESS
    assert result.n_requested == 3 and result.n_valid == 3
    assert result.value["track_err_km"] == pytest.approx(0.25 * KM_PER_DEG, rel=1e-9)
    assert result.unit == "km"


def test_track_error_no_valid_data():
    forecast, observation = _det_case()
    observation["lat"].values[:] = np.nan
    metric = TrackError()
    result = metric.finalize(metric.accumulate(_batch(forecast, observation)))
    assert result.status is ResultStatus.NO_VALID_DATA
    assert result.n_valid == 0
    assert result.warnings


def test_track_error_rejects_multi_storm_batch():
    n = len(LEADS)
    fcst_lat = np.tile((np.array([20.5, 21.0, 21.5]) + 0.25)[:, None], (1, 2))
    obs_lat = np.tile(np.array([20.5, 21.0, 21.5])[:, None], (1, 2))
    coords = {"lead_time": LEADS, "storm": ["2501", "2502"]}
    forecast = xr.Dataset(
        {
            "lat": (("lead_time", "storm"), fcst_lat),
            "lon": (("lead_time", "storm"), np.full((n, 2), 121.0)),
            "pmin": (("lead_time", "storm"), np.full((n, 2), 950.0)),
            "vmax": (("lead_time", "storm"), np.full((n, 2), 30.0)),
        },
        coords=coords,
    )
    observation = xr.Dataset(
        {
            "lat": (("lead_time", "storm"), obs_lat),
            "lon": (("lead_time", "storm"), np.full((n, 2), 121.0)),
            "pmin": (("lead_time", "storm"), np.full((n, 2), 960.0)),
            "vmax": (("lead_time", "storm"), np.full((n, 2), 25.0)),
        },
        coords=coords,
    )
    batch = _batch(forecast, observation)
    with pytest.raises(MetricError):
        TrackError().validate(batch)


def _slice_state(state, leads):
    """把一条完整曲线状态按 lead 切成子状态（merge 的输入形态）。"""
    curve = state.data["curve"]
    n = len(curve["lead_h"])
    index = [k for k, lead in enumerate(curve["lead_h"]) if lead in leads]
    skip = {"members", "search"}
    sliced = {
        key: (
            [value[k] for k in index]
            if isinstance(value, list) and key not in skip and len(value) == n
            else value
        )
        for key, value in curve.items()
    }
    sliced["n_leads"] = len(index)
    sliced["n_matched"] = int(np.isfinite(np.asarray(sliced["track_err_km"], dtype="f8")).sum())
    return MetricState(
        metric_name=state.metric_name,
        metric_version=state.metric_version,
        data={"curve": sliced, "n_valid": sliced["n_matched"]},
        n_accumulated=1,
    )


def test_track_error_merge_reorders_and_rejects_duplicates():
    forecast, observation = _det_case()
    metric = TrackError()
    full_state = metric.accumulate(_batch(forecast, observation))
    first = _slice_state(full_state, [6.0, 18.0])
    second = _slice_state(full_state, [12.0])
    merged = metric.merge([first, second])
    got, expected = merged.data["curve"], full_state.data["curve"]
    for field in CURVE_FIELDS:
        left, right = expected[field], got[field]
        if field == "valid_bjt":
            assert left == right
        else:
            assert np.allclose(
                np.asarray(left, dtype=float),
                np.asarray(right, dtype=float),
                rtol=0,
                atol=0,
                equal_nan=True,
            ), field
    assert got["n_leads"] == expected["n_leads"]
    # 重复 lead = 同一条路径被算了两遍 → 必须报错
    with pytest.raises(MetricError):
        metric.merge([first, _slice_state(full_state, [6.0])])
    # 版本不一致 → 报错
    bad = MetricState(
        metric_name="track_error",
        metric_version="9.9.9",
        data={"curve": dict(second.data["curve"])},
        n_accumulated=1,
    )
    with pytest.raises(MetricError):
        metric.merge([first, bad])


# ---------------------------------------------------------------- TrackErrorEns

def test_track_error_ens_scheme_a_and_b():
    forecast, observation = _ens_case()
    metric = TrackErrorEns()
    assert metric.PRODUCT_KIND == "ensemble"
    assert metric.needs_members() is True
    batch = _batch(forecast, observation, init_obs=(20.0, 121.0))
    metric.validate(batch)
    result = metric.finalize(metric.accumulate(batch))
    curve = result.curve
    assert curve["forecast_type"] == "ens"
    assert curve["members"] == ["m0", "m1"]
    # B：成员位置先平均（均值在 obs−0.25°）→ 0.25°；
    # A：成员误差再平均（(0.25+0.75)/2）→ 0.5°；首时次 anchor = 起报时刻实况。
    assert curve["track_err_km"] == pytest.approx([0.25 * KM_PER_DEG] * 2, rel=1e-9)
    assert curve["track_err_km_a"] == pytest.approx([0.5 * KM_PER_DEG] * 2, rel=1e-9)
    assert curve["at_km"] == pytest.approx([-0.25 * KM_PER_DEG] * 2, rel=1e-9)
    assert curve["at_km_a"] == pytest.approx([-0.25 * KM_PER_DEG] * 2, rel=1e-9)
    assert curve["ct_km"] == pytest.approx([0.0] * 2, abs=1e-9)
    assert curve["n_members"] == [2, 2]
    assert curve["n_valid_members"] == [2, 2]
    assert curve["wind_err_ms"] == pytest.approx([5.0] * 2)  # 强度类两方案恒等
    assert set(ENS_SUMMARY_FIELDS) <= set(result.value)


def test_track_error_ens_first_lead_anchor():
    forecast, observation = _ens_case()
    with_anchor = _batch(forecast, observation, init_obs=(20.0, 121.0))
    curve = TrackErrorEns().accumulate(with_anchor).data["curve"]
    assert math.isfinite(curve["at_km"][0]) and math.isfinite(curve["ct_km"][0])
    # 没有起报时刻实况：首时次 at/ct 留空；第二时效用上一实况定向
    without = _batch(forecast, observation)
    curve = TrackErrorEns().accumulate(without).data["curve"]
    assert math.isnan(curve["at_km"][0]) and math.isnan(curve["ct_km"][0])
    assert math.isfinite(curve["at_km"][1])


def test_track_error_ens_partial_members():
    forecast, observation = _ens_case(leads=(6.0,), nan_lead=0, nan_member=1)
    curve = TrackErrorEns().accumulate(
        _batch(forecast, observation, init_obs=(20.0, 121.0))
    ).data["curve"]
    assert curve["n_members"] == [2]
    assert curve["n_valid_members"] == [1]
    # 平均只按能诊断出中心的 m0（obs+0.25°），不整片变 NaN
    assert curve["track_err_km"] == pytest.approx([0.25 * KM_PER_DEG], rel=1e-9)
    assert curve["track_err_km_a"] == pytest.approx([0.25 * KM_PER_DEG], rel=1e-9)


def test_metric_product_kind_guards():
    det_forecast, det_observation = _det_case()
    det_batch = _batch(det_forecast, det_observation)
    ens_forecast, ens_observation = _ens_case()
    ens_batch = _batch(ens_forecast, ens_observation, init_obs=(20.0, 121.0))
    # 确定性指标收到成员场 → 被拦
    with pytest.raises(MetricError):
        TrackError().validate(ens_batch)
    # 集合指标没有成员轴 → 被拦
    with pytest.raises(MetricError):
        TrackErrorEns().validate(det_batch)


def test_track_error_ens_merge_keeps_extra_fields():
    forecast, observation = _ens_case()
    metric = TrackErrorEns()
    full_state = metric.accumulate(_batch(forecast, observation, init_obs=(20.0, 121.0)))
    merged = metric.merge(
        [_slice_state(full_state, [6.0]), _slice_state(full_state, [12.0])]
    )
    got, expected = merged.data["curve"], full_state.data["curve"]
    for field in CURVE_FIELDS + ENS_EXTRA_FIELDS:
        if field == "valid_bjt":
            assert got[field] == expected[field]
        else:
            assert np.allclose(
                np.asarray(got[field], dtype=float),
                np.asarray(expected[field], dtype=float),
                rtol=0,
                atol=0,
                equal_nan=True,
            ), field
