# -*- coding: utf-8 -*-
"""种子规则与起报反推（protocols.py 纯函数）的单元测试。

表驱动地钉住历史回归：2507@0722 的非网格记录、日期窗、逐号台风判定——
种子是台风链"静默出错"的重灾区。
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

from xmetai_evaluation.io.babj_reader import BabjCatalog, BabjReader
from xmetai_evaluation.pipeline.protocols import (
    babj_init_times,
    babj_seed_moment,
    sample_leads,
    typhoon_storm_filter,
)

INIT_BJT = datetime(2025, 6, 10, 8, 0)  # 起报北京时（init 00 UTC + 8h）


def test_seed_empty_returns_none():
    assert babj_seed_moment(set(), INIT_BJT, 6.0, 6.0) is None


def test_seed_takes_first_on_grid_at_or_after_offset():
    moments = {datetime(2025, 6, 10, 14, 0), datetime(2025, 6, 10, 20, 0)}
    assert babj_seed_moment(moments, INIT_BJT, 6.0, 6.0) == datetime(2025, 6, 10, 14, 0)


def test_seed_skips_off_grid_records():
    # 2507@0722 归档回归：17:00 是非 6h 网格记录，20:00 才是第一条能配上时效的
    moments = {datetime(2025, 6, 10, 17, 0), datetime(2025, 6, 10, 20, 0)}
    assert babj_seed_moment(moments, INIT_BJT, 6.0, 6.0) == datetime(2025, 6, 10, 20, 0)


def test_seed_date_window_boundaries():
    moments = {datetime(2025, 6, 10, 14, 0)}
    # 起报日必须落在台风首末实况的日期窗内
    assert babj_seed_moment(moments, datetime(2025, 6, 9, 8, 0), 6.0, 6.0) is None
    assert babj_seed_moment(moments, datetime(2025, 6, 11, 8, 0), 6.0, 6.0) is None
    # 边界相等：首日/末日当天可以起链
    assert babj_seed_moment(moments, INIT_BJT, 6.0, 6.0) == datetime(2025, 6, 10, 14, 0)


def test_seed_offset_zero_requires_init_moment():
    moment = datetime(2025, 6, 10, 8, 0)
    assert babj_seed_moment({moment}, INIT_BJT, 0.0, 6.0) == moment
    assert babj_seed_moment({datetime(2025, 6, 10, 14, 0)}, INIT_BJT, 0.0, 6.0) is None


def test_seed_honours_step_grid():
    moments = {datetime(2025, 6, 10, 14, 0), datetime(2025, 6, 10, 20, 0)}
    assert babj_seed_moment(moments, INIT_BJT, 6.0, 12.0) == datetime(2025, 6, 10, 20, 0)
    assert babj_seed_moment({datetime(2025, 6, 10, 14, 0)}, INIT_BJT, 6.0, 12.0) is None


def _storm_rows(start: datetime, count: int, lat0: float):
    return [
        (start + timedelta(hours=6 * k), 0, 121.0, lat0 + 0.1 * k, 960, 25)
        for k in range(count)
    ]


def test_babj_init_times_per_storm_and_filter(tmp_path, make_babj):
    babj_dir = tmp_path / "babj"
    make_babj(
        babj_dir / "babj2501.dat", "WUTIP", "2501",
        _storm_rows(datetime(2025, 6, 10, 8, 0), 10, 18.0),  # 06-10 08:00 – 06-12 14:00
    )
    make_babj(
        babj_dir / "babj2502.dat", "MUN", "2502",
        _storm_rows(datetime(2025, 6, 11, 8, 0), 3, 16.0),    # 06-11 08:00 – 06-11 20:00
    )
    obs = SimpleNamespace(
        source_id="babj",
        reader=BabjReader(source_id="babj"),
        catalog=BabjCatalog(babj_dir),
    )
    inits = [
        datetime(2025, 6, 10, 0, 0),  # 只在 2501 的日期窗内
        datetime(2025, 6, 11, 0, 0),  # 两号都在
        datetime(2025, 6, 13, 0, 0),  # 窗口外
        datetime(2025, 6, 30, 0, 0),  # 空档期：不得放行
    ]
    assert babj_init_times(obs, inits, 8.0, None, 6.0, 6.0) == inits[:2]
    assert babj_init_times(obs, inits, 8.0, ["2502"], 6.0, 6.0) == [inits[1]]
    assert babj_init_times(obs, inits, 8.0, ["2501"], 6.0, 6.0) == inits[:2]


def test_sample_leads_keeps_window_endpoints():
    assert sample_leads([0.0, 6.0, 12.0, 18.0, 24.0], 6) == [6.0, 12.0, 18.0, 24.0]
    assert sample_leads([6.0, 12.0, 24.0, 36.0, 48.0], 24) == [24.0, 48.0]


def test_storm_filter_variants():
    assert typhoon_storm_filter({}) is None
    assert typhoon_storm_filter({"storm_ids": ["2501", "2502"]}) == ["2501", "2502"]
    assert typhoon_storm_filter({"tcid": 2501}) == ["2501"]
    assert typhoon_storm_filter({"storm_ids": []}) is None
