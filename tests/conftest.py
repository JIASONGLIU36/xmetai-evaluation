# -*- coding: utf-8 -*-
"""外层共享 fixture：unit / integration 都从这里取。

工厂本体在 ``tests/runner.py``；本文件只做 fixture 出口，不写逻辑。
"""
import pytest

from tests import runner


@pytest.fixture(scope="session")
def grid():
    """合成网格 ``(lat, lon)``：0.25°，所有设计位置都落在格点上。"""
    return runner.GRID_LAT, runner.GRID_LON


@pytest.fixture()
def make_babj():
    """写 BABJ 报文：``make_babj(path, tcname, tcid, rows)``。

    ``rows``: ``(北京时 datetime, 时效 h, lon, lat, pmin_hpa, vmax_ms)``，
    时效 000 的行才是分析实况。
    """
    return runner.write_babj


@pytest.fixture()
def make_det_forecast():
    """写确定性预报目录：``make_det_forecast(root, init_utc, centers)``。"""
    return runner.write_fuxi_single


@pytest.fixture()
def make_ens_forecast():
    """写集合预报目录：``make_ens_forecast(root, init_utc, member_centers)``。"""
    return runner.write_fuxi_ens
