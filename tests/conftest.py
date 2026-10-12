# -*- coding: utf-8 -*-
"""Shared fixture exports for the unit / integration suites.

工厂本体按产品线分文件（``tests/runner.py`` 台风、``tests/runner_precipitation.py``
站点降水、``tests/runner_weather_field.py`` 天气场），这里只把它们
暴露成 fixture，避免各测试文件各写一份本地 helper。
"""
import pytest

from tests import runner as typhoon_runner
from tests import runner_precipitation as precipitation_runner


@pytest.fixture(scope="session")
def grid():
    """合成网格 ``(lat, lon)``：0.25°，所有设计位置都落在格点上。"""
    return typhoon_runner.GRID_LAT, typhoon_runner.GRID_LON


@pytest.fixture()
def make_babj():
    """写 BABJ 报文：``make_babj(path, tcname, tcid, rows)``。

    ``rows``: ``(北京时 datetime, 时效 h, lon, lat, pmin_hpa, vmax_ms)``，
    时效 000 的行才是分析实况。
    """
    return typhoon_runner.write_babj


@pytest.fixture()
def make_det_forecast():
    """写确定性预报目录：``make_det_forecast(root, init_utc, centers)``。"""
    return typhoon_runner.write_fuxi_single


@pytest.fixture()
def make_ens_forecast():
    """写集合预报目录：``make_ens_forecast(root, init_utc, member_centers)``。"""
    return typhoon_runner.write_fuxi_ens


@pytest.fixture()
def make_station_obs():
    """写 Diamond 站点文件：``make_station_obs(path, moment, rows)``。

    ``rows``: ``(站号, 经度, 纬度, 海拔 m, 降水 mm)``；路径用 ``station_file``
    取分层布局（``root/YYYYMMDD/HH.000``）、``flat_station_file`` 取扁平布局。
    """
    return precipitation_runner.write_diamond


@pytest.fixture()
def station_file():
    """分层布局路径：``station_file(root, moment) -> root/YYYYMMDD/HH.000``。"""
    return precipitation_runner.station_file


@pytest.fixture()
def flat_station_file():
    """扁平布局路径：``flat_station_file(root, moment) -> root/YYYYMMDDHH.000``。"""
    return precipitation_runner.flat_station_file
