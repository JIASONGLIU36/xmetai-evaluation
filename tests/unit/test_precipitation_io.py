# -*- coding: utf-8 -*-
"""io/station_reader.py：Diamond 站点文件的发现 / 时间解析 / 拼接（合成数据）。

对标台风套件的 ``test_typhoon_io_writer.py``（那一层测 BABJ diamond 7 的
catalog + reader）。IO 层的错都是"静默错"：目录被漏扫、站点被平移一格、
只在某一时次出现的站丢了经纬度被区域筛选误剔除。这里用一个
**2 时次 × 3 站**的小矩阵把这些全部手算钉死。
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from xmetai_evaluation.core.contracts import DataIndex, DataKind, DataRequest
from xmetai_evaluation.core.errors import DecodeError, DiscoveryError
from xmetai_evaluation.io.station_reader import (
    DiamondStationCatalog,
    DiamondStationReader,
    load_station_whitelist,
    parse_station_file_time,
)

T0 = datetime(2025, 8, 20, 0, 0)
T1 = datetime(2025, 8, 20, 1, 0)

#: 两个时次故意写成不同的站、且站号顺序乱放（读取端必须自己按站号排序）
EARLY = [(50007, 113.0, 22.0, 10.0, 1.0), (50005, 112.0, 21.0, 5.0, 0.0)]
LATE = [(50005, 112.0, 21.0, 5.0, 2.5), (50006, 114.0, 23.0, 7.0, 3.0)]


def _request():
    return DataRequest(source_id="diamond_station", variables=["precipitation"])


def _index(*paths):
    return DataIndex(source_id="diamond_station", available=list(paths), ambiguous=[])


# ---------------------------------------------------------------- 目录发现

def test_catalog_discovers_layered_layout_in_time_order(tmp_path, make_station_obs, station_file):
    root = tmp_path / "station"
    make_station_obs(station_file(root, T0), T0, EARLY)
    make_station_obs(station_file(root, T1), T1, LATE)

    index = DiamondStationCatalog(root).discover(_request())
    assert [path.name for path in index.available] == ["00.000", "01.000"]
    assert [path.parent.name for path in index.available] == ["20250820", "20250820"]
    assert index.ambiguous == []
    assert index.source_id == "diamond_station"


def test_catalog_discovers_flat_layout(tmp_path, make_station_obs, flat_station_file):
    flat = tmp_path / "flat"
    make_station_obs(flat_station_file(flat, T0), T0, EARLY)
    make_station_obs(flat_station_file(flat, T1), T1, LATE)

    index = DiamondStationCatalog(flat).discover(_request())
    assert [path.name for path in index.available] == ["2025082000.000", "2025082001.000"]


def test_catalog_errors(tmp_path):
    with pytest.raises(DiscoveryError):
        DiamondStationCatalog(tmp_path / "missing").discover(_request())

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DiscoveryError):
        DiamondStationCatalog(empty).discover(_request())

    # 有日期目录但没有 .000 文件
    dated = tmp_path / "dated" / "20250820"
    dated.mkdir(parents=True)
    with pytest.raises(DiscoveryError):
        DiamondStationCatalog(tmp_path / "dated").discover(_request())


# ---------------------------------------------------------------- 拼接

def test_reader_builds_dense_time_station_matrix(tmp_path, make_station_obs, station_file):
    root = tmp_path / "station"
    make_station_obs(station_file(root, T0), T0, EARLY)
    make_station_obs(station_file(root, T1), T1, LATE)
    catalog = DiamondStationCatalog(root)

    bundle = DiamondStationReader().read(_request(), catalog.discover(_request()))
    dataset = bundle.payload

    assert bundle.kind is DataKind.STATION_OBSERVATION
    assert bundle.standard_vars == {"precipitation": "tp"}
    assert bundle.semantic.units == {"precipitation": "mm"}
    assert bundle.provenance.reader_id == "diamond_station"
    assert len(bundle.provenance.input_files) == 2

    assert set(dataset.dims) == {"time", "station"}
    np.testing.assert_array_equal(
        dataset["time"].values, np.asarray([T0, T1], dtype="datetime64[ns]")
    )
    # 站号取两个时次的并集、升序排列
    assert dataset["station"].values.tolist() == [50005, 50006, 50007]
    assert dataset["station_id"].values.tolist() == [50005, 50006, 50007]
    # 只在 01:00 出现的 50006 也要有经纬度（取首次出现的时次）
    np.testing.assert_allclose(dataset["lat"].values, [21.0, 23.0, 22.0])
    np.testing.assert_allclose(dataset["lon"].values, [112.0, 114.0, 113.0])
    np.testing.assert_allclose(dataset["altitude"].values, [5.0, 7.0, 10.0])
    # 该站没报的时次留 NaN，不是 0
    np.testing.assert_allclose(
        dataset["precipitation"].values,
        [[0.0, np.nan, 1.0], [2.5, 3.0, np.nan]],
        equal_nan=True,
    )
    assert dataset["precipitation"].attrs["units"] == "mm"


def test_reader_sorts_unsorted_index(tmp_path, make_station_obs, station_file):
    root = tmp_path / "station"
    make_station_obs(station_file(root, T0), T0, EARLY)
    make_station_obs(station_file(root, T1), T1, LATE)

    bundle = DiamondStationReader().read(
        _request(), _index(station_file(root, T1), station_file(root, T0))
    )
    np.testing.assert_array_equal(
        bundle.payload["time"].values, np.asarray([T0, T1], dtype="datetime64[ns]")
    )


def test_reader_applies_station_whitelist(tmp_path, make_station_obs, station_file):
    root = tmp_path / "station"
    make_station_obs(station_file(root, T0), T0, EARLY)
    make_station_obs(station_file(root, T1), T1, LATE)
    catalog = DiamondStationCatalog(root)

    bundle = DiamondStationReader(station_whitelist=[50007, 50005]).read(
        _request(), catalog.discover(_request())
    )
    assert bundle.payload["station"].values.tolist() == [50005, 50007]

    # 白名单一个都命中 -> 全部文件无效
    with pytest.raises(DecodeError, match="No valid station data parsed"):
        DiamondStationReader(station_whitelist=[99999]).read(
            _request(), catalog.discover(_request())
        )


def test_reader_rejects_empty_index_and_unparsable_files(tmp_path):
    with pytest.raises(DecodeError, match="No files in index"):
        DiamondStationReader().read(_request(), _index())

    junk = tmp_path / "junk.000"
    junk.write_bytes("nope\n".encode("gbk"))
    with pytest.raises(DecodeError, match="No valid station data parsed"):
        DiamondStationReader().read(_request(), _index(junk))


# ---------------------------------------------------------------- 文件名与白名单

def test_parse_station_file_time_both_layouts_and_error(station_file, flat_station_file):
    assert parse_station_file_time(station_file("root", T1)) == T1
    assert parse_station_file_time(flat_station_file("root", T1)) == T1
    with pytest.raises(ValueError, match="无法解析观测文件时间"):
        parse_station_file_time(Path("root") / "junk.000")


def test_load_station_whitelist_sources_and_errors(tmp_path):
    assert load_station_whitelist(None) is None
    # 空列表原样返回（falsy）：下游 `set(...) if whitelist else None` 会当成"不限站"
    assert load_station_whitelist([]) == []
    # 直接给列表：按给定顺序返回（下游存成 set，顺序不影响筛选）
    assert load_station_whitelist([50007, 50005]) == [50007, 50005]
    assert load_station_whitelist({"50005"}) == [50005]

    listing = tmp_path / "stations.txt"
    listing.write_text("50007 首个字段是站号\n\n50005\n", encoding="utf-8")
    assert load_station_whitelist(str(listing)) == [50005, 50007]
    assert load_station_whitelist(listing) == [50005, 50007]  # Path 也认

    with pytest.raises(ValueError, match="不存在"):
        load_station_whitelist(tmp_path / "missing.txt")

    blank = tmp_path / "blank.txt"
    blank.write_text("站点 名单\n", encoding="utf-8")
    with pytest.raises(ValueError, match="为空或格式无效"):
        load_station_whitelist(blank)
