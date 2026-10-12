# -*- coding: utf-8 -*-
"""FDP 降水站点检验（Fengqing × Diamond 站点）真实数据档：与标准脚本产物对表。

跑两条真配置，产物与 ``D:\\xmetai-evalation\\fdp\\`` 下两个标准脚本的输出比：

- ``fdp_precip_ens_fengqing``    ↔ ``ensemble_verifier.py``         → AROC / BSS
- ``fdp_precip_single_fengqing`` ↔ ``tp_deterministic_verifier.py`` → 列联表 hits/fa/misses

标准答案是**内联**在下面的，没做任何四舍五入。ens 那份抄自标准脚本落下的
``ensemble_verification_2026082000.csv``；single 那份用的是脚本打在控制台上的整数
列联表（理由见 ``GOLDEN_COUNTS``）。

fixture 在 ``DATA_ROOT``（见下）：``fengqing/``（起报 20260820、时效 6/12/18）
+ ``rain/``（0820 的 00Z~06Z 七个 Diamond 文件）。目录不齐（或没装 netCDF4）时整个
模块 skip；平时用 ``pytest -m realdata`` 选中才跑。

⚠ 两条**已知差**，都已查到根因，不是口径问题，别当 bug 去查：

1. **≥13 阈值上 1 个站翻转**。标准脚本累 6 小时观测走 ``pd.concat + groupby.sum``，
   站 758426 得**恰好 13.0**；框架（和 numpy 的 ``np.sum``）得 12.999999999999998。
   该站预报 22.23，于是标准判命中、框架判空报。这类最后一位浮点差在 88213 个站里有
   1700 个，只有这一个恰好压在阈值上。
2. **≥0.1 阈值上 1 个站翻转**。站号 **999999** 是一条哨兵行：7 个时次文件里坐标
   **每个时次都不同**（00Z 在 116.22/38.09，06Z 在 107.09/32.51），降水恒为 0。
   标准脚本取窗口末端（06Z）的坐标 → 插出 3.96 ≥ 0.1 → 空报；框架取读取集并集里
   首次出现（01Z）的坐标 → 0.02 < 0.1 → 正确否定。坐标不稳的站共 23 个，只有这个
   翻转了判定。

所以 single 那份的 golden 用**整数列联表**、``COUNT_TOL = 1`` 逐格留 1 个站的余量。
标准 CSV 里的 TS/Bias 只印 4 位小数（``TS_25.0mm = 0.0057``，相对精度才 1e-2），拿
它对表还不如整数计数精确，也拦不住真回归——真错了是成百上千个站地偏。

3. **ens 的 AROC / BSS 只能对到 ~1e-3，对不到 1e-4**。标准脚本累 6 小时观测走
   ``pd.concat(...).groupby('station_id').sum()``（float64），框架走 numpy 求和；
   两者对 ~1700 个站差在最后一位。要命的是这几个站的 6h 合计**正好压在阈值上**：
   ≥4mm 有 2 个站（pandas 得 4.0、numpy 得 3.9999999999999996，差 4.4e-16），
   ≥13mm 有 1 个站（13.0 对 12.999999999999998）。阈值是阶跃，这几个站就翻了干/湿。
   ``BSS = 1 − BS/(r(1−r))`` 再把这抖动放大 ``1/(r(1−r))`` 倍（≥13mm 是 75 倍），
   于是 ≥4mm 相对差 6.0e-4、≥13mm 9.4e-4。
   这条**不是口径差**：把标准脚本原样在本机复现（pandas 核 + 取窗口末端那批站点和
   坐标），四档 BSS/AROC 与它的 CSV **逐位相同**（Δ = 0.000e+00）—— 站点集(88137)、
   cos|lat| 权重、中国框、``>=`` 判据、阈值表、集合概率、站点插值全都一致，差的只有
   浮点求和次序。真想要逐位相同，就得把框架的观测累加也改成 pandas 那种求和次序，
   那是拿实现细节当口径，不值当。容差见 ``ENS_REL_TOL``。

   顺带否掉两条**不是**原因的：观测累加用 float32 会更糟（≥4mm 翻 9 个站，不是 2 个），
   框架实测是 float64（≥13mm 与 numpy/f64 复现只差 3e-9，与 float32 差 1.4e-5）；
   站点坐标取「窗口末端」还是「读取集首见」在这份数据上对 BSS 的影响 <3e-5，量级上与
   上面那 6e-4 差一个数量级，不是主因。
"""
from pathlib import Path

import pandas as pd
import pytest

from tests import runner

pytestmark = pytest.mark.realdata
pytest.importorskip("netCDF4")

#: 最小 fixture 根：预报 fengqing/ + 站点实况 rain/ 两件套在它下面。换机器/换位置
#: 改这一行（配置里的 ``_DATA_ROOT`` 指的是同一处，两份配置同根）。
DATA_ROOT = Path("/workspace/data/shenzw/fdp_test_data")

#: 起报时刻与时效——两份配置里都写死的，golden 也是照这个口径出的。
#: 只有时效 6 出得了分（``rain/`` 只到 0820 06Z），详见两份配置的模块 docstring。
LEAD = 6

#: 对表容差。**不能收到 1e-4** —— 见模块 docstring 第 3 条：观测累加核（pandas
#: groupby vs numpy）在 ≥4mm 翻 2 个站、≥13mm 翻 1 个站，BSS 又把这抖动放大
#: 1/(r(1−r)) 倍。实测四档相对差（AROC / BSS）：
#:   ≥0.1mm  1.1e-5 / 2.6e-5      ≥4mm   5.5e-5 / 6.0e-4
#:   ≥13mm   2.5e-4 / 9.4e-4      ≥25mm  1.9e-7 / 1.0e-9
#: 取 2e-3，对最差那档留约 2 倍余量，仍只拦真回归（站数掉了、单位错了、插值挂了
#: 都是百分之几到几十的量级；口径漂移另有 ``test_ens_coverage`` 的 1e-9 自洽哨）。
ENS_REL_TOL = 2e-3

#: single 的列联表逐格容差：允许差 1 个站（模块 docstring 里那两处已知差）。
COUNT_TOL = 1

#: 阈值。ens 是集合概率口径的四个量级，single 是确定性口径的三个（标准脚本
#: ``PRECIP_THRESHOLDS_6H``），两者不同源，别合并。
ENS_THRESHOLDS = (0.1, 4.0, 13.0, 25.0)
SINGLE_THRESHOLDS = (0.1, 13.0, 25.0)

# --------------------------------------------------------------------------
# 标准答案（内联；来源见模块 docstring）
# --------------------------------------------------------------------------

#: ``ensemble_verifier.py`` 的 ``ensemble_verification_2026082000.csv``，站数 88137。
GOLDEN_ENS_STATIONS = 88137
GOLDEN_ENS_AROC = {
    0.1: 0.6245680078738955,
    4.0: 0.6657321108320154,
    13.0: 0.6475079796338439,
    25.0: 0.62575693567997,
}
GOLDEN_ENS_BSS = {
    0.1: -2.868622175693045,
    4.0: -12.237714889070345,
    13.0: -29.956449078745194,
    25.0: -26.538095365402437,
}

#: ``tp_deterministic_verifier.py`` 的 ``tp_deterministic_6h_2026082000.csv`` —— 但它
#: 印到 4 位小数，当 golden 太糙。改用脚本 ``verify_6h`` 打在控制台上的整数列联表
#: （``hits`` / ``fa`` / ``misses``）：全精度，且能直接对上框架的
#: ``diagnostics/categorical_wide.csv``。
#:
#: ⚠ 这份 golden 是**喂了桥接文件**跑出来的：标准脚本读不了集合文件——它按
#: ``df_tag='FCST'`` 的确定性命名找文件，宽松 glob ``*Fengqing*SURFACE*6HOR*FCST*``
#: 会子串命中 ``..._ENS_FCST_...``，而它又不降维 member，于是
#: ``read_model_tp`` 返回 (21, 721, 1440)、``fc_values`` 成 (21, 88137) 而观测是
#: 一维，直接在 ``obs_values[mask]`` 上 IndexError。所以先把集合文件按
#: xarray ``.mean(dim='member')``（与本配置模板里的 ``ensemble_mean`` 变换同一口径、
#: 同为不加权）降成均值、按标准命名落盘再喂进去。
GOLDEN_SINGLE_STATIONS = 88137
GOLDEN_COUNTS = {
    0.1: {"hits": 15041, "false_alarms": 49766, "misses": 834},
    13.0: {"hits": 750, "false_alarms": 34880, "misses": 357},
    25.0: {"hits": 81, "false_alarms": 13732, "misses": 323},
}

_missing = [sub for sub in ("fengqing", "rain") if not (DATA_ROOT / sub).is_dir()]
if _missing:
    pytest.skip(
        f"fixture 不存在: {DATA_ROOT}（缺 {', '.join(_missing)}/）", allow_module_level=True
    )


# --------------------------------------------------------------------------
# 跑链
# --------------------------------------------------------------------------

def _run(config_name: str, output_dir: Path) -> Path:
    """按真实配置跑一条链，只覆盖产物目录与并发形态。

    数据根**不覆盖**——配置里的 ``_DATA_ROOT`` 指的就是这份 fixture。
    并发写成 serial + ``resume: False``：测试必须每次从零算，否则会捡到上一次落在
    ``.states/segNN_*.pkl`` 的状态，而那份状态**不校验配置指纹**，改了指标口径也会
    被静默复用。
    """
    from xmetai_evaluation.cli import run_evaluation
    from xmetai_evaluation.configs.base import load_config

    cfg = load_config(config_name)
    if isinstance(cfg, list):
        # 确定性那份配置模块导出的是 ``cfgs``（站点段 + 默认注释掉的 FSS 段），
        # 按 name 挑出要跑的那条。
        picked = [item for item in cfg if item.name == config_name]
        assert len(picked) == 1, (
            f"{config_name} 里没有唯一同名的 cfg，实有 {[item.name for item in cfg]}"
        )
        cfg = picked[0]
    cfg.output_dir = str(output_dir)
    cfg.execution = {**cfg.execution, "mode": "serial", "n_workers": 1, "resume": False}
    run_evaluation(cfg)
    return output_dir


@pytest.fixture(scope="session")
def ens(tmp_path_factory):
    """ens 配置跑一次（``weather_ts_ens_prob``：逐成员 → AROC / BS / BSS）。"""
    root = _run("fdp_precip_ens_fengqing", tmp_path_factory.mktemp("ens"))
    return root, runner.read_scores(root)


@pytest.fixture(scope="session")
def single(tmp_path_factory):
    """single 配置跑一次（``fdp_precip_ts``：集合平均 → 插值到站 → TS / 偏倚）。"""
    root = _run("fdp_precip_single_fengqing", tmp_path_factory.mktemp("single"))
    return root, runner.read_scores(root)


# --------------------------------------------------------------------------
# 取数与比对
# --------------------------------------------------------------------------

def _probability_wide(out_dir: Path) -> pd.DataFrame:
    """``diagnostics/probability_wide.csv``——ens 的主表。"""
    return pd.read_csv(out_dir / "diagnostics" / "probability_wide.csv")


def _categorical_wide(out_dir: Path) -> pd.DataFrame:
    """``diagnostics/categorical_wide.csv``——single 的主表，带整数列联表。"""
    return pd.read_csv(out_dir / "diagnostics" / "categorical_wide.csv")


def _rel(actual: float, expected: float, label: str, tol: float = ENS_REL_TOL) -> None:
    """相对差断言；``expected`` 是标准脚本的数。"""
    relative = abs(actual - expected) / abs(expected)
    assert relative <= tol, (
        f"{label}: 框架 {actual!r} vs 标准脚本 {expected!r}，相对差 {relative:.3e}"
    )


def _assert_station_count(scores: pd.DataFrame, expected: int, label: str) -> None:
    """站数逐行对齐标准脚本。

    ens 那条特别值得钉住：它的模板 ``weather_ts_ens_prob`` **没带**
    ``observation_window``，缺省是 ``complete``（6h 窗口缺任一时次就整站丢掉），
    会少 98 个站（88039）。配置里补的那行 ``"observation_window": "reference"``
    就是为它。删了这行，站数掉回 88039，这里和下面的对表会一起红。
    """
    counts = set(scores["n_valid"])
    assert counts == {expected}, (
        f"{label} 站数 {sorted(counts)} != 标准脚本的 {expected}；"
        "ens 那份多半是 observation_window 掉了（见本函数 docstring）"
    )


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------

def test_ens_probability_parity(ens):
    """ens：AROC / BSS 逐阈值对表（标准脚本的 CSV 只出这两族，BS 它没给）。"""
    _, scores = ens
    wide = _probability_wide(ens[0])
    assert len(wide) == len(ENS_THRESHOLDS), f"主表行数 {len(wide)} != {len(ENS_THRESHOLDS)}"
    _assert_station_count(scores, GOLDEN_ENS_STATIONS, "ens")

    for thr in ENS_THRESHOLDS:
        rows = wide[wide["threshold_mm"] == thr]
        assert len(rows) == 1, f"主表里 ≥{thr}mm 有 {len(rows)} 行"
        _rel(float(rows["AROC"].iloc[0]), GOLDEN_ENS_AROC[thr], f"AROC @ ≥{thr}mm")
        _rel(float(rows["BSS"].iloc[0]), GOLDEN_ENS_BSS[thr], f"BSS @ ≥{thr}mm")


def test_ens_coverage(ens):
    """兜底：长表覆盖、BSS 基准口径、以及标准脚本没给的 BS 列完整。

    ``BS_ref == base_rate·(1−base_rate)`` 这条是**口径哨**：配置里的
    ``reference_reader`` 现在被注释掉了，BSS 走样本自己的气候频率，与标准脚本
    ``ensemble_verifier.py`` 的 ``compute_bss`` 同基准。哪天把外部参考接回来，
    BS_ref 会换成 ``mean((p_clim − o)²)``，两边 BSS 就不再可比——这条会先红。
    """
    out, scores = ens
    assert set(scores["metric"]) == {"aroc", "bs", "bss"}
    assert len(scores) == len(ENS_THRESHOLDS) * 3
    assert set(scores["variable"]) == {"precipitation"}
    assert set(scores["sample_unit"]) == {"station"}
    assert set(scores["region"]) == {"china"}, "站点口径必须带中国框，别退化成全球"
    assert set(scores["lead_h"]) == {float(LEAD)}
    assert (scores["status"] == "success").all()
    assert set(scores["threshold"]) == set(ENS_THRESHOLDS)

    wide = _probability_wide(out)
    assert set(wide["n_points"]) == {GOLDEN_ENS_STATIONS}
    assert set(wide["region"]) == {"china"}
    for _, row in wide.iterrows():
        rate = float(row["base_rate"])
        assert 0.0 < rate < 1.0, f"气候频率 {rate} 不在 (0,1)"
        _rel(float(row["BS_ref"]), rate * (1.0 - rate), f"≥{row['threshold_mm']}mm 的 BS_ref")
        # BSS = 1 − BS/BS_ref，三列必须自洽（对不上就是 writer 取数分叉）
        _rel(
            float(row["BSS"]),
            1.0 - float(row["BS"]) / float(row["BS_ref"]),
            f"≥{row['threshold_mm']}mm 的 BSS（对 BS/BS_ref）",
            tol=1e-9,
        )


def test_single_contingency_parity(single):
    """single：整数列联表逐格对表，每格留 1 个站的余量（见模块 docstring）。"""
    _, scores = single
    wide = _categorical_wide(single[0])
    assert len(wide) == len(SINGLE_THRESHOLDS)
    _assert_station_count(scores, GOLDEN_SINGLE_STATIONS, "single")

    for thr in SINGLE_THRESHOLDS:
        rows = wide[wide["threshold_mm"] == thr]
        assert len(rows) == 1, f"主表里 ≥{thr}mm 有 {len(rows)} 行"
        for column, expected in GOLDEN_COUNTS[thr].items():
            actual = int(rows[column].iloc[0])
            assert abs(actual - expected) <= COUNT_TOL, (
                f"≥{thr}mm 的 {column}: 框架 {actual} vs 标准脚本 {expected}"
                f"（差 {actual - expected}，容差 {COUNT_TOL}）"
            )


def test_single_consistency(single):
    """single：长表里的五个指标必须能由整数列联表算回来。

    ``ts`` / ``pod`` / ``far`` / ``miss_rate`` / ``frequency_bias`` 都是列联表的派生量，
    对不上就是两个 writer 的取数口径分叉了。顺带把站数钉住——它和 ens 那份必须一致
    （同一批站点、同一个中国框），模板 ``fdp_precip_ts`` 里本来就带着
    ``observation_window: reference``，所以这边不用改配置。
    """
    out, scores = single
    wide = _categorical_wide(out)
    assert set(scores["metric"]) == {"ts", "pod", "far", "miss_rate", "frequency_bias"}
    assert len(scores) == len(SINGLE_THRESHOLDS) * 5
    assert set(scores["region"]) == {"china"}
    assert set(scores["lead_h"]) == {float(LEAD)}
    assert (scores["status"] == "success").all()

    derived = {
        "ts": lambda h, f, m: h / (h + f + m),
        "pod": lambda h, f, m: h / (h + m),
        "far": lambda h, f, m: f / (h + f),
        "miss_rate": lambda h, f, m: m / (h + m),
        "frequency_bias": lambda h, f, m: (h + f) / (h + m),
    }
    for thr in SINGLE_THRESHOLDS:
        row = wide[wide["threshold_mm"] == thr].iloc[0]
        hits, fls, miss = (
            int(row["hits"]), int(row["false_alarms"]), int(row["misses"]),
        )
        assert hits + fls + miss <= int(row["n_pairs"])
        for metric, formula in derived.items():
            expected = formula(hits, fls, miss)
            cell = scores[(scores["metric"] == metric) & (scores["threshold"] == thr)]
            assert len(cell) == 1, f"长表里 {metric} @ ≥{thr}mm 有 {len(cell)} 行"
            _rel(float(cell["value"].iloc[0]), expected, f"{metric} @ ≥{thr}mm（对列联表）", tol=1e-9)
