# -*- coding: utf-8 -*-
"""FDP 降水检验（确定性口径）：Fengqing tp × Diamond 站点实况。

流程 ``["fdp_precip_ts"]``：集合平均场 → 双线性插值到站 → TS / POD / FAR /
频率偏差。评的是「集合平均当确定性预报」的落区与量级。

与 ``fdp_precip_ens_fengqing`` 同源（数据根、时段、时效、执行全同），差别只在
pipeline：那份走 ``weather_ts_ens_prob``，逐成员算概率评分 BS / AROC / BSS。
**改路径、改时段记得两份都改。**

⚠ 起报日和时效都被数据钉死了，两样都得跟着数据走：

* **起报日**：``fengqing/`` 里只有起报 **20260820**（时效 006/012/018 三个文件）。
  ``start_date`` 写的就是起报日，``discover`` 拿它拼 ``fengqing/{起报日}/``；
  目录不存在时是**静默 skip**（io/gridded.py:79-82），一个文件都没扫到才抛
  ``No fengqing files found``。所以日期写错不会报「日期错」，只会报这句。
* **时效**：站点侧 ``rain/`` 只有 20260820 的 00Z~06Z 七个文件，而 6h 窗口是
  闭区间 ``[有效时刻-5h, 有效时刻]``（transforms/temporal.py:43），所以只有
  **有效时刻 0820 06Z** 那个窗（01Z~06Z 六个时次）凑得齐 —— 即起报 0820 的
  **时效 6**。时效 12 / 18 落在 06Z 之后，窗口取不到观测。

换到服务器上数据齐了，把 ``_LEAD_TIMES`` 和日期一起放开。

格点 FSS 段（``_grid``）默认不启用 —— 它要**格点**降水实况。``cra_root`` 里现在
有 tp 了，但那份是**编的**（``tmp_data/make_fake_precip.py`` 生成，观测侧和
fengqing 的 SURFACE 侧都是），够把链路跑通、看 FSS 随尺度的衰减，数字没有气象
意义；也没出正式数，理由和冒烟要怎么调见那一段的注释。
"""
from xmetai_evaluation.configs.base import EvalConfig

# 数据根：**本机 tmp_data/ 就是它的镜像**，整份传上去就行，下面三颗子树同根
# （与 fdp_rmse_single_fengqing 的 _DATA_ROOT 是同一个字面量）。两份配置里也是
# 同一份字面量，改一处记得改两处（没有抽公共模块，就是为了让单独一份拿出去
# 也能直接跑）。
_DATA_ROOT = "/workspace/data/shenzw/fdp_test_data"
_FENGQING_ROOT = f"{_DATA_ROOT}/fengqing"
_CRA_ROOT = f"{_DATA_ROOT}/cra_root"
_STATION_ROOT = f"{_DATA_ROOT}/rain"
# 只有时效 6 配得上：观测只到 0820 06Z（详见模块 docstring）。
_LEAD_TIMES = [6]


def _execution():
    """两条配置各拿一份新的 execution —— 共用同一个 dict 会被就地改写串味。"""
    return {
        "mode": "auto",
        "n_workers": 4,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {
            "observation": "resident",  # 站点观测整 run 驻留（模板推导的缺省也是这个）
        },
        "resume": True,
    }


# ── 站点段：集合平均 → 插值到站 → TS ──────────────────────────────────────
# 模板 ``fdp_precip_ts`` 自带 ``ensemble_mean``：fengqing 是 21 成员的集合文件，
# 而 ``ts_score`` 声明的是 DETERMINISTIC_FIELD（metrics/categorical.py:75），
# ``Metric.validate`` 见到 member>1 直接 raise（metrics/base.py:125）。站点协议
# 这条路**没有** Matcher 兜底，所以降维只能靠模板里那个变换。
# 窗口 6h、不经 24h 累积，按 UTC 对齐 —— 模板 fdp_precip_ts 自己就写死
# local_utc_offset_hours=0，本配置不写 options，吃的就是那个缺省。对得上是因为
# 这份站点数据的文件名标签（2026082000–06）本来就是 UTC 有效时刻。
# ⚠ 换成服务器上按**北京时**打标签的 Diamond 归档时要补
# options={"local_utc_offset_hours": 8}（fuxi / fgvp / rainstorm 全家都是这么写
# 的），否则整段错 8 小时，一个样本都配不上。
_station = EvalConfig(
    name="fdp_precip_single_fengqing",
    description="FDP 降水站点检验（确定性，Fengqing × Diamond 站点）：TS / POD / FAR / 频率偏差",
    pipeline="fdp_precip_ts",

    forecast_reader={
        "type": "fengqing",
        "root_dir": _FENGQING_ROOT,
        "variables": ["tp"],
        "step_hours": 6.0,
        "lead_times": _LEAD_TIMES,
    },
    observation_reader={
        "type": "station",          # 别名 diamond_station
        "root_dir": _STATION_ROOT,
        "variable": "precipitation",
        # 不写 station_list = 用文件里的全部站点；要只用白名单就填那个 .dat 的路径
    },
    # 不配 reference_reader：fdp_precip_ts 的指标没有 needs_reference 的，
    # 配了也不会去建（构建参考的是另一份配置，见 fdp_precip_ens_fengqing）。

    start_date="20260820",   # 起报日；fengqing/ 里只有这一天
    end_date="20260820",
    limit=None,

    output_dir="/workspace/szwCode/evaluation_results/fdp_precip_single_fengqing",
    # ⚠ 配置级 writers 是**替换**模板的，这里把它写全：
    # fdp_precip_ts → csv_long + categorical_wide。漏了对应那项，报告里就没有那张主表。
    writers=["csv_long", "categorical_wide"],
    log_level="INFO",
    execution=_execution(),
)

# ── 格点段：FSS（默认不启用）─────────────────────────────────────────────
# FSS 是网格邻域检验，**不吃站点观测**，要格点降水实况。cra_root 里那份 tp 是
# **编的**（CRA40LAND_SURFACE_*，tmp_data/make_fake_precip.py），够跑通链路、
# 看 FSS 随尺度衰减，数字没有气象意义。编的这份只有起报 20260820、时效
# 6/12/18（对应有效时刻 0820 的 06/12/18）—— 和 fengqing 那份 fixture 正好对齐，
# 所以下面的起报日和时效就是按它设的。出正式数就换成服务器上带 tp 的 CRA40
# 归档（或直接换成 CMPAS 的格点降水 reader），同时把时效放开到 6…42。
_grid = EvalConfig(
    name="fdp_precip_fss",
    description="FDP 降水空间检验（Fengqing × CRA40）：FSS",
    pipeline="fdp_precip_fss",

    forecast_reader={
        "type": "fengqing",
        "root_dir": _FENGQING_ROOT,
        "variables": ["tp"],
        "step_hours": 6.0,
        "lead_times": [6, 12, 18],
    },
    observation_reader={
        "type": "cra",
        "root_dir": _CRA_ROOT,
        "variables": ["tp"],
    },

    start_date="20260820",   # 起报日；编的那份 tp 只有这一天
    end_date="20260820",
    limit=None,

    output_dir="/workspace/szwCode/evaluation_results/fdp_precip_fss",
    # 模板给的是 ["csv_long"]；FSS 的窗口维在长表里靠 group 列区分，不用额外视图。
    writers=["csv_long"],
    log_level="INFO",
    execution=_execution(),
)

cfgs = [
    _station,
    # _grid,   # ← 取消注释即启用 FSS（前提：观测侧拿得到格点降水）
]
