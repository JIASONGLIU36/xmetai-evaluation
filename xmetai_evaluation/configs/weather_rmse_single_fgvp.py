# -*- coding: utf-8 -*-
"""确定性连续量检验（FuXi × ERA5）：对标 xu ``scripts/single_fuxi.sh``。

流程 ``weather_field_scores``：格点预报插值到 ERA5 网格、按 (起报, 时效)
配对，逐变量出 RMSE / 纬向谱 / ACC / 活跃度。

z500 报 m²/s²（不除 g）、
q 报 g/kg、tp 报 mm。
``ws850`` 任何文件里都没有，三侧（预报/实况/气候态）
都由 sqrt(u²+v²) 现合成；
``ws10m`` 例外——ERA5 store 自带 10m 风速通道，
实况侧直接取用。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics


# 全量表（正式跑用，替换上面一行）。ws850 现合成、ws10m 实况直取，见 docstring：
VARS = [
    "z500", "q700", "t700", "t2m", "t850", "msl",
    "u850", "v850", "u10m", "v10m", "ws10m", "ws850",
]

# ── 逐变量指标表 ──────────────────────────────────────────────────────────
# 每个变量点名的指标各出一行分。可用指标（components.py 注册表，完整说明
# 见 README「评估指标说明」）：
#   rmse / bias             误差族，只要预报+观测
#   acc / acc_uncentered    距平族，还要气候态参考（acc 的 centered 参数见下）
#   activity                活跃度比，也要气候态
#   zonal_spectrum / spectrum  谱族，拿完整场，重指标
#   spherical_bands         球谐带功率（总波数分带，老仓对每个带 spectrum 的
#                           变量都出 spherical_bands_<var>.csv，这里同口径；
#                           配了 regions 时恒全球）
#   crps / spread_error     集合族（确定性流程不用）
#   ts_score / fss / ensemble_probability  分类/空间/概率族（本流程不用）
# ⚠ 每个用到的指标都必须点到某组 variables 下（漏点名的会拿到整批多变量
#   数据，single_variable 直接报错——这是框架故意的，防静默出假数）。

# 全量表（正式跑用，替换上面一块）：
VAR_METRICS = {
    "z500": ["rmse", "zonal_spectrum", "spherical_bands", "acc", "activity"],
    "q700": ["rmse"],
    "t700": ["rmse", "zonal_spectrum", "spherical_bands"],
    "t2m": ["rmse", "zonal_spectrum", "spherical_bands"],
    "t850": ["rmse"],
    "msl": ["rmse", "zonal_spectrum", "spherical_bands"],
    "u850": ["rmse", "activity", "zonal_spectrum", "spherical_bands"],
    "v850": ["rmse", "activity", "zonal_spectrum", "spherical_bands"],
    "u10m": ["rmse", "activity"],
    "v10m": ["rmse", "activity"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# acc 用哪种口径：True = 经典皮尔逊（距平去均值再相关）；False = uncentered
# （FDP / WeatherBench2 口径，分子分母都带均值项）。xu 对标的是 False，

METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# zonal_spectrum 取到多少个波数。720 = 全球 0.25° 的 Nyquist；写小了只截
# 曲线前段（大尺度），写超过 Nyquist 没有意义。
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}

_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

cfg = EvalConfig(
    name="weather_rmse_single_fgvp",
    description="确定性连续量检验（FuXi × ERA5）：12 要素逐变量的 RMSE / 谱 / ACC / 活跃度",
    # 走哪套流程模板。模板 = 协议 + 变换 + 指标的预设组合（全部可选值
    # `xmetai-eval --list-pipelines`）。本流程 = grid_valid_time 协议 +
    # ensemble_mean 变换（确定性输入没有成员维，变换为空操作）。
    pipeline="weather_field_scores",

    # ── 数据源三件套：type 决定布局与单位换算 ──────────────────────────
    # 预报侧 type 可选（components.py 注册）：fuxi / fuxi_phys / fuxi_ens /
    # fuxi_ens_phys / fengqing / fengqing_phys / cra。带 _phys 后缀 =
    # xu 复刻口径（z500 报 m²/s² 不除 g、要素表全），不带 = 旧口径
    # （z500 除 g 报位势米、只有 5 个要素）。fuxi 与 fuxi_phys 的时效
    # 都按文件序号推（lead_from="index"，001.nc → +6h，步长写 step_hours）；
    # fengqing 按文件名时效位推。
    forecast_reader={
        "type": "fuxi_phys",
        "root_dir": os.environ.get(
            "FGVP_OUTPUT", "/workspace/data/shenzw/fgvp_single_output"
        ),
        "variables": VARS,
        "step_hours": 6.0,
        # 时效列表（小时），三选一：
        #   不写这个键     从目录索引推：0, 6, 12, …, 目录里的最大时效
        #                 （本配置目录推到 360h，61 个时效）
        #   写列表         覆盖：块只按列表里的时效读，可写稀疏集
        #                 （如 [24, 48, 72, 96, 120] 只评前 5 天，省读盘省
        #                  内存）——本流程无累积窗，每个时效都是独立样本，
        #                  稀疏集不会"凑不齐窗"
        #   显式写 None    与不写等价（不限制）
    },
    # 观测侧 type 可选：era5_zarr（格点 ERA5，pl/sfc 两个 zarr，变量自动
    # 路由到所在 store）/ station（站点降水）/ cra / ref_probability。
    observation_reader={
        "type": "era5_zarr",
        "stores": {
            "pl": os.environ.get(
                "ERA5_PL_STORE",
                f"{_ERA5_STORE_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
            ),
            "sfc": os.environ.get(
                "ERA5_SFC_STORE",
                f"{_ERA5_STORE_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
            ),
        },
        "variables": VARS,
    },
    # 参考场，只有 acc / activity 用得上（配置里给了、指标用不上时不会构建，
    # 所以不相关的配置可以照抄）。type 可选：daily_climatology（逐日气候态
    # 文件，smooth_days 控制取场前的环形平滑天数）/ climatology / ref_probability
    # （气候概率，BSS 用）。smooth_days 是气象参数（= xu 的 --climo-window），
    # 与 execution.loads 的 window:W（IO 缓存）没有关系。
    reference_reader={
        "type": "daily_climatology",
        "root_dir": os.environ.get("ERA5_CLIMO", "/workspace/data/worm/era5_clim_phys_14.nc"),
        "smooth_days": 15,  # ±7.5 天中心滑动、跨年首尾相接，抹平单日气候态噪声
    },

    # 评测时段（YYYYMMDD 或 YYYYMMDDHH），只管起报日的范围；观测/参考读
    # 多少由框架按时效跨度自己算。
    start_date=os.environ.get("START_DATE", "20250102"),
    end_date=os.environ.get("END_DATE", "20251215"),
    limit=None,  # 限起报数：冒烟改 2；None = 不限。改回 None 重跑时 resume
                 # 会复用冒烟算过的块，不白算

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/evaluation_results/weather_rmse_single_fgvp",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",  # DEBUG / INFO / WARNING / ERROR

    # ── 输出视图（可多选）──────────────────────────────────────────────
    #   csv_long          统一长表 scores.csv（主表，任何评测都该带上）
    #   spectrum          逐波数谱曲线 diagnostics/spectrum_*.csv（谱指标
    #                     必加：720 个点的曲线走长表会爆行数）
    #   categorical_wide  分类检验宽表（ts_score 配套）
    #   probability_wide  集合概率宽表 AROC/BS/BSS
    #   coverage / details / json  覆盖率、诊断明细、JSON 快照
    writers=["csv_long", "spectrum"],

    # ── 协议参数 ────────────────────────────────────────────────────────
    options={
        # 采样口径，二选一：
        #   "init_lead"   每 (起报, 时效) 一个样本，逐起报报满整段时效。
        #                 逐日起报配 15 天时效必须用它——"valid_time" 口径下
        #                 同一有效时刻只认最新起报，每起报只剩头 4 个时效
        #   "valid_time"  每个有效时刻一个样本、最新起报获胜（缺省）
        "sample_by": "init_lead",
        # 分纬度带评估：{带名: {"lat_min": 南界, "lat_max": 北界}}，边界闭
        # 区间。每个样本除全球行外逐带再出一行，长表 region 列区分（全球行
        # 为空）；标量指标逐带出分，谱/FSS 恒全球（掩掉的子区域上算出来的
        # 不是同一个物理量）。不分区就删掉这个键，输出回到只有全球行。


        # 下面是经典三分带（热带 ±20° + 两半球中高纬，拼满全球不重叠），
        # 改边界直接改数字。
        "regions": {
            "tropics": {"lat_min": -20, "lat_max": 20},
            "nh_extratropics": {"lat_min": 20, "lat_max": 90},
            "sh_extratropics": {"lat_min": -90, "lat_max": -20},
        },
    },

    # ── 并发与数据加载 ──────────────────────────────────────────────────
    # 可用键就这 6 个，全部字面量；只覆盖写到的键（不写 = 按指标族与数据源
    # 形态推导），写错键名直接报配置错、不会静默忽略。
    #
    # mode             并发形态，四选一：
    #                    serial     单进程逐块（冒烟、单块用）
    #                    threads    1 进程 N 线程，共享同一份驻留（轻指标用）
    #                    processes  进程池；块按时间连续分段、一段固定一个
    #                               子进程顺序跑完（段内相邻块命中窗块缓存）
    #                    auto       推导：单块→serial；重指标→processes；
    #                               轻指标→threads。本流程含谱 → 即 processes
    # n_workers        进程/线程数。processes 缺省 = CPU 核数。上限往往是
    #                  内存不是核数（见下方内存备注）
    # chunk_days       一个块装几个起报日（缺省 1）——管起报跨度
    # lead_chunk_days  一个块装几天时效（缺省 1）——管时效跨度，是单块内存
    #                  的主旋钮；0 = 不切（整段时效一块，与切块功能加入前
    #                  逐位一致，逃生口）
    # loads            角色驻留 {角色: 策略}。角色只有 observation / reference
    #                  （预报恒为 slice、不在表里，写了报错）：
    #                    slice     每块现读现弃（缺省）
    #                    resident  整 run 读一次驻留（气候态用；fork 前预热、
    #                              子进程 COW 共享，全程只有 1 份）
    #                    window    自动定窗 = 一个起报日组的观测日跨度
    #                    window:W  滚动缓存上限 W 天；实际取 min(W, 自动值)，
    #                              写大了不更省、只多占内存。窗块每进程各一份
    #                              （不像 resident 那样共享）
    # resume           True = 已完成块的状态落 output_dir/.states/，重跑跳过
    #
    # 本配置规模：lead 到 360h、按 lead_chunk_days=1 切 16 个时效窗 × 348
    # 起报 = 5568 块。observation 用 window:16（= 整段时效 15 天 + 起报 1 天，
    # 即自动定窗值）：块序是起报日外层、时效窗内层，相邻起报组的观测日大
    # 面积重叠，滚动窗让一个观测日整 run 只读一次。

    # 内存备注（实测口径）：气候态 resident 预热 ~67G（共享 1 份，与 worker
    # 数无关）+ 每 worker ~2G 工作集；观测窗块单通道 ~0.8G/进程。

    execution={
        "mode": "processes",
        "n_workers": 24,       # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {
            "observation": "window:16",
            "reference": "resident",     # 日序气候态整 run 驻留（实测 67G，fork 共享 1 份）
        },
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
