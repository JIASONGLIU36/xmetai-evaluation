# -*- coding: utf-8 -*-
"""集合场检验（FuXi 集合 × ERA5）：对标 xu ``scripts/ensemble_fuxi.sh``。

流程 ``weather_ens_field_scores``：格点预报插值到 ERA5 网格、按 (起报, 时效)
配对，同一批集合样本上既看确定性误差（集合均值 vs 实况），也看集合分布
本身的质量（CRPS 直接吃原始成员，不走均值）。

z500 报 m²/s²（不除 g）、msl 报 Pa、风报 m/s。
``ws200`` 任何文件里都没有，三侧（预报/实况/气候态）都由 sqrt(u²+v²)
现合成。
"""
import os

from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# ── 要素表 ────────────────────────────────────────────────────────────────
# 逐个变量配对出分。可选值 = 两边 reader 布局里都有的名字（预报侧见
# io/layouts.py 的 FUXI_ENS_PHYS_LAYOUT，观测侧见 ERA5_ZARR_LAYOUT）。

# 全量表（正式跑用，替换上面一行）。ws200 现合成，见 docstring：
VARS = ["z500", "msl", "u200", "v200", "ws200"]

# ── 逐变量指标表 ──────────────────────────────────────────────────────────
# 每个变量点名的指标各出一行分。可用指标（components.py 注册表，完整说明
# 见 README「评估指标说明」）：
#   rmse / bias             误差族，只要预报+观测
#   acc / acc_uncentered    距平族，还要气候态参考（acc 的 centered 参数见下）
#   activity                活跃度比，也要气候态
#   zonal_spectrum          纬向谱（去纬向均值 rfft，cos 纬度加权），重指标
#   spherical_bands         球谐带功率（总波数分带，老仓 spherical_bands_*.csv
#                           同口径），重指标；只对全球含极网格有定义——配了
#                           下方 regions 分带时它恒全球，不逐带出分
#   crps / spread_error     集合族：crps 吃原始成员；spread_error 出 Spread /
#                           RMSE(集合平均) / 离散度-误差比三个数——老仓 regr_ens
#                           对集合**无条件**出这两个表（spread_*.csv /
#                           spread_rmse_ratio_*.csv），不看 metrics 列表，所以
#                           这里每个变量都点名（对齐老档案）
#   ts_score / fss / ensemble_probability  分类/空间/概率族（本流程不用）
# ⚠ 每个用到的指标都必须点到某组 variables 下（漏点名的会拿到整批多变量
#   数据，single_variable 直接报错——这是框架故意的，防静默出假数）。
# ⚠ crps 只路由给 z500（xu 也只对它算；spread 老仓是全变量出的）。


# 全量表（正式跑用，替换上面一块）：
VAR_METRICS = {
    "z500": ["rmse", "crps", "spread_error", "acc", "activity",
             "zonal_spectrum", "spherical_bands"],
    "msl": ["rmse", "spread_error", "acc"],
    "u200": ["rmse", "spread_error", "activity", "zonal_spectrum", "spherical_bands"],
    "v200": ["rmse", "spread_error", "activity", "zonal_spectrum", "spherical_bands"],
    "ws200": ["rmse", "spread_error", "activity", "zonal_spectrum", "spherical_bands"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# acc 用哪种口径：True = 经典皮尔逊（距平去均值再相关）；False = uncentered
# （FDP / WeatherBench2 口径，分子分母都带均值项）。xu 对标的是 False。
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# zonal_spectrum 取到多少个波数。720 = 全球 0.25° 的 Nyquist；写小了只截
# 曲线前段（大尺度），写超过 Nyquist 没有意义。
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}
# spherical_bands 的分带边界（总波数闭区间列表）。默认沿用老仓分段
# (1,4)/(5,20)/(21,40)/(41,64)/(65,128)，与老档案对拍就别动；要改就整表
# 覆盖，如 ((1, 10), (11, 40))。带边界上限受网格约束：
# 2*max_hi < nlat-1 且 max_hi < nlon//2（ERA5 721×1440 到 128 没问题）。
# METRIC_OPTIONS["spherical_bands"] = {
#     **METRIC_OPTIONS["spherical_bands"],
#     "bands": ((1, 4), (5, 20), (21, 40), (41, 64), (65, 128)),
# }
# spread 的归一口径：0 = 除以成员数 N（老仓 vfc regr_ens 的 p.std(ddof=0)，
# 对拍 ref_result 的 spread_*.csv 用这个）；1 = 除以 M−1 无偏（FDP 口径，
# 不写时的默认）。两口径差 sqrt((M-1)/M)，M=51 时约 1%。
METRIC_OPTIONS["spread_error"] = {**METRIC_OPTIONS["spread_error"], "ddof": 0}

_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

cfg = EvalConfig(
    name="weather_rmse_ens_fgvp",
    description="集合场检验（FuXi 集合 × ERA5）：z500/msl/u200/v200/ws200 的 RMSE / CRPS / ACC / 活跃度 / 谱 / 球谐带功率",
    # 走哪套流程模板。模板 = 协议 + 变换 + 指标的预设组合（全部可选值
    # `xmetai-eval --list-pipelines`）。本流程 = grid_valid_time 协议 +
    # ensemble_mean 变换（确定性指标吃集合均值，crps 例外——协议内部
    # 另给原始成员）。
    pipeline="weather_ens_field_scores",

    # ── 数据源三件套：type 决定布局与单位换算 ──────────────────────────
    # 预报侧 type 可选（components.py 注册）：fuxi_ens / fuxi_ens_phys /
    # fuxi / fuxi_phys / fengqing / fengqing_phys / cra。带 _phys 后缀 =
    # xu 复刻口径（z500 报 m²/s² 不除 g、要素表全），不带 = 旧口径
    # （z500 除 g 报位势米、只有 5 个要素）。fuxi_ens 与 fuxi_ens_phys
    # 的时效都按文件序号推（lead_from="index"，001.nc → +6h，步长写
    # step_hours）；fengqing 按文件名时效位推。
    forecast_reader={
        "type": "fuxi_ens_phys",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fgvp_ens_output"
        ),
        "variables": VARS,
        "step_hours": 6.0,
        # 时效列表（小时），三选一：
        #   不写这个键     从目录索引推：0, 6, 12, …, 目录里的最大时效
        #   写列表         覆盖：块只按列表里的时效读，可写稀疏集
        #                 （如 [24, 48, 72, 96, 120] 只评前 5 天，省读盘省
        #                  内存）——本流程无累积窗，每个时效都是独立样本，
        #                  稀疏集不会"凑不齐窗"
        #   显式写 None    与不写等价（不限制）
        "lead_times": None,
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

    # ── 评测时段（periods 与 start_date/end_date **二选一**，同时给会报配置错）──
    # 当前跑五段：起报日共 228 个 × 16 个时效窗 = **3648 块**。五段的结果汇成
    # **一套**统计量（每个 变量×指标×时效只有一行，不是每段各一行）；段与段之间
    # 的空档**不切块、不出样本**。换回单时段：把这行注掉、把下面 start_date/
    # end_date 两行解开。
    #
    # 五段 = ``fgvp_ens_merged`` 里**完整**的起报日（2025-11-19 体检清单实测）：
    #     20250102 .. 20250411  100 天
    #     20250504 .. 20250518   15 天
    #     20250702 .. 20250721   20 天
    #     20250723 .. 20251007   77 天
    #     20251104 .. 20251119   16 天      共 228 天
    # 目录里另有 48 天是**残缺**的（缺成员目录 / 缺步文件），**不能放进来**：
    #   ``io/gridded.py`` 里 ``lead_from="index"`` 的布局不按文件名推时效，按
    #   "该成员目录里找到的第几个文件"推（``(position+1)*step_hours``）——成员
    #   中间缺一个步文件，它后面所有时效**整体前移一格**（021.nc 被当成 L120），
    #   不报错也不告警（``_warn_on_member_mismatch`` 只比成员数，且单日起报比不
    #   出来）。残缺日补齐后（``check_merge_ens.py --copy/--move``）可以并回来。
    #
    # ⚠ 起报日从 8 个涨到 228 个，单块内存不变（块口径是"1 起报日 × 1 天时效"），
    #   但**下面内存备注里那条"换回全量表必须降 n_workers"现在真的生效了**：
    #   VARS 是本文件的 5 要素表（不是 z500 冒烟态），按那条备注气候态 resident
    #   会到 ~67G、窗缓存 ~95G，n_workers=24 是 400G 量级。跑之前先按机器内存
    #   把 n_workers 降到 8~12，或把 observation 退回 slice。
    #
    # ✓ 本配置**不会**为空档期多读观测，这点与站点协议的两个 config 不同。
    #   grid_valid_time 的观测时刻是按实际 (起报 × 时效) 现算的
    #   （execution/plan.py 的 ``_grid_valid_times``），空档期算不出任何有效时刻，
    #   ERA5 侧一个请求都不会发。站点协议的 ``station_observation_span`` 取的是
    #   「最早起报 → 最晚起报」的连续包络，那边空档才会被照读。同一份 periods
    #   在两种协议下观测读量不同，这是协议口径的差别，不是配置的问题。
    #   （observation 的 window:16 滚动缓存跨过 4 道空档时会落空重读，
    #    只是那几天边界缓存，与空档跨度无关。）
    periods=[
        ("20250102", "20250411"),   # 100 天
        ("20250504", "20250518"),   #  15 天
        ("20250702", "20250721"),   #  20 天（7/22 残缺，单独剔在外面）
        ("20250723", "20251007"),   #  77 天
        ("20251104", "20251119"),   #  16 天
    ],

    # 单时段写法（当前未启用）：
    # start_date=os.environ.get("START_DATE", "20250102"),
    # end_date=os.environ.get("END_DATE", "20251215"),
    limit=None,  # 限起报数：冒烟改 2；None = 不限。改回 None 重跑时 resume
                 # 会复用冒烟算过的块，不白算

    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/evaluation_results/weather_rmse_ens_fgvp",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",  # DEBUG / INFO / WARNING / ERROR

    # ── 输出视图（可多选）──────────────────────────────────────────────
    #   csv_long          统一长表 scores.csv（主表，任何评测都该带上）
    #   spectrum          逐波数谱曲线 diagnostics/spectrum_*.csv（zonal_spectrum
    #                     必加：720 个点的曲线走长表会爆行数）
    #   categorical_wide  分类检验宽表（ts_score 配套）
    #   probability_wide  集合概率宽表 AROC/BS/BSS
    #   coverage / details / json  覆盖率、诊断明细、JSON 快照
    # 注：spherical_bands 不需要 spectrum writer——每带只有 3 行标量，
    # 直接进长表（带名在 group 列）。
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
        # 为空）；标量指标（RMSE/ACC/活跃度/CRPS 等）逐带出分，
        # zonal_spectrum / spherical_bands / FSS 恒全球（球谐与纬向谱要
        # 完整球面场，掩掉的子区域上算出来的不是同一个物理量）。
        # 不分区就删掉这个键，输出回到只有全球行。

        # 下面是经典三分带（热带 ±20° + 两半球中高纬，拼满全球不重叠），
        # 改边界直接改数字。
        "regions": {
            # "tropics": {"lat_min": -20, "lat_max": 20},
            # "nh_extratropics": {"lat_min": 20, "lat_max": 90},
            # "sh_extratropics": {"lat_min": -90, "lat_max": -20},
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
    #                               轻指标→threads。本流程含 crps+谱 → 即
    #                               processes
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
    # 本配置规模（z500 冒烟态）：lead 到 360h、按 lead_chunk_days=1 切 16 个
    # 时效窗 × 起报数 = 16×N 块。observation 用 window:16（= 整段时效 15 天
    # + 起报 1 天，即自动定窗值）：块序是起报日外层、时效窗内层，相邻起报组
    # 的观测日大面积重叠，滚动窗让一个观测日整 run 只读一次。

    # 内存备注（实测口径，z500 单要素）：气候态 resident 预热 ~5.6G（共享
    # 1 份，与 worker 数无关）+ 每 worker ~1.7G 工作集（集合还多一份成员场：
    # 1 要素 × 4 时效 × 51 成员 = 204 个场，_combine 拼接时峰值翻倍）+ 观测
    # 窗块单通道 ~0.8G/进程。
    # ⚠ 换回全量表（5 要素）：气候态 resident 涨到 ~67G、工作集 ×5、窗缓存
    #   ×5（~95G），n_workers=24 就是 400G 量级——必须把 n_workers 降到
    #   8~12，或把 observation 退回 slice。

    execution={
        "mode": "processes",
        "n_workers": 24,       # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {
            "observation": "window:16",
            "reference": "resident",     # 日序气候态整 run 驻留（z500 实测 5.6G，fork 共享 1 份）
        },
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
