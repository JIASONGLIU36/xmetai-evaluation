# -*- coding: utf-8 -*-
"""FDP 集合场检验：Fengqing × CRA40。

流程 ``["fdp_field_scores", "fdp_ens_crps", "fdp_activity_spectrum"]`` 三段按顺序
跑、结果合并落同一个 output_dir。与 ``fdp_rmse_single_fengqing`` 的唯一区别是中间
多挂了 ``fdp_ens_crps``——同一批集合样本上既看确定性误差（集合均值 vs 实况），也看
集合分布本身的质量（CRPS 直接吃原始成员，不走均值）。

其余（数据源、要素表、时段、执行）与单场那份逐字一致，改路径记得两边都改。

要素缩到只有 z500（与标准脚本 ``ensemble_verifier.py`` 同口径——它也只算 Z500），
原因是多要素会在 CRA 侧撞 MergeError，详见 VARS 那处的注释。时段与时效照旧对齐旧
实现的 ``ensemble_verification_2026082000.csv``，``rmse`` / ``crps`` / ``spread``
这几列可以直接对表。

数据根：容器里的一份 fixture（fengqing 集合预报 + CRA40 实况），路径由下面的
``_DATA_ROOT`` 给出——**上传到哪就改那一行**。旧实现（``ensemble_verifier.py``）
在同一份数据上跑过一版，产物是 ``output/ensemble_verification_2026082000.csv``
（起报 2026082000、时效 6/12/18，Z500 的 CRPS / spread / RMSE / spread-RMSE 比）。
本配置的时段与时效就是照着它设的，跑完可以直接对表。
"""
from xmetai_evaluation.configs.base import EvalConfig, metric_options_from_var_metrics

# 只评 z500 —— 与标准脚本 ensemble_verifier.py 同口径（它也只算 Z500）。
# ⚠ 别急着把 t2m / u10 / v10 加回来：CRA40 实况走的是 CRA_LAYOUT，它带着非空的
#   grib_filters，于是每个变量各按自己的 filter_by_keys 单独开一次文件、最后合并；
#   t2m 带出 heightAboveGround=2、u10/v10 带出 =10，两个标量坐标对不上，
#   就在 io/gridded.py:364 那句 xr.merge 上撞 MergeError。
#   单变量不走那条合并分支，所以以前没暴露。要在配置侧加回多变量，先修那句 merge。
VARS = ["z500"]

# 指标表 -> metric_options。表里列出的指标名会被**路由**到这些变量。
# ⚠ 指标名不在这张表里 ≠ 这个指标不算！模板里声明了的指标一律会被实例化、
#   拿到 variable=None 跑**整批**（见 execution/executor.py::metric_runs 与
#   plan.py 里 metrics 的构建），只是吃不到这里的路由。所以：
#     * 删掉 "acc" → acc 仍会跑，没有 climate 时按零场兜底 → 静默出一列假数；
#     * 删掉 "activity" → activity 的 _anomalies 在无参考时**直接 raise MetricError**
#       （metrics/specialized.py:135），而 MetricError 一路 re-raise 到顶层
#       （executor.py:185 / :266 / :603），整个 run 挂掉。
#   要真正关掉某个指标，只能把它所在的整段从 pipeline 里去掉。
VAR_METRICS = {
    "z500": ["rmse", "bias", "acc", "activity", "spectrum", "crps", "spread_error"],
}

METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
# ACC 口径：True = 经典皮尔逊；False = FDP / WeatherBench2 的 uncentered。
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
# 谱的纬度朝向：不写 = 按框架传进来的形态原样算 —— weather 全系列就是这么跑的，
# 那边一行都不用动。标准脚本 activity_spectrum_verifier.py 把模式场和实况**都归到
# 纬度降序**（90→−90）再算，而这个谱函数的相位项用的是数组**行号**、
# **对翻转并不免疫**（它注释里写"纬度降序或升序均可"是错的）：不写这行，
# 谱的每个波数都跟它家对不上，奇数波数能差到 2 倍。
METRIC_OPTIONS["spectrum"] = {**METRIC_OPTIONS["spectrum"], "lat_order": "descending"}
# spread 的归一口径：不写 = 除以 M−1 无偏（FDP 口径）；对拍老仓 ref_result 才写
# ddof: 0（除以成员数 N）。要跟老档案对账时把这行打开。
# METRIC_OPTIONS["spread_error"] = {**METRIC_OPTIONS["spread_error"], "ddof": 0}

# ⚠ 数据根：改成你上传到容器里的实际路径，下面两行不用动。
#   目录里应当有 fengqing/（YYYYMMDD/Fengqing_1.0_GLB_*_ENS_FCST_*.nc，每个起报
#   120 个文件 = 60 时效 × PLEVELS+SURFACE）、cra_root/（YYYYMMDD/*.grib2）
#   和 cli_fake/（气候态）。
#   现在指的是**裁剪过的最小 fixture**（只留起报 20260820 的 006/012/018 三个时效
#   的 PLEVELS、以及 06/12/18 三个时刻的 ART_ATM 实况），供
#   tests/integration/test_fdp_fengqing_realdata.py 对表用。换真资料改这一行。
_DATA_ROOT = "/workspace/data/shenzw/fdp_test_data"
_FENGQING_ROOT = f"{_DATA_ROOT}/fengqing"
_CRA_ROOT = f"{_DATA_ROOT}/cra_root"

cfg = EvalConfig(
    name="fdp_rmse_ens_fengqing",
    description="FDP 集合场检验（Fengqing × CRA40）：RMSE / Bias / CRPS / Spread-Error / ACC / 活跃度 / 谱",
    pipeline=["fdp_field_scores", "fdp_ens_crps", "fdp_activity_spectrum"],

    forecast_reader={
        "type": "fengqing",
        "root_dir": _FENGQING_ROOT,
        "variables": VARS,
        "step_hours": 6.0,
        # 与旧实现那版对齐（见 docstring 提到的 ensemble_verification_2026082000.csv）：
        # 起报 20260820、时效 6/12/18 → 有效时刻 0820 的 06/12/18，三个样本。
        "lead_times": [6, 12, 18],
    },
    observation_reader={
        "type": "cra",
        "root_dir": _CRA_ROOT,
        "variables": VARS,
    },
    # ⚠⚠ 这份气候态是**编的**（tmp_data/make_fake_clim.py 生成到 cli_fake/），
    #   只为把链路跑通 —— ACC / 活跃度这两列的数字**没有任何意义**，别读它们。
    #   要读的 rmse / bias / crps / spread_error 四列不吃参考场，不受影响。
    #   为什么不能干脆删掉这一块：模板 fdp_field_scores 声明了 acc、
    #   fdp_activity_spectrum 声明了 activity，两者 needs_reference() 都是 True。
    #   指标是模板声明的，配置只能给参数、不能把它从段里摘掉；而 activity 在没有
    #   参考时**直接 raise MetricError**（metrics/specialized.py:135），整轮 run 会挂。
    #   所以要么挂着这个参考，要么把这两段整个从 pipeline 去掉。
    #
    #   ⚙ 走的是 climatology reader（CRA CLI_6HOUR 口径），**和标准脚本吃同一份
    #   文件**：{root}/ART_ATM_GLB_0P25_CLI_ANAL_{MMDD}{HH}.grib2，按"月日+时次"
    #   索引、与年份无关，读 gh@500。所以哪怕气候态是编的，两边也是逐位同源 ——
    #   活跃度那一列能直接对表。换真资料：把真的 CLI_6HOUR 目录指过来，或者只改
    #   下面 root_dir 这一行，文件名和目录层级都不用动。
    #   （旧写法是 daily_climatology + 单文件 clim_fake_z500.nc：按年内日序索引、
    #   还要 ±7.5 天环形平滑、5° 粗网格再插值到 0.25°，跟标准脚本的采样就不是
    #   同一份东西，活跃度只能对到 0.7%。换成目录式之后这个差没了。）
    #   ⚠ 这个 reader **不做单位换算**（只把 units 标成 DEFAULT_UNITS 里的 "m"），
    #   grib 里必须**已经是米**。标准脚本那条读气候态的路径同样不调 convert_z500，
    #   两边口径一致；真 CRA40 CLI 文件本来就是米，别往里塞 m²/s² 的东西。
    reference_reader={
        "type": "climatology",
        "root_dir": f"{_DATA_ROOT}/cli_fake",
    },

    # 起报 0820、时效 6/12/18 —— 与旧实现 ensemble_verification_2026082000.csv 同口径，
    # 跑完可直接对表。cra_root 只有 20260820 一天，所以有效时刻必须落在 0820：
    #   起报 2026082000 的 6/12/18 → 有效 0820 的 06/12/18  ✅ 有实况
    #   再往后（时效 24 → 有效 0821）就没有实况了，会各报一次 "No cra files found"
    #   后跳过，不影响其余样本。
    # 想多要几个样本就换成「起报 0819、时效 24/30/36/42」——有效时刻同样是
    # 0820 的 00/06/12/18，四个样本全配得上，只是与上面那份参考值不同时效、对不上表。
    start_date="20260820",
    end_date="20260820",
    limit=None,

    output_dir="/workspace/szwCode/evaluation_results/fdp_rmse_ens_fengqing",
    # 三段模板的输出都要写全：field_scores → csv_long；ens_crps → csv_long；
    # activity_spectrum → csv_long + spectrum。
    writers=["csv_long", "spectrum", "json"],
    metric_options=METRIC_OPTIONS,
    log_level="INFO",

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
    #                               轻指标→threads
    #                   ⚠ 本流程**写死 processes**，不用 auto —— crps 和谱都是
    #                     吃满单核的 numpy 计算，threads 形态下 GIL 会把它们
    #                     串起来，n_workers 调多大都不加速。
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
    # ⚠ observation 这里用 slice，不抄 weather 系列的 window:16 —— 那个窗口是给
    #   「时效跨 15 天」的配置用的：块序是起报日外层、时效窗内层，相邻起报组的
    #   观测日大面积重叠，滚动窗让一个观测日整 run 只读一次。本配置时效只有
    #   6/12/18（都在同一天），起报日 +1 天 ⇒ 观测日 +1 天，相邻起报组的观测日
    #   **不重叠**，窗口一次都命中不上、纯占内存。将来时效拉长（如 24..240）
    #   再换 window:16。

    # 内存备注（口径取自 weather_rmse_ens_* 的实测，z500 单要素）：
    #   每 worker ~1.7G 工作集
    # + 每 worker 一份成员场（集合独有）：1 要素 × 4 时效 × 51 成员 = 204 个场，
    #   _combine 拼接时峰值翻倍。
    # 本配置只有 1 个要素、没有气候态驻留那份 5.6G，所以比 weather 系列轻。

    execution={
        "mode": "processes",
        "n_workers": 24,        # 24 核 → 24 进程：np.fft 单线程，1 worker≈1 核不超订
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {
            "observation": "slice",     # CRA 按时刻一个 grib2；本配置时效同一天，窗块命中不上
            "reference": "resident",    # 日序气候态整 run 驻留，fork 共享 1 份
        },
        "resume": True,         # 已完成块的状态落 .states/，重跑跳过
    },
)
