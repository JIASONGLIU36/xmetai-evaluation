# -*- coding: utf-8 -*-
"""集合降水分类检验（FuXi 集合 × Diamond 站点）：一条命令跑两段。

口径由负责人给定，对标原版 ``run_categorical.py``（``--window-ts 24 --window-aroc 6``）：

    窗口 24  集合平均场   TS / POD / FAR / 频率偏差        -> weather_ts_ens
    窗口 6   逐成员        AROC / BS / BSS（要气候概率）    -> weather_ts_ens_prob

两段的窗口不同，而一条模板只有一个 ``window_hours``（它同时决定观测累积长度和
有效时效的筛选），所以配置里用列表声明两条模板，框架按顺序各跑一段、结果合并后
落盘一次：

    scores.csv                        两段的统一长表，靠 window_h 列区分（24 / 6）
    diagnostics/categorical_wide.csv  仅 24h 的 TS 宽表
    diagnostics/probability_wide.csv  仅 6h 的 AROC/BS/BSS 宽表
    manifest.json                     运行记录（segments 里有每段的窗口）

另外 ``periods`` 还声明了五个**互不相连的时段**（2025 年 1/2–4/11、5/4–5/18、
7/2–7/21、7/23–10/7、11/4–11/19，共 228 个起报日）。这与"两段窗口"是**两根
独立的轴**：上面两条模板**都会**评这五个时段，结果同样合并落盘，所以实际是
窗口 2 种 × 起报日 228 个 × 时效窗（24h 段 15 / 6h 段 16）= **7068 块**。段间
空档不切块、不出样本（但观测会被照读，见 ``periods`` 那行的 ⚠）。

阈值不在这里配：两条模板自带的阈值常量（24h 的 [0.1,10,25,50,100]、
6h 的 [0.1,4,13,25]）已与原版输出逐格一致，且 6h 那组必须与参考文件的 4 列同集合。

数据：
    预报  {fuxi_ens_output}/YYYYMMDD/member_*/001.nc…（TP，逐 6h）
    观测  Diamond 站点降水（北京时，协议默认 +8 对齐）
    参考  气候概率目录 ref/MMDDHH.000（**必填**，见 BSS_REF_DEFAULT）
"""

import os

from xmetai_evaluation.configs.base import EvalConfig

#: BSS 的外部气候概率目录（``ref/MMDDHH.000``：站号 + 4 个超越式概率）。
#: 负责人给定的固定输入，对应原版的 ``--ref``；换目录直接改这一行。
BSS_REF_DEFAULT = os.environ.get("BSS_REF", "/workspace/data/worm/r0/ref")

cfg = EvalConfig(
    name="weather_ts_ens_fgvp",
    description="FuXi 集合降水评估：24h 集合平均 TS + 6h 逐成员概率 AROC/BS/BSS",
    # 两段：先 24h 的 TS，再 6h 的概率评分（顺序即执行顺序，结果合并落盘）
    pipeline=["weather_ts_ens", "weather_ts_ens_prob"],

    forecast_reader={
        "type": "fuxi_ens",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fgvp_ens_merged"
        ),
        "variable": "tp",
        "step_hours": 6.0,      # 布局步长：lead_from="index"，001.nc → +6h
        # 时效列表（小时）。None / 整个键不写 = 从目录索引推：取最大时效 lead_max，
        # 按 step_hours 从 0 铺起（0, 6, 12, …, lead_max）。
        #
        # ⚠ 本配置的**两段都带累积窗**（24h 段要 4 个连续 6h 步、6h 段要 1 步），
        # 而预热时效只从这张列表里取。要写就必须**按 6h 密排**，只写采样点会让
        # 窗凑不齐步数、一个样本都出不来：
        #     ✗ [24, 48, 72, …, 360]         24h 段每个窗只有 1 步
        #     ✓ [6, 12, 18, 24, 30, …, 360]  与推导结果同集合
        # 只读部分时效可显著降内存（如 [6, 12, 18, 24]）；注意 24h 段要求时效里
        # 有 24 的倍数，裁太短会让那一段一个样本都跑不出来。
        "lead_times": None,
    },
    observation_reader={
        "type": "station",
        "root_dir": os.environ.get("STATION_OBS", "/workspace/data/worm/r0/2025"),
        "variable": "precipitation",
        "station_list": os.environ.get(
            "STATION_LIST", "/workspace/data/worm/r0/zd_sta_10285.dat"
        ),
    },
    # BSS 的气候概率参考。必须给：缺了会降级成样本气候频率 r(1-r)，
    # 那算出来的 BSS 与原版不可比。
    reference_reader={"type": "ref_probability", "root_dir": BSS_REF_DEFAULT},

    # ── 评测时段（periods 与 start_date/end_date **二选一**，同时给会报配置错）──
    # 这里有两根**互相独立**的"多段"轴，别混：
    #   ① 上面的 ``pipeline`` 列表 —— 两条模板各跑一段（24h TS / 6h 概率）
    #   ② 这个 ``periods``         —— 五个互不相连的时间段，**两条模板都会评**
    # 所以实际是 窗口 2 种 × 起报日 228 个 × 时效窗（24h 段 15 / 6h 段 16）
    # = **7068 块**。
    # 五段的结果汇成**一套**统计量（每个 变量×时效×阈值只有一行，不是每段各
    # 一行）；段与段之间的空档**不切块、不出样本**。
    # 换回单时段：把这行注掉、把下面 start_date/end_date 四行解开。
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
    # ⚠ 观测跨度取的是「最早起报 → 最晚起报」（protocols.py 的
    #   ``station_observation_span``），所以**段间空档的观测会被照读一遍**。
    #   现在的包络是 1/2 → 11/19（再往后延 15 天时效 ≈ 12/4），本配置又是
    #   observation=resident，等于整个 run 把 2025-01-01 到 2025-12-04 的
    #   Diamond 逐时文本全读一次。这是拿 IO 换省事，不是错误——真介意这块，
    #   得再加一层"按段裁观测跨度"。**预报侧不读空档**（块按起报日切）。
    periods=[
        ("20250102", "20250411"),   # 100 天
        ("20250504", "20250518"),   #  15 天
        ("20250702", "20250721"),   #  20 天（7/22 残缺，单独剔在外面）
        ("20250723", "20251007"),   #  77 天
        ("20251104", "20251119"),   #  16 天
    ],

    # 单时段写法（当前未启用）：
    # start_date=os.environ.get("START_DATE", "20250102"),
    # 止于 15 号、不是 31 号：**站点观测档案到 2025-12-31 23:00 就没了**，预报比它长。
    # 起报越靠后，长时效的验证时刻越容易越过这条线，那种块整块评不出样本（日志里是
    # 「观测窗口不完整」+「没有任何样本可评」），只会白占一个失败块。15 号的最长时效
    # 落在 12-30，留一天余量。观测补齐后可改回来，`.states/` 里的旧块会原样复用。
    # end_date=os.environ.get("END_DATE", "20250104"),
    limit=None,  # 限起报数，直接改这里；None = 不限

    output_dir=os.environ.get(
        "EVAL_OUTPUT", "/workspace/szwCode/evaluation_results/weather_ts_ens_fgvp"
    ),
    log_level="INFO",

    # 协议口径（station_valid_time）：8 = 北京时，Diamond 观测自带的口径，也是协议
    # 缺省。另可配 region（区域筛选）、weights（"cos_lat"）。两段共用这一份。
    options={
        "local_utc_offset_hours": 8,

        # 分纬度带评估：{带名: {"lat_min": 南界, "lat_max": 北界}}，**闭区间**。
        # 每个样本除全球行外逐带再出一行（长表 region 列区分，全球行为空），
        # 站点按**站点纬度**整站归带。不带分带就删掉这个键，输出回到只有全球行。
        #
        # ⚠ ``options`` 是**两段共用**的，所以 24h 的 TS 段和 6h 的概率段会**各出**
        #   5 份行（全球 + 4 带）。读盘与插值不会重做——带样本与全球样本共用
        #   ``build_batch`` 缓存的那个批次（全场插值只做一次），只有指标累加按带
        #   各跑一遍，所以代价是计算量 ×5 而不是读盘量 ×5。批次本身也共享，
        #   峰值内存只多出几条带各自的指标状态。
        #
        # 带名就是报告里的显示名，**报告不翻译也不写死任何带名**；表按南界从低
        # 到高排，改边界/改带名直接改这里。
        #
        # 闭区间的代价：正好落在 25 / 32 / 40 上的站**两条带都算**（与格点侧
        # 口径一致，改成半开会让两类产物的分带对不上）。站点纬度极少正好压线，
        # 真压上了也就是这两条带各多它一个站，不构成口径问题。
        #
        # 南方那条的南界写 0 而不是 -90：这是中国站点集（zd_sta_10285.dat），
        # 赤道以南没有站；换成含南半球的站点集时这一带要重划。
        #
        # 报告侧的「附 L」只汇总 24h 那段的 TS；概率段的带行**照常落盘**
        # （scores.csv 与 diagnostics/probability_wide.csv 里都有 region 列），
        # 只是报告暂时不出——第六节本来就没有渲染器，见
        # ``assets/templates/weather_ts_ens.md`` 顶部那条说明。
        "regions": {
            # "南方": {"lat_min": 0, "lat_max": 25},
            # "长江中下游": {"lat_min": 25, "lat_max": 32},
            # "华北": {"lat_min": 32, "lat_max": 40},
            # "东北": {"lat_min": 40, "lat_max": 90},
        },
    },

    # ⚠ **不要在这里写 writers**。本配置是两段（weather_ts_ens + weather_ts_ens_prob），
    # 而配置级的 writers 是**一个列表同时作用到两段**——写了就会让 6h 概率那段也去
    # 出 categorical_wide（TS 宽表），而它真正该出的是 probability_wide。
    # 不写，各段才用各自模板自带的那份：24h 段 ["csv_long", "categorical_wide"]、
    # 6h 段 ["csv_long", "probability_wide"]。

    # ── 并发与数据加载：execution ──────────────────────────────────────
    # 可用键**就下面这 6 个**，全部是字面量，只覆盖写到的键（不写 = 用推导值）。
    # 写错键名会直接报配置错，不会静默忽略。
    #
    #   mode             ——**写死 processes**，两段都用进程。推导缺省给 24h 段的
    #                     形态是 threads（ts_score 属轻量指标），但本配置是集合源
    #                     （一个 run 要打开约 106 万个 nc），而 threads 下**所有**
    #                     netCDF 打开都排在 ``io/base.py::hdf5_guard`` 的同一把全局
    #                     锁上——那把锁正是为「HDF5 不是线程安全」加的：同进程内
    #                     多线程并发读会破坏 HDF5 的进程级状态，轻则假
    #                     "NetCDF: HDF error"（打开成功、读数据失败），重则段错误，
    #                     而它被协议层的 except + continue 吞掉，只是**静默跳过那个
    #                     起报**。串行化解决了正确性，代价是 I/O 段完全排队、并发
    #                     收益基本归零；本配置的读盘量又恰好是大头。
    #                     进程形态下每个子进程有独立的 HDF5 状态，锁不争用、各读
    #                     各的；resident 观测仍靠 fork 前的预热 + 写时复制共享，
    #                     不会按进程数翻倍。代价：每块的指标结果要 pickle 回父进程。
    #                       24h 段（weather_ts_ens）      进程 × CPU 核数
    #                       6h 段（weather_ts_ens_prob）  进程 × CPU 核数（本来就是）
    #   n_workers        worker 数。**本配置写死**，理由是核数在这台机器上太容易
    #                    读错：框架的缺省已改走 affinity mask（``sched_getaffinity``，
    #                    认 cgroup/cpuset 配额），但 Slurm 不开 task/affinity 插件时
    #                    连它也会报回宿主机的 192。上一轮就是这么炸的——192 个进程
    #                    各物化一块 51 成员的预报场，几分钟内全被 OOM 打死，症状是
    #                    满屏 BrokenProcessPool，离真正的原因很远。
    #                    写多少：**以 ``nproc`` 为准**。内存要跟着核数一起算——单块
    #                    内存的主项是预报场（成员数 × 单块时效 × 格点字节数），按进程
    #                    数线性叠加；站点观测只有 0.67 GB 且写时复制共享，不是大头。
    #
    #                      单块读取集 = 采样时效 ∪ 往前预热的时效（plan._chunk_work）：
    #                        24h 段  {6,12,18,24}     × 51 成员 = 204 场 ≈ 1.7 GB
    #                        6h  段  {0,6,12,18,24}   × 51 成员 = 255 场 ≈ 2.1 GB
    #                      单场 8.2 MB 是 0.25° 全球（721×1440，float64）——锚点是
    #                      plan.py 里「15 天 × 51 成员约 25GB」那一句反推出来的。
    #                      于是 **n_workers × ~2 GB 就是申请内存的下限**：24 进程
    #                      要 ~50 GB。给不了就把这个数降到 8~12（超了核数本来也在
    #                      互相抢），或者给 forecast_reader 写一段稠密前缀的
    #                      ``lead_times`` 把时效砍半（内存跟着砍半）。
    #   chunk_days       一个工作块装几个**起报日**（缺省 1）——管起报跨度
    #   lead_chunk_days  一个工作块装几天**时效**（缺省 1）——管时效跨度，是单块
    #                    内存的**主旋钮**。两段的窗口不同（24h / 6h），但切窗是按
    #                    **采样时效**切的，1 天（= 4 个 6h 时效）对两段都成立。
    #                    取值 0 = 不切（整段时效一块，集合预报会把整段 × 51 成员
    #                    全物化，约 25GB，非必要别用）
    #   loads            角色驻留覆盖 {角色: 策略}，角色只有 observation / reference：
    #                      slice      每个块现读现弃
    #                      resident   整个 run 读一次驻留内存
    #                      window:W   按 W 天块滚动缓存（LRU），省相邻块的重复读
    #                      window     裸写不带数字 = 自动定窗；落成的值回写成
    #                                 "window:N" 记进 manifest 的 execution.loads
    #                    本配置只写 observation：站点观测的推导缺省就是 resident
    #                    （Diamond 是逐时文本，按块现读等于把全年观测读几千遍）。
    #                    reference 是 ref_probability，逐样本直读、没有整包，
    #                    **写它会直接报配置错**，所以这一项必须留空。
    #   resume           True 时已完成块的状态落 output_dir/.states/，重跑跳过
    #
    # 下面 loads 一行就是推导结果，写出来是为了好改。代价：写死之后不再自动
    # 跟随数据源——换了观测读取器要记得回来核对一遍。
    execution={
        "mode": "processes",    # 两段都走进程：threads 会把读盘全排在 hdf5_guard 的锁上
        # ⚠ 先跑 nproc 确认配额，再把这个数改成它（缺省虽已认 affinity mask，
        #   Slurm 不开 task/affinity 插件时仍会读成宿主机的 192）。
        "n_workers": 24,
        "chunk_days": 1,        # 一个块装 1 个起报日
        "lead_chunk_days": 1,   # 一个块装 1 天时效（= 4 个 6h 时效）
        "loads": {"observation": "resident"},
        "resume": True,         # 长时段正式跑建议开：中断后接着算
    },
)
