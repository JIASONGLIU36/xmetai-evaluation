# -*- coding: utf-8 -*-
"""FDP 降水检验（集合口径）：Fengqing tp × Diamond 站点实况。

流程 ``["weather_ts_ens_prob"]``：逐成员 → AROC / BS / BSS。评的是集合本身的
可靠性（排序能力、概率误差、相对气候基准的技巧），**不做集合平均** —— 均值会把
成员信息抹掉。

与 ``fdp_precip_single_fengqing`` 同源（数据根、时段、时效、执行全同），差别只在
pipeline：那份走 ``fdp_precip_ts``，评集合平均场的确定性命中/空报。
**改路径、改时段记得两份都改。**

⚠ 起报日和时效都被数据钉死了，和 ``fdp_precip_single_fengqing`` 同一套（详见那份
的 docstring）：``fengqing/`` 只有起报 20260820，``rain/`` 只有 0820 的 00Z~06Z，
所以起报日 20260820，真能出分的时效只有 **6**。换到服务器上数据齐了，一起放开。

⚠ ``_LEAD_TIMES`` 里那个 **12 不是笔误**：本流程带 ``time_window_accumulator``，
它得先拿 ``np.diff`` 反推时效步长，读取集只有一个时效时直接抛
"Need at least 2 time steps"（transforms/temporal.py:100）。框架对这个情形留的
口子是「读取集往前多带一步」（execution/plan.py:287 的 warmup），但那步只从
**已声明的**时效里取 —— 我们要的 6 前面是 0，而 fixture 里没有时效 000 的文件，
只能往后借一步。多出来的 12 不在实况覆盖里，观测窗口取不到，协议会记一条
WARNING 把它跳过：不产生样本，也不影响 6 的结果。

BSS 的气候基准：本配置**没给**外部参考（原因见 ``reference_reader`` 那行），走样本
自己的气候频率 r(1−r)，与 FDP 标准脚本 ``ensemble_verifier.py:692`` 同口径。给了
外部参考才会换成 mean((p_clim − o)²)，那是另一种基准，两边的 BSS 不能直接比。
"""
from xmetai_evaluation.configs.base import EvalConfig

# 数据根：**本机 tmp_data/ 就是它的镜像**，整份传上去就行（与
# fdp_rmse_single_fengqing 的 _DATA_ROOT 是同一个字面量）。两份配置里也是同一
# 份字面量，改一处记得改两处（没有抽公共模块，就是为了让单独一份拿出去也能
# 直接跑）。
_DATA_ROOT = "/workspace/data/shenzw/fdp_test_data"
_FENGQING_ROOT = f"{_DATA_ROOT}/fengqing"
_STATION_ROOT = f"{_DATA_ROOT}/rain"
# BSS 的气候概率参考：ref/MMDDHH.000（站号 + 与阈值一一对应的概率列）。
# 就是负责人给的那份，与 fuxi 那份指向同一个目录，换位置直接改这一行。
# ⚠ reader 只在这一层 glob（不递归），文件必须**直接**躺在下面，名字形如 082014.000。
#
# ⚠ 查表时刻：protocols.py:703 拿 valid_time 去查，而 reader 的文档头写明文件名是
#   **北京时**、HH 是 6h 窗的**结束时刻**。valid_time = init + lead + offset，本配置
#   offset=0 ⇒ 查表用的键是 UTC 时刻，会比北京时命名的文件早 8 小时。
#
# ⚠ 口径提醒：给了这个参考，BS_ref 就换成 mean((p_clim − o)²)（metrics/
#   probabilistic.py:313）；**不给**则退回 BS_ref = r(1−r)，r 取本时效本批样本
#   自己的观测事件频率（同文件 :319）。FDP 标准脚本 ensemble_verifier.py:692
#   算的正是后者 —— 要跟标准脚本出的 BSS 对表，就不该给这个参考。
_REF_PROB_ROOT = "/workspace/data/worm/r0/ref"
# 时效 6 是唯一出得了分的；多声明的 12 只为给累积器凑出步长，见模块 docstring。
_LEAD_TIMES = [6, 12]


def _execution():
    """``loads`` 里**不能**写 ``reference``：ref_probability 逐样本直读概率、
    没有「整包」可言，写了会被 ExecutionStrategy 直接判配置错
    （execution/strategy.py:219）。观测是站点，本来就默认整段驻留。"""
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


cfg = EvalConfig(
    name="fdp_precip_ens_fengqing",
    description="FDP 降水站点检验（集合，Fengqing × Diamond 站点）：AROC / BS / BSS",
    pipeline="weather_ts_ens_prob",

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
    # ⚠ 这一行**必须写**，不能吃缺省。模板 weather_ts_ens_prob 自己没有 options，
    #   于是 local_utc_offset_hours 落到 spec.py:182 的缺省 **8**；而兄弟配置那条
    #   fdp_precip_ts 的模板把同一个值写死成 **0**。两段评的是同一批站点、同一个
    #   预报，差 8 小时就是第二段一个样本都配不上（观测窗口取不到，静默跳过）。
    #   取 0 的依据：这份站点数据的文件名标签就是 UTC 有效时刻 —— 7 个文件从
    #   2026082000 到 06 的全国/华南雨量单调上升（华南 0.100 → 0.435 mm/h），
    #   是「清晨低谷爬向午后峰值」，不是北京时深夜该有的样子。
    options={
        "local_utc_offset_hours": 0,
        # ⚠ 这三项都是**对齐标准脚本**用的，别删。
        #   * weights=cos_lat：标准脚本给站点乘 w=cos(|lat|)
        #     （fdp/ensemble_verifier.py:1151、:1272 的 w_station）。不加，
        #     AROC 会整体高约 0.01 —— 实测同数据 ≥0.1 时 0.6339 vs 0.6246，
        #     加上权重后逐阈值对到 1e-4。
        #   * region=中国框：标准脚本的 filter_china_stations（15–55N/70–140E）。
        #     只剔掉 12 个站，对数字影响 <0.001，但口径要一致。
        #   * observation_window=reference：模板缺省是 complete（6h 窗口缺任一时次
        #     就整站丢掉），会少 98 个站（88039 对标准脚本的 88137），AROC/BSS
        #     相对差到 1.3e-3 —— 对表就只剩「量级对」的意义了。标准脚本是拿
        #     窗口末端那个文件打底、其余时次 fillna(0) 累加
        #     （fdp/ensemble_verifier.py:573-583），正是 reference 口径。
        #   模板 weather_ts_ens_prob 是照 fuxi 那套写的，三项都没带；
        #   兄弟配置的 fdp_precip_ts 模板里本来就带着（含 observation_window），
        #   所以那边不用写。这里写字面量而不是 import CHINA_REGION：这份配置能
        #   单独拿出去跑。
        "weights": "cos_lat",
        "region": {"lat": [15.0, 55.0], "lon": [70.0, 140.0], "name": "china"},
        "observation_window": "reference",
    },
    # ⚠ 冒烟阶段**先关掉**外部参考。不是路径错，是这份 fixture 用不了它：
    #   上面 offset=0 是被观测钉死的 —— rain/ 只有 0820 00Z~06Z，6h 闭窗只有
    #   valid_time = 0820 06:00 那一个凑得齐，其它 offset 下窗口整个落空、
    #   一个样本都配不上。而查表键 = valid_time 的 MMDDHH = **082006**，
    #   参考归档里没有这一时次（实测报「ref 缺少时次 082006」）。
    #   就算有也不能用：它是北京时的 00–06 窗，而我们真正评的是 UTC 01–06
    #   = 北京时 09–14，差 8 小时，拿到的是错槽位的气候概率。
    # 关掉之后 BSS 退回 BS_ref = r(1−r)（metrics/probabilistic.py:317），r 取本
    # 时效本批样本自己的观测事件频率 —— 正是 FDP 标准脚本 ensemble_verifier.py:692
    # 的口径，冒烟阶段反倒是对得上的那个基准。
    # 服务器上换成按北京时打标签的 Diamond 归档、offset 改 8 之后，取消下面这行的
    # 注释即可（那时 valid_time = 北京时 14 点，键是 082014）。
    # reference_reader={"type": "ref_probability", "root_dir": _REF_PROB_ROOT},

    start_date="20260820",   # 起报日；fengqing/ 里只有这一天
    end_date="20260820",
    limit=None,

    output_dir="/workspace/szwCode/evaluation_results/fdp_precip_ens_fengqing",
    # ⚠ 配置级 writers 是**替换**模板的，这里把它写全：
    # weather_ts_ens_prob → csv_long + probability_wide。漏了对应那项，报告里就没有那张主表。
    writers=["csv_long", "probability_wide"],
    log_level="INFO",
    execution=_execution(),
)
