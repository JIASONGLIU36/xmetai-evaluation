# xmetai-evaluation

气象模型**离线评测**框架：输入已经推理完成的预报产品 + 观测/再分析/气候态，输出统一的结构化指标与报告。

**只做评测，不做推理**：不加载模型权重、不训练、不调度推理。

```
xmetai-inference → 预报产品文件 → xmetai-evaluation → 评测结果与报告
```

## 能力边界

- 输入预报：NetCDF 格点场（FuXi 确定性 / FuXi 集合 / Fengqing）。
- 输入观测/参考：Diamond 站点观测、BABJ 台风路径报文、CRA40 再分析、ERA5 zarr、
  CRA 气候态、日序气候态、BSS 气候概率。
- 产品类型：确定性场、集合成员场、概率产品。
- 评测对象：格点/站点连续量、降水分类与概率、集合标定、**台风路径与强度**。
- 输出：`scores.csv` 长表、分类/概率宽表、台风逐场次路径表、JSON 快照、覆盖率；可选图件。

## 快速开始

```bash
pip install -e .
cd xmetai-evalation-v2
# 看有哪些评测功能
python -m xmetai_evaluation.cli --list-pipelines         

# 开始评测（配置可以是内置名，也可以是 .py 路径；数据路径由环境变量提供）
python -m xmetai_evaluation.cli --config weather_rmse_single_fgvp --log-file /workspace/szwCode/evaluation_results/weather_rmse_single_fgvp_ctrl/smoke_tp.log

# --log-file 可省略：省略后日志只打到控制台（INFO 级），不落文件；
# --config 可以更换，如 weather_rmse_single_fgvp

# 测试
python -m pytest tests/integration/test_fdp_fengqing_realdata.py -m realdata
```

## 目录结构

```text
xmetai-evaluation/
├── pyproject.toml                  # 打包 + CLI 入口 xmetai-eval
├── xmetai_evaluation/
│   ├── cli.py                      # 统一入口：加载配置 → 交给 Runner
│   ├── components.py               # 内置组件注册（按名字查表，不分类型分支）
│   ├── configs/                    # EvalConfig + 任务配置（部分用环境变量覆盖路径）
│   │   ├── base.py                 # EvalConfig 定义 + load_config()
│   │   ├── weather_ts_single_fgvp.py       # FuXi 确定性降水分类检验
│   │   ├── weather_ts_ens_fuxi.py       # FuXi 集合降水：24h TS + 6h 概率两段一趟跑完
│   │   ├── weather_rmse_single_fuxi.py     # FuXi 确定性连续量（RMSE/谱/ACC/活跃度）
│   │   ├── weather_rmse_ens_fuxi.py # FuXi 集合场（RMSE/CRPS/Spread/ACC/活跃度/谱/球谐带）
│   │   └── fdp_rmse_single_fengqing.py          # FDP 要素检验（z500 的 RMSE/Bias/ACC）
│   ├── core/                       # contracts / errors / registry / variables / logging
│   ├── execution/                  # 执行层：profiles / strategy / plan / loader / executor
│   ├── io/                         # gridded / layouts / station_reader / climatology_reader / netcdf_reader / base
│   ├── transforms/                 # interpolation / temporal / regrid
│   ├── metrics/                    # rmse / bias / acc / categorical / probabilistic / ensemble / spatial / specialized
│   ├── pipeline/                   # runner(唯一编排) / spec / pipelines / protocols / matcher
│   ├── output/                     # store / table（统一长表与落盘）
│   └── visualization/              # precipitation_plots / ts_report
├── ref/                            # 参考实现（只读档案，不入库、不参与运行）
├── evaluation_results/             # 评测产物（不入库）
└── reports/                        # 报告图件（不入库）
```

## 执行流程

一条链路，无第二套循环：

```
cli → load_config(EvalConfig) → PipelineSpec(流程模板+数据)
    → Runner(唯一编排) → execution(切块+并发+归并) → ResultStore 落盘
```

- **流程模板**（`pipeline/pipelines.py`）只声明「怎么算」：协议 + 变换链 + 指标 + 输出视图。
- **配置**（`configs/*.py`）只声明「算什么」：数据在哪、评哪段时间。
- **组件注册**（`components.py`）：新增数据源/指标只加注册项，不动 Runner。

## 配置

配置是一个 `EvalConfig` 实例（`configs/base.py`），核心字段：

| 字段 | 含义 |
|---|---|
| `pipeline` | 走哪套流程模板（必填）；写成列表则按顺序各跑一段，结果合并落同一个 `output_dir` |
| `forecast_reader` / `observation_reader` | 数据源：`{"type": <reader>, "root_dir": ..., "variable": ...}` |
| `reference_reader` | 参考场：气候态（ACC/活跃度需要）或气候概率（BSS 需要）。配置里给了、但某段的指标用不上时不会构建 |
| `start_date` / `end_date` | 评测时段（`YYYYMMDD` 或 `YYYYMMDDHH`） |
| `output_dir` | 结果输出目录 |
| `transform_options` / `metric_options` | 覆盖模板里的变换/指标参数 |
| `writers` | 覆盖模板的输出视图（默认 `csv_long`） |

最小示例：

```python
from xmetai_evaluation.configs.base import EvalConfig

cfg = EvalConfig(
    name="my_eval",
    description="FuXi 集合降水分类检验",
    pipeline="weather_ts_ens",
    forecast_reader={"type": "fuxi_ens", "root_dir": "/data/fuxi_ens", "variable": "tp", "step_hours": 6.0},
    observation_reader={"type": "station", "root_dir": "/data/station", "variable": "precipitation"},
    start_date="20250101",
    end_date="20251231",
    output_dir="evaluation_results/my_eval",
)
```

要注意 `transform_options` / `metric_options` / `options` 是**各段共用**的：配置里写 `{"time_window_accumulator": {"window_hours": 6}}` 会把每一段的窗口都改成 6。窗口属于"怎么算"，写在模板里（`pipeline/pipelines.py`）。多段的完整例子见 `configs/weather_ts_ens_fuxi.py`。

内置配置用环境变量覆盖数据路径与时段：`START_DATE` / `END_DATE` / `EVAL_OUTPUT` 各配置通用；
数据路径按配置各取所需——`FUXI_OUTPUT` / `FUXI_ENS_OUTPUT`（FuXi 确定性 / 集合）、`FENGQING_OUTPUT`、
`FGVP_OUTPUT`、`STATION_OBS` / `STATION_LIST`（站点观测与白名单）、`BSS_REF`（气候概率参考）、
`ERA5_CLIMO`（ERA5 气候态）、`FDP_CRA_ROOT` / `CRA_CLI_ROOT`（FDP 的要素场与气候态参考）。

### 批量评测（一份文件评多个模型）

配置模块也可以定义 `cfgs = [EvalConfig(...), ...]`（复数）而不是 `cfg`：

```bash
xmetai-eval --config my_multi_config
```

框架按列表**顺序**逐个执行（不并行：每个 run 内部已经吃满 n_workers）。
单个模型失败会记日志并继续跑后面的，结束时统一报成败、任一失败退出码为 1。
观测/气候态/时段/指标等共用项写一份、预报源按模型换（用一个 `MODELS`
表循环生成 `cfgs` 最顺手）；每个模型各落各的 `output_dir`，产物与单模型
配置完全同构，下游报告与对拍不用改。

### 分纬度带评估（regions）

两类协议都支持在 `options` 里声明纬度带，逐带出分：

```python
options={
    "sample_by": "init_lead",
    "regions": {
        "tropics": {"lat_min": -20, "lat_max": 20},
        "nh_extratropics": {"lat_min": 20, "lat_max": 90},
    },
}
```

格点协议（`grid_valid_time`）：每个 (起报, 时效) 样本除全球行外再逐带各出一行。
站点协议（`station_valid_time`，TS 系列）同样支持，站点按**站点纬度**整站归带
（读 `station_lat`，没有就找 `lat`；两者都取不到直接报错，不静默回退成全球分）。

长表里用 `region` 列区分（全球行该列为空）。实现口径：带边界是闭区间（正好压线
的格点/站点**两条带都算**）；数据不裁、只把 `valid_mask` 收缩到带内，标量指标
（RMSE/Bias/ACC/活跃度等按掩码加权或筛点的）逐带出分；谱类
（`zonal_spectrum`/`spherical_bands`/`spectrum`）与 FSS 需要**完整空间场**，
不分区、只出全球行。不写 `regions` 键则完全回到只有全球行的老行为。

报告侧：连续场走「附 L 分纬度带结果」，TS 走「附 L 分纬度带结果（可选）」——
两块都**只留带行、全球行不参与**（各带平均再平均 ≠ 全球平均），图走独立编号
`图 L1`，加不加都不动正文图号。`weather_rmse_single_fuxi` 已启用经典三分带、
`weather_ts_single_fgvp` 与 `weather_ts_ens_fuxi` 已启用中国四带（南方 / 长江中下游
/ 华北 / 东北），可作模板。集合那份的 `options` 是**两段共用**的，所以 24h 的 TS 段
与 6h 的概率段都会出带行；报告的「附 L」只汇总 TS 那段，概率段的带行落在
`scores.csv` / `diagnostics/probability_wide.csv` 里（第六节本来就没有渲染器）。

## 执行策略（并发与数据加载）

评测的本质是三个数据集做差：**预报（驱动集）、观测（配对集）、参考（辅助集）**。
工作块永远从预报侧切，切法是两个维度的**叉积**：起报日（默认一天一块）× 时效
（默认一天时效一块）；观测和参考要读多少，由块内的起报与时效推出的有效时刻
跨度决定。并发形态和数据驻留不靠人拍脑袋，由两层推导：

1. **能力画像**（`execution/profiles.py`）：这一段的指标族联合起来要不要
   参考源 / 成员 / 整场、计算轻重——误差族（rmse/bias）一种策略，距平族
   （acc/activity）要气候态是另一种，集合族（crps）与谱族（spectrum）是重活。
2. **执行策略**（`execution/strategy.py`）：画像 × 协议 × 数据源规模推导
   出并发形态（`auto`：单块=串行、重指标=进程、轻指标=线程）、块大小和
   每个数据角色的驻留方式：

   | 驻留策略 | 语义 | 缺省用于 |
   |---|---|---|
   | `slice` | 每块现读现弃 | 预报（恒为此策略）、格点实况（`cra`/`era5_zarr`） |
   | `resident` | 整个 run 读一次驻留 | 站点观测、气候态参考场（CRA CLI_6HOUR / 日序） |
   | `window:W` | 按 W 天块滚动缓存，保留当前块 ±1（回跳要用） | 格点实况（`cra`/`era5_zarr`）**推荐**显式改成它，推导缺省仍是 `slice` |

   格点实况的推导缺省是 `slice`，但**配 `window` 基本总是更好**：块序会让观测读
   反复回跳（原因见本节末「`loads` 的自动定窗」那段），`slice` 下每个观测日被重读
   十几遍。现有 era5 配置一律写 `window:16`。代价是窗块真占内存且**每个子进程各
   一份**（不像 `resident` 那样 fork 共享），所以它是和 `n_workers` 绑在一起的一笔账。

   数据源按「角色 × 类」归类（16 个注册名），缺省驻留策略由类决定：

   | 角色 | 类 | 注册名 | 缺省驻留 |
   |---|---|---|---|
   | **预报**（驱动集） | 格点预报（确定性/集合） | `fuxi` `fuxi_ens` `fengqing`（及 `_phys` 单位变体） | 恒 `slice`，不进加载层 |
   | **观测**（配对集） | ① 站点观测 | `station`（`diamond_station` 别名） | `resident` |
   | | ② 格点实况 | `cra` `era5_zarr` | `slice` |
   | **参考**（辅助集） | ① 整场气候态 | `climatology` `daily_climatology` | `resident` |
   | | ② 逐站概率 | `ref_probability` | 不进加载层（协议逐样本直读，不可配 `loads`） |

配置里用 `execution` 字典覆盖个别键（不写 = 用推导值，键全部字面量）：

```python
execution = {
    "mode": "auto",       # serial / threads / processes / auto
    "n_workers": 4,
    "chunk_days": 1,      # 一个工作块装几个起报日
    "lead_chunk_days": 1, # 一个工作块装几天时效；0 = 不切（整段时效一块）
    "loads": {"observation": "resident", "reference": "window:30"},
    "resume": False,      # True：已完成块的状态落 output_dir/.states/，重跑跳过
}
```

### 并发形态：serial / threads / processes

| 形态 | 结构 | 数据驻留 | 适合 |
|---|---|---|---|
| `serial` | 单进程顺序跑块 | 每块现读现弃 | 冒烟、单块 |
| `threads` | **1 个进程**内 N 线程并发块 | resident / window 全局只有 1 份（loader 有锁）；预报各块自读 | 轻指标（站点 TS 等），典型加速 2~3× |
| `processes` | 进程池一次建好活到跑完；块按**时间连续分段**，一段固定一个子进程顺序处理 | fork 下 resident 由父进程预热、子进程 COW 共享 ≈ 1 份；spawn 下每个子进程各持一份副本 | 重指标（谱/CRPS/FSS），真并行 |

- **threads 的原理与优势**：GIL 让纯 Python 逻辑串行，但 numpy 计算、
  zarr/NetCDF 读盘都释放 GIL，轻指标能蹭到 IO+部分计算的并行。它独有的
  姿势是把窗宽开大（如 `window:20`），5 个并发的相邻日块共用同一个观测
  窗口块——数据全局只有一份，内存与并发数无关。
- **processes 的内存账**：单进程 = 块工作集（预报窗口 + 观测切片 + 配对
  批次，随 `chunk_days × lead_chunk_days` 涨）；总量 ≈ resident 驻留 ×1 +
  工作集 × `n_workers`。**`n_workers` 由内存上限决定，不是核数**。单块内存的
  主旋钮是 `lead_chunk_days`（时效跨度，集合预报上省得最多），`chunk_days`
  管起报跨度。Linux fork 下 resident 只存 1 份（COW 共享）；spawn 平台每个
  子进程各存一份，`n_workers` 要按这个算内存。
- **auto**：单块 = serial、重指标 = processes、轻指标 = threads。

要点：

- **块是唯一并行单位**，块内数据用完即弃、块间零共享；进程模式下块按
  **时间连续分段**，一段固定给一个子进程顺序处理——观测 `window` 缓存
  只有在"同一进程先后处理相邻块"时才能命中，随机抢块会让它形同虚设。
  fork 下由父进程预热 resident 缓存、子进程写时复制共享；spawn 平台没有
  共享，但角色声明是可序列化的，子进程会反序列化一份 loader 副本自建
  缓存（段内窗块照样复用，代价是内存 ×段数）。两条路都不改变结果。
- **窗块预取**：非 threads 形态下，读满一个 `window` 块后，后台线程会把下一个
  块提前读进来。完整口径（开关条件、预测依据、三道防白读的闸、值不值的判据）
  见上面「[数据预取](#数据预取把下一个窗块提前读进来)」。
- **块级进度看哪行**：进程模式下父进程要等**整段**返回才拿到结果（那条 tqdm
  因此全程不动，`.states/` 也是段末才一次性落盘），所以块级进度由子进程自己报：
  每 `PROGRESS_EVERY` 块（`execution/executor.py`，默认 50）打一行

     段 2/5 进度 150/1114 块（已用 540s，3.60s/块，按此速率还需 3470s，最新块 20250312_L24-48）

  5 段并行就是 5 条这样的流，段末那块必打。速率与剩余时间只看本段已跑部分的
  均值，前慢后快时偏保守。
- **日志量的口径**：逐块流水（`起报时间`、`预报/实况读取完成`、`配对完成`、
  `同网格按索引对齐` 这些每块各一次的行）是 DEBUG；INFO 只留块级进度、段级
  汇总与真正的告警。要看逐块细节用 `--log-file`（文件 handler 恒为 DEBUG，
  内容一条不少）。每块都会重复的静态 WARNING（"这些变量在预报或观测中不存在，
  已跳过"）按消息去重，每进程只打一次——缺哪些变量是配置与数据源决定的事实，
  重复 5568 遍只会把真告警淹掉。
- **切块不改变结果**：所有内置指标的状态都是和式（sum/count/moment），
  归并可交换；grid 协议"同一 valid_time 只认最新起报"的口径由
  `duplicate_key_policy="replace"` 在跨块归并时重现——比的是坐标里的
  `init_time` 而不是块序，因为同一个有效时刻可以由 (早起报, 长时效) 和
  (晚起报, 短时效) 两条路径够到，最新起报配的反而是最小 lead。
- 逐段执行口径写进 `manifest.json` 的 `execution` 键（形态/块数/画像），
  结果可比的前提是知道它是在什么策略下算的。

### 时效维：延伸期与集合预报的内存旋钮

`chunk_days` 只切起报，**不切时效**——一个块仍要把整段时效跨度全物化。延伸期
预报（15 天 × 51 成员 × 0.25° 单要素）单块就是几十 GB，这时要调的是
`lead_chunk_days`：

- 时效窗落在**采样格点**上，不是原始时效跨度。站点协议是滑动累积窗，"可评
  样本"只落在完整窗口的末端（24h 窗 → lead 为 24 的倍数），按原始跨度切会切出
  一个样本都没有的空块；按采样格点分组，空窗结构上不会出现。
- **读取集 ≠ 采样集**。累积窗要窗口完整才出值，所以读取集会从窗内首个样本往前
  多带 `窗口步数 − 1` 个时效作为**预热**；读取集可以跨窗重叠，但采样集是
  (起报, 时效) 的纯划分，重复读不会重复计数。预热只从源里实际存在的时效取，
  时效是稀疏显式声明（如 `[24, 48, 72]`）时中间步拼不出累积窗，这不是切块造成的。
- `lead_chunk_days * 24` 小于累积窗口时会打 warning：窗比一个累积窗还窄，每个块
  都要往前读预热，读放大且样本稀疏。
- `lead_chunk_days: 0` 是逃生口：整段时效一块，与不切时效的结果逐位一致。

集合成员**不**单独切：CRPS 这类指标需要同一 (起报, 时效) 的全部成员，时效窗
一开，单块的成员场自然就小了。

`loads` 里的 `window` 可以不带数字（`{"observation": "window"}`）：计划层自动定窗宽
= **整段时效跨度**（`max(leads)/24` 天）+ 预热 + 块内起报跨度（`chunk_days` 天），
threads 下再加 (并发数−1)×`chunk_days` 的相邻日跨度（processes 段内串行，这项为 0）。
落成的具体值（如 `"window:16"`）会回写进 manifest 的 `execution.loads` 供核对。

`window:W` 里的 **W 是上限**：实际用 `min(W, 自动值)`。比自动值大不会减少重读、只多占
内存（进程模式下窗块是每个子进程各一份）；比它小则是主动拿内存换 IO。两个值不一样时
日志会另打一行说明实际收到多少。

> ⚠ **这条 2026-09 反转过，别按旧口径理解。** 窗宽要按**整个时效跨度**定，不是按
> 单块的时效窗宽（`lead_chunk_days`）定。原因在块序：块序是「起报日外层、时效窗
> 内层」，一个起报日组要顺次扫完全部时效窗（观测日 D … D+L），下一组又从头开始
> （D+1 …），所以观测读是"向前扫 L 天、再回跳 L−2 天"。窗宽 ≥ 整个时效跨度，回跳
> 时上一轮读过的窗块才还在缓存里（loader 另外保留当前块 **±1** 块），一个观测日整
> run 只读一次。
>
> 旧写法按单块时效窗宽（1 天）+ 预热 + `chunk_days` 定，算出来只有 2~3 天：回跳落空，
> 读放大约 L 倍，`window` 相比 `slice` 几乎没有改善。loader 的块保留同样是配套改的
> ——只留当前块、或只留"前一块"，回跳都会落空；必须**前后各留一块**。这三处（定窗
> 基数、±1 保留、预取闸门）是一组，改一处要一起看。

`window` 切的是**时间跨度**不是文件：窗口块调 `builder((start, end))`，
reader 自己决定读文件哪些部分（NetCDF/HDF5 支持索引级部分读；日序气候态
按 day-of-year 只挑请求日子 ± 平滑窗的几行，还分纬度带读）。大 NC 走
window 从不需要整文件进内存；同一进程先后处理相邻块直接命中窗口缓存，
全年顺序跑每段数据基本只从盘上读一次。

### 块大小什么时候调：`chunk_days` / `lead_chunk_days`

**调大省的是"每块的固定开销"，不是 IO，也不是计算。** 每块都要从头重建一遍东西
（`executor.py:collect_chunk_states`）：

```python
register_builtin_components()
registry.build(READER, 预报)   registry.build(READER, 观测)   registry.build(READER, 参考)
transforms = {...build(TRANSFORM)...}      metrics = [...build(METRIC)...]
```

再加上 reader 构造时开 zarr/GRIB、catalog discover、预报目录索引。这一坨与块内实际
算了多少无关，块数按 `chunk_days` × `lead_chunk_days` 线性降，它就线性降。

**IO 不会跟着降。** `window` 生效后每个观测日整 run 只读一次；块变宽 → 每块跨度变宽、
块数同比例变少，**总读取量是常数**（`slice` 同理）。

| | 块数（era5 det：348 起报 × 16 窗，`1/1`） |
|---|---|
| `chunk_days: 1`, `lead_chunk_days: 1` | 5568 |
| `lead_chunk_days: 2` | 2784 |
| `chunk_days: 2` | 2784 |
| `2 / 2` | 1392 |

**什么时候该动：**

| 症状 | 动作 |
|---|---|
| 每块耗时很短、日志读盘占比又高 | 固定开销是大头 → 调大 `lead_chunk_days`（1 → 2 起步） |
| 子进程空转，块数 < `n_workers` | 块数太少 → 调大 `lead_chunk_days`（这条也正是段数 = `min(n_workers, 待跑块数)` 的后果） |
| 内存吃紧 / 要保 `n_workers` | 调小 → 单块工作集线性降；`lead_chunk_days` 是**单块内存的主旋钮** |
| 想改块大小但不想动别处 | 别碰 `chunk_days`，见下面那个坑 |

**代价三条**：单块内存线性涨；负载均衡变差——段数 = `min(n_workers, 块数)`，块数远大于
`n_workers` 时段数不变、只是每段块数变少，块数掉到 `n_workers` 附近才会空转（与下面
「时效切块的调度」那条一致）；resume 粒度变粗（崩一次重算的块更大）。

**怎么判断。** 两个观测点：日志的 `读盘占比 … = ZZ%`，和 tqdm 报的**每块耗时**

    段 2/5 进度 150/1114 块（已用 540s，3.60s/块，…）

每块几秒钟就完了、读盘占比又高 → 固定开销是主要成本，调大立竿见影；反过来块内本来就
在算谱、算 CRPS，固定开销占比小，调大收益有限。

> ⚠ **改 `chunk_days` 会抬高自动定窗的最小值**（`_auto_window_days` 里 `+ chunk_days`
> 那一项），而你写死的 `"window:16"` 会被 `min(16, 新最小值)` 夹住、**不会自动跟着涨**。
> `chunk_days: 2` 时最小值变 17、仍用 16，量级无害；调到 8（最小值 23）就明显偏小了。
> 要一起调就把 `loads` 改成裸写 `"window"` 让它自己跟。

### 时效切块的调度

切完之后这张块表怎么排、怎么发给并发单元、中断了怎么续——都只影响**顺序与开销**，
不影响分数。

**块表怎么排。** 两维叉积，块序固定为「起报组升序 → 组内时效窗升序」：

```
20250101_L6-18    20250101_L24-42    …  20250101_L342-360    ← 起报日 1 的全部时效窗
20250102_L6-18    20250102_L24-42    …  20250102_L342-360    ← 起报日 2
…
```

`chunk_id` 形如 `20250101_L24-48`（窗内只有一个采样时效就写成 `L6`）；`chunk_days > 1`
时起报段写成 `20250101-20250103_L0-120`。这个 id 同时是 resume 状态的文件名，
**必须带时效段**，否则不同时效窗的状态会互相覆盖。

**每块带两份时效清单**，都写进这一块的子声明：

| 清单 | 内容 | 落到子声明的 | 给谁用 |
|---|---|---|---|
| `read_leads` | 窗内采样时效 ∪ 往前预热的时效 | `forecast.params["lead_times"]` | 预报 reader 的文件过滤条件 |
| `sample_leads` | 只有窗本身 | `forecast.params["sample_leads"]` | 协议决定哪些 (起报, 时效) 出分 |

写进去是**覆盖**不是合并：留着一份全时效的 `lead_times` 会让每个块把整段时效读回来，
时效切分就白做了。预热长度 = `(window_hours / step_hours − 1) × step_hours`，只在流程
含 `time_window_accumulator` 时非 0（24h 窗 + 6h 步长 → 18h）。

时效窗分组锚在 lead=0、按采样点归属（`int(lead // (lead_chunk_days × 24))`）——空窗
结构上不出现。空块 `processed == 0`，会被判成**失败块**进 `manifest["failed_chunks"]`，
resume 下每次重跑再失败一次；而它什么都不产出、不改变 `scores.csv`，所以只比分数
是抓不到的。

**与并发形态的配合**：

- **threads** 把块表按序提交（哪个线程先空谁领下一个），并发的几个块时间相邻、共用
  同一个 loader 的 resident / window 缓存，所以自动定窗要把并发跨度算进去。
- **processes** 把块表按**连续段**切开，一段一个子进程顺序跑完（段内顺序 = 块序），
  这样同一段里相邻块的观测跨度大体相邻，滚动窗缓存才有机会命中；把块逐个丢进池里
  随机抢，每块都要为不相邻的跨度重新物化窗块，缓存形同虚设。
- 段数 = `min(n_workers, 待跑块数)`。所以 `lead_chunk_days` 调小、块数变多时，
  **变的是单块内存，不是并行粒度**（段数不变，每段的块数变少）。反过来块数少于
  `n_workers` 时会有子进程空转，该调大 `lead_chunk_days`。

**中断续跑**（`resume: True`）：完成块的状态落在
`output_dir/.states/seg{流程段号:02d}_{chunk_id}.pkl`。`seg` 前缀是 `pipeline` 列表里的
**流程段**序号（只有 `pipeline=["a", "b"]` 这种多段配置才会出现 `seg02`），与上面的**进程**
分段无关——进程分段不进文件名。重跑逐块查、命中就跳过；全部命中时整段直接跳过（连
resident 预热和进程池都不建）。

`chunk_id` 只由起报组和时效窗决定，**不含 `limit`、不含 `n_workers`**，所以先把
`limit` 调小冒烟跑一遍、再改回 `None` 正式跑，冒烟算过的块会原样命中（见下面「跑之前
怎么估」）。反过来，改 `lead_chunk_days` 重跑**不会**复用改之前的块——`chunk_id` 变了
就是另一批块。这是有意的：口径不同不该混算。

**改 `lead_chunk_days` 不该改变任何分数**：采样集是 (起报, 时效) 的纯划分，每块只对
自己的采样点累加，归并按和式相加。回归测试
`tests/integration/test_chunked_equivalence.py` 钉的就是这条——一块跑完 vs 逐日起报切
vs 再按时效切窗 vs 换并发形态，`scores.csv` 必须逐行一致，且 `manifest` 里不许有
失败块。

### 数据预取：把下一个窗块提前读进来

`window` 角色的读盘可以和块内计算叠起来：读完当前覆盖的块之后，后台线程把
**按日期顺序的下一个块**预读进待用槽；下一个请求到了直接命中，省掉一次同步读。
实现全在 `execution/loader.py` 的 `_schedule_next_block` / `_prefetch_loop`。

**什么时候开。** `RunLoader(prefetch=...)` 由计划层按并发形态定：`processes` 与
`serial` 开，`threads` 关。threads 下多个块并发读同一个 loader，各自排一个预取只会
互相顶掉；而且它本来就有 N 路并发读，轮不到预取补空档。另外**只有 `window` 角色会排**
——`slice` 没有"下一个块"的概念，`resident` 整 run 只读一次。

**预取哪个块。** `max(本次请求覆盖的块号) + 1`。依据是请求跨度随时间**单调向后延伸**
（时效窗推进抬 `lead_max`、起报组推进抬起报），下一次要读的块几乎总是当前最大块号 +1。
块号不存在（已经是最后一块）就不排。

**结构：一条队列 + 一个线程 + 每角色一个待用槽。**

```
_schedule_next_block()        在物化路径里同步调用
    ├─ 过三道闸（见下）
    ├─ 懒启后台线程（daemon，全 loader 只 1 个，名 runloader-prefetch）
    └─ queue.put((role, spec, 块号, 块跨度))

_prefetch_loop()              后台线程
    while True:
        role, spec, index, span = queue.get()
        bundle = self._build(spec, span)      ← 同一个 builder
        if 槽里没有更新的块: 槽 = (index, bundle)
```

预取走的是**同一个 builder**，所以它花的读盘时间同样计入 `builder_seconds`，读来的块
也照样算进日志里那个「物化 N 次」——这是有意的，让"预取多读了几次、值不值"可测。

**领取。** 请求进来先看待用槽：块号还在本次请求里、且窗块缓存中没有，就直接搬进窗块
缓存（跳过同步读）；块号已经落到请求后面（`< min(wanted)`）就扔掉；其余留在槽里等下次。

**三道防白读的闸。**

1. 槽里已经是同一个块 → 不重排。
2. 该块**已经在窗块缓存里**（±1 保留正好把它留着）→ 不重排。没有这道闸，预取会在
   **每个请求**上重读一次"当前块的下一个块"——±1 保留恰好让目标块常驻。
3. 后台线程读得慢、期间又排了更新的块 → 旧结果丢掉（槽只放一个）。

**代价。** 内存每角色 +1 个窗块（稳态窗块缓存是当前块 ±1 共 3 块，加预取最多 4 块）。
**正确性零代价**：预取不参与计算、不改变任何结果，猜错只是白读一次，真正的请求到了
按正常路径物化；预取失败只记 DEBUG、不抛。

**值不值。** 上限收益 ≈ `min(1, 读盘/计算)`——计算占九成时它只值一成。看日志那行

    读盘占比：builder 累计 X.Xs / 容量 Y.Ys（… × N worker）= ZZ%；物化 N 次、缓存命中 N 次

`ZZ%` 只有个位数就说明读盘本来不是瓶颈，预取白占内存；关掉的方式是把 `plan.py` 里的
`prefetch=mode != "threads"` 改成 `False`。

> **一条必须守住的不变量**：预取线程绝不能在 **fork 之前**启动过。`threading.Lock`
> 不可重入、fork 又不复制持锁的线程，父进程若在 fork 前起过预取线程并恰好持锁，
> 子进程里那把锁就是死的。现在靠"父进程只 `loader.warm()` 读 `resident`、从不读窗块"
> 保证。spawn 平台无此问题（`__setstate__` 会把线程与队列重建）。

### 跑之前怎么估

正式跑一个全年段之前，要能先答出三个数：**切多少块、单块读多少时效、要吃多少内存**。
前两个是纯算术（计划层本身也不读数据，日志里「N 个起报切成 M 个工作块」那行就是结果），
第三个只能实测。

**一、块数。** 四步，从时效列表倒推：

1. **时效列表**。没显式声明 `lead_times` 时，从 catalog 索引拿最大时效 `lead_max`，按布局
   步长 `step_hours` **从 0 铺起**：`0, step, 2·step, …, lead_max`，共 `floor(lead_max/step)+1`
   个。这是真实时效的**超集**——有的布局带 0 时效文件（valid = 起报时刻本身），漏了它
   驻留缓存会静默缺一个有效时刻；多出来的时刻只是没人用，无害。
2. **采样集**。`station_valid_time` 只认「完整累积窗的末端」（`lead ≥ window_hours` 且是其
   整数倍）；`grid_valid_time` 逐有效时刻配对，**全部时效都是采样点**。协议不同，同一份
   时效列表能差出十几倍。
3. **时效窗数** = 采样点按 `int(lead // (lead_chunk_days × 24))` 分组的组数。分组锚在 lead=0
   上（空窗结构上不出现），代价是**首尾两组通常不满**——别拿 `max(lead) / 24` 直接估，
   在 grid 协议下会多算一组。
4. **块数** = `ceil(起报日数 / chunk_days) × 时效窗数`；**段数** = `min(n_workers, 待跑块数)`。

拿 `weather_rmse_single_fuxi` 那类配置（`step_hours=6`、lead 到 360h、grid 协议、
`chunk_days=1`、`lead_chunk_days=1`）走一遍：

| 步 | 算式 | 数 |
|---|---|---|
| 时效列表 | `0, 6, …, 360` | 61 |
| 采样集 | grid 协议 → 全取 | 61 |
| 时效窗 | `int(lead//24)`：第 0 组 4 个（含 lead 0）、第 1–14 组各 4 个、第 15 组 1 个（只有 360） | **16** |
| 块数 | 348 个起报 × 16 窗 | **5568** |
| 段数 | `min(5, 5568)` | **5**（1114/1114/1114/1113/1113） |

首窗 4 个、中间各 4 个、末窗 1 个——所以日志里 `lead_time` 出现 4 / 3 / 1 三种值都是对的：
`1` 是末窗，`3` 是首窗里 lead=0 那个文件不存在、被 reader 过滤掉了。

**二、单块读多少时效** = `read_leads` = 窗内采样时效 ∪ 往前带的预热。预热只在流程含
`time_window_accumulator` 时非 0，长度 `(window_hours / step_hours − 1) × step_hours`。上面那份
配置的预报 `tp` 本身就是 6h 累积量、流程里没有累加器，所以预热为 0，首窗只读
`[0, 6, 12, 18]` 四个——日志里「单块读时效 4 个」就是它。

**三、内存。** 先分清哪些是常量、哪些是倍数：

```
峰值  ≈  Σ(驻留角色的常驻)   +   n_workers × 单块工作集
          ↑ 与块数、worker 数无关      ↑ 随 chunk_days × lead_chunk_days 线性缩
```

- **驻留部分**由 `loads` 决定，跟切块无关。`resident` 是整段读一次（fork 下父进程预热、
  子进程 COW 共享 ≈ 1 份；spawn 平台每个子进程各一份，×段数）。**切时效省不到它**——
  想让它跟着降，得把 `loads` 改成 `window:`。
- **单块工作集** = 预报窗口 + 观测切片 + 配对批次，三者都随 `chunk_days × lead_chunk_days`
  线性缩。量纲上：0.25° 全球单要素单时次 ≈ 4 MB（721 × 1440 × 4B），乘上要素数和
  `read_leads` 个数是**下界**；真实峰值更高，因为 `_combine` 会有一段中间帧与合并结果并存
  （≈2–3 倍），谱类指标还要额外的 FFT 工作数组。
- 所以 **`n_workers` 的实际卡口是内存除以单块工作集，不是核数**。

估出来的数别当真，实测只差一次小跑。启动后看容器的 cgroup 计数：

```bash
echo "上限: $(cat /sys/fs/cgroup/memory.max)"
echo "已用: $(cat /sys/fs/cgroup/memory.current)"
```

`weather_rmse_single_fuxi` 全年的实测账（`resident` 气候态 + 5 worker）：

| 项 | 实测 |
|---|---|
| 气候态 `resident` 预热 | **67.25 GB**、991.7 s |
| 单 worker 工作集 | **≈ 2.0 GB**（(77.31 − 67.25) / 5） |
| 5 worker 峰值占用 | **77.31 GB**，占 240 GB 上限的 32% |

配置注释里那些数字（「~96G」之类）是**估的**，实测比它小 43%；要用就现测，别抄注释。

盯的是**它会不会随时间单调爬**：观测配 `slice`（现读现弃）时应该稳在常驻 + 工作集附近
小幅波动，明显往上走是句柄没释放，得停下来查。

**先冒烟再正式。** `limit` 只截前 N 个起报，而 `chunk_id` 不含 `limit`，所以小跑算过的块在
正式跑时原样命中（前提是 `resume: True`），不会白算——这就是配置里 `limit=None` 旁边那句
注释的由来。反过来，**改 `chunk_days` / `lead_chunk_days` / `window_hours` 之后重跑不会复用**，
`chunk_id` 变了就是另一批块。

## 已注册组件

| 类型 | 注册名 |
|---|---|
| Reader | `fuxi`、`fuxi_ens`、`fengqing`（及 `fuxi_phys`/`fuxi_ens_phys`/`fengqing_phys` 单位变体）、`cra`、`era5_zarr`、`station`（`diamond_station` 别名）、`babj`（BABJ 台风路径报文）、`climatology`、`daily_climatology`、`ref_probability` |
| Transform | `grid_to_station`、`time_window_accumulator`、`ensemble_mean` |
| Metric | `rmse`、`bias`、`acc`、`acc_uncentered`、`ts_score`、`ensemble_probability`、`crps`、`spread_error`、`fss`、`activity`、`spectrum`、`zonal_spectrum`、`spherical_bands`、`track_error`、`track_error_ens` |
| Protocol | `station_valid_time`（插值到站点，按有效时刻配对）、`grid_valid_time`（插值到实况网格）、`typhoon_track`（链式诊断台风中心，对 BABJ 配对）、`typhoon_track_ens`（同上前者逐成员，批次带成员轴） |
| Writer | `csv_long`（始终写出）、`coverage`、`details`、`json`、`categorical_wide`、`probability_wide`、`spectrum`、`typhoon_cases` |

## 评测能力清单

每行是一个可直接跑的评测能力（流程名 `pipeline`），输入/输出文件相对 `output_dir`。

流程与配置按三大业务块统一前缀命名：`fdp_`（业务天气评测，参考 `ref/fdp`）、`weather_`（天气模型验证，参考 `ref/tiqnqi` 两个库）、`clim_`（气候，参考 `ref/qihou`，待落地）。fdp 与 weather 的连续量/集合指标有重叠，属正常——两者是不同业务线，共用同一套 metric 实现。

「评估指标」一列给的是 **`scores.csv` 的 `metric` 列取值**（即过滤长表用的那个字符串），
括号里是该指标在表里变化的坐标轴（阈值 / 时效 / 窗口 / 邻域窗口）。
标注「明细」的是**不进 `scores.csv`** 的诊断量，含义见下面「评估指标说明」。

「执行画像」一列是这段指标族的**联合资源需求**（`execution/profiles.py` 推导）：
重 = 有谱/邻域/逐成员积分类重活，参考 = 有指标要气候态/概率参考，成员 = 有指标
要原始集合成员，整场 = 有指标要完整未插值场。`auto` 模式据此选缺省并发形态
（重 → processes、轻 → threads、单块 → serial），配置里的 `execution` 可覆盖。

表里那些数据源名（`fuxi`、`era5_zarr`、Diamond 站点…）的目录怎么摆、文件叫什么、
变量与单位是什么，见下面「评测数据与格式」。

| 评测能力 | 输入数据 | 输出 CSV | 评估指标（`metric` 列取值） | 执行画像 → 缺省形态 |
|---|---|---|---|---|
| **确定性降水分类检验**<br>`weather_ts_det` | 确定性格点降水预报（`fuxi`，tp）+ Diamond 站点降水观测 | `scores.csv`、`diagnostics/categorical_wide.csv` | `ts`、`pod`、`far`、`miss_rate`、`frequency_bias`<br>逐 阈值（0.1/10/25/50/100 mm）× 24h 时效 | 轻 · threads |
| **集合降水分类检验（24h）**<br>`weather_ts_ens` | 集合格点降水预报（`fuxi_ens`，tp）+ Diamond 站点降水观测 | `scores.csv`、`diagnostics/categorical_wide.csv` | `ts`、`pod`、`far`、`miss_rate`、`frequency_bias`（集合平均场）<br>逐 阈值（0.1/10/25/50/100 mm）× 24h 时效；明细 `hits`/`misses`/`false_alarms`/`correct_negatives`/`n_pairs` | 轻 · threads |
| **集合降水概率评分（6h）**<br>`weather_ts_ens_prob` | 集合格点降水预报（`fuxi_ens`，tp）+ Diamond 站点降水观测 + 气候概率参考 `ref_probability` | `scores.csv`、`diagnostics/probability_wide.csv` | `aroc`、`bs`、`bss`（逐成员概率，不做集合平均）<br>逐 阈值（0.1/4/13/25 mm）× 6h 时效；明细 `BS_ref`/`base_rate`/`n_points` | 重·参考·成员 · processes |
| **确定性连续量检验**<br>`weather_field_scores` | 格点场预报（`fuxi`，z500 等）+ 格点实况 + 气候态（ACC/活跃度必需） | `scores.csv`、`diagnostics/spectrum_{var}.csv`、`diagnostics/spectrum_by_init.csv` | `rmse`、`acc`、`activity_ratio`、`activity_forecast`、`activity_observation`、`activity_bias`、`spectrum_power_ratio`（纬向谱）、`spherical_band_power_forecast/observation/ratio`（球谐带，group 列分频带）<br>逐波数谱曲线另出宽表 | 重·参考·整场 · processes |
| **集合连续评分**<br>`weather_ens_crps` | 集合格点场预报（`fuxi_ens`）+ 格点实况 | `scores.csv` | `crps`、`spread`、`rmse`（集合平均场）、`spread_error_ratio` | 重·成员 · processes |
| **集合场检验**<br>`weather_ens_field_scores` | 集合格点场预报（`fuxi_ens`）+ 格点实况（ERA5）+ 气候态（ACC/活跃度必需） | `scores.csv`、`diagnostics/spectrum_{var}.csv`、`diagnostics/spectrum_by_init.csv` | `rmse`、`crps`、`spread`、`rmse`（集合平均场）、`spread_error_ratio`、`acc`、`activity_ratio`、`activity_forecast`、`activity_observation`、`activity_bias`、`spectrum_power_ratio`、`spherical_band_power_*`<br>谱曲线同 `weather_field_scores` | 重·参考·成员·整场 · processes |
| **台风路径与强度检验（确定性）**<br>`weather_typhoon_det` | 确定性格点预报（`fuxi_phys`，msl 必给；u10m/v10m 给不了就没有强度项）+ BABJ 路径报文（`babj`，diamond7） | `scores.csv`、`typhoon/tc<编号>_<起报>.csv`（逐时效 15 列）、`_meta.json`、`typhoon/typhoon.csv`（全场合拼） | `track_err_km`、`at_km`、`ct_km`、`wind_err_ms`、`pmin_err_hpa`<br>逐 场次（台风 × 起报）：每项一个时效均值；逐时效曲线在路径表里，不进长表 | 轻 · threads（配置改 processes：瓶颈是读盘不是算） |
| **台风路径与强度检验（集合）**<br>`weather_typhoon_ens` | 集合格点预报（根目录按起报分日期目录，日期目录下放 `member_*/`；msl 必给）+ BABJ 路径报文 | 同上（路径表多 `n_members`/`n_valid_members`） | 确定性那 5 项 + `track_err_km_a`、`at_km_a`、`ct_km_a`<br>无后缀 = 方案 B（各成员位置先平均，再与实况求误差）；`_a` = 方案 A（各成员先各算误差，再对成员平均）。两者只在路径三项上有别；强度两项代数恒等，只出一行 | 轻·成员 · threads（配置改 processes） |
| **集合连续评分**<br>`fdp_ens_crps` | 集合格点场预报（`fengqing`）+ CRA40 再分析实况 | `scores.csv` | `crps`、`spread`、`rmse`（集合平均场）、`spread_error_ratio` | 重·成员 · processes |
| **要素场检验**<br>`fdp_field_scores` | 格点场预报（`fengqing`，z500 等）+ CRA40 实况 + 气候态（可选，ACC 必需） | `scores.csv` | `rmse`、`bias`、`acc` | 轻·参考 · threads |
| **中国区站点降水检验**<br>`fdp_precip_ts` | 格点降水预报 + Diamond 站点降水观测（中国区，cos 纬度加权） | `scores.csv`、`diagnostics/categorical_wide.csv` | `ts`、`pod`、`far`、`miss_rate`、`frequency_bias`<br>逐 阈值（0.1/13/25 mm）× 6h 时效（UTC 对齐，窗口不要求观测完整） | 轻 · threads |
| **降水空间检验**<br>`fdp_precip_fss` | 格点降水预报 + 格点降水实况（CRA / CMPAS） | `scores.csv` | `fss`（阈值 13 mm × 邻域窗口 1/3/5/15/31/63，每个组合一行）<br>明细 `window`/`n_points` | 重·整场 · processes |
| **活跃度比 / 功率谱**<br>`fdp_activity_spectrum` | 格点场预报（z500）+ 格点实况 + 气候态（活跃度比必需） | `scores.csv`、`diagnostics/spectrum_{var}.csv`、`diagnostics/spectrum_by_init.csv` | `activity_ratio`、`activity_forecast`、`activity_observation`、`activity_bias`、`spectrum_power_ratio`（二维谱，不减纬向均值）<br>逐波数谱曲线另出宽表 | 重·参考·整场 · processes |

内置配置里 `fdp_rmse_single_fengqing` 另外声明了 `json` writer，会多写一份 `scores.json`；
`weather_rmse_single_fuxi` 与 `weather_rmse_ens_fuxi` 声明的是 `spectrum`。
那都是配置的选择，不属于流程模板的产出。
注意配置里的 `writers` 是**替换**模板自带的那一份、不是追加，所以
`weather_rmse_single_fuxi` 要把模板的 `details` 一并写上，谱曲线才有落盘的地方。

### 评估指标说明

分类检验（降水阈值）：

| 指标 | 含义 | 口径 | 方向 |
|---|---|---|---|
| `ts` | TS（Threat Score，即 CSI）：命中占「命中+漏报+空报」的比例 | `hits / (hits + misses + false_alarms)` | 0~1，越大越好 |
| `pod` | 命中率：实况发生的事件里被预报到的比例 | `hits / (hits + misses)` | 0~1，越大越好；空报多也能拿高分，需与 `far` 同看 |
| `far` | 空报率：预报的事件里实况没发生的比例 | `false_alarms / (hits + false_alarms)` | 0~1，越小越好 |
| `miss_rate` | 漏报率 | `1 − POD` | 0~1，越小越好 |
| `frequency_bias` | 频率偏差：预报事件数 / 实况事件数 | `(hits + false_alarms) / (hits + misses)` | 1 为无偏，>1 空报偏多、<1 漏报偏多 |

概率评分（集合成员，超越式口径 `x ≥ 阈值`）：

| 指标 | 含义 | 口径 | 方向 |
|---|---|---|---|
| `aroc` | 概率排序能力（ROC 曲线下面积） | 以「成员中超过阈值的比例」为预报概率画 ROC，再积分；由概率直方图直接算，不落 ROC 点列 | 0.5 = 无技巧，1 = 完美排序 |
| `bs` | Brier 评分：概率预报的均方误差 | `mean((p − o)²)`，`p` = 成员超越频率，`o ∈ {0,1}` | 0 最好，单位同概率² |
| `bss` | Brier 技巧评分：相对气候基准的技巧 | `1 − BS / BS_ref` | >0 好于气候基准，1 = 完美，<0 不如气候 |
| `bs_ref`（明细） | 参考 BS | 有 `ref_probability` 参考时 `mean((p_clim − o)²)`；缺参考时降级为样本气候频率 `r(1−r)`——**降级后 BSS 与原版不可比** | 同上 |
| `base_rate`（明细） | 事件样本频率 | 该档样本里实况达到阈值的比例 | 用于判断样本是否失衡 |
| `n_points`（明细） | 参与该档评分的配对数 | — | — |

连续量 / 集合评分：

| 指标 | 含义 | 口径 | 方向 |
|---|---|---|---|
| `rmse` | 均方根误差 | `sqrt(Σw(f − o)² / Σw)`，`w` 为区域权重 | 越小越好，单位同变量 |
| `bias` | 平均误差（系统性偏差） | `Σw(f − o) / Σw` | 0 为无偏，>0 预报偏大 |
| `acc` | 距平相关系数：预报距平场与实况距平场的（加权）相关 | 距平 = 场 − 气候态；默认 `centered`（距今平再减域加权均值，经典皮尔逊），FDP/WeatherBench2 的 uncentered 口径用 `acc_uncentered` | −1~1，越大越好；**缺气候态会用零场兜底、结果无意义**（行状态标 `partial`） |
| `crps` | 连续排序概率评分：集合分布与实况的整体差异 | 闭式解，逐点只用有限成员，缺测成员不参与；不做非负截断 | 越小越好 |
| `spread` | 集合离散度 | `sqrt(Σᵢ(mᵢ − m̄)² / (M − ddof))`；`ddof` 可配（`metric_options["spread_error"]["ddof"]`）：1 = M−1 无偏（FDP 口径，默认），0 = 除以 N（vfc 口径，对拍 vfc 档案时用）——两口径差 √((M−1)/M)，M=51 时约 1% | 与 `rmse` 同量级才有意义 |
| `spread_error_ratio` | 离散度-误差比 | `SPREAD / RMSE(集合平均场)` | ≈1 标定良好，<1 过度自信，>1 欠自信 |
| `activity_ratio` | 活跃度比：预报的距平变化幅度相对实况 | `std(预报距平) / std(实况距平)`，面积加权 | <1 偏平滑（系统性偏弱），>1 偏噪；**缺气候态时无意义** |
| `activity_forecast` | 预报距平标准差 | 同 `activity_ratio` 的分子 | 诊断用，与实况侧同看 |
| `activity_observation` | 实况距平标准差 | 同 `activity_ratio` 的分母 | 诊断用 |
| `activity_bias` | 活跃度偏差：预报距平标准差 − 实况距平标准差 | `std(预报距平) − std(实况距平)` | 0 为无偏，<0 预报偏平滑；单位同 `activity_forecast` |
| `fss` | 邻域分数技巧评分：邻域平滑后再比「有/无」 | `1 − Σ(p_f − p_o)² / Σ(p_f² + p_o²)`，`p` 为邻域内超过阈值的格点占比 | 越大越好；窗口越大越接近随机基准，看技巧随尺度衰减 |
| `spectrum_power_ratio` | 总功率比：预报能量相对实况 | 预报功率谱总量 / 实况功率谱总量；逐波数曲线由 `spectrum` writer 另出 | 1 表示总能量不偏；单看总量会掩盖分布失真，需与谱曲线同看 |
| `spherical_band_power_forecast` | 球谐带功率（预报侧）：按**总波数**分带的球谐能量 | 老仓 `vfc/metrics/spectrum.py` 同口径（Legendre 递推 + Clenshaw-Curtis 积分，逐位移植）；带边界 `bands` 可配，默认 (1,4)/(5,20)/(21,40)/(41,64)/(65,128)，带名落 `group` 列 | 谱能量随尺度的分布，与实况侧同看 |
| `spherical_band_power_observation` | 球谐带功率（实况侧） | 同上 | 同上 |
| `spherical_band_power_ratio` | 球谐带功率比 | 预报带功率 / 实况带功率，逐带一行 | ≈1 标定良好；小波数带偏低 = 大尺度系统性衰减，大波数带偏高 = 噪声过剩 |

台风路径与强度（逐时效算完再对时效取均值；逐时效的完整曲线在 `typhoon/tc*.csv`，
不进长表）：

| 指标 | 含义 | 口径 | 方向 |
|---|---|---|---|
| `track_err_km` | 路径误差：预报中心与实况中心的距离 | 大圆距离；预报中心由链式搜索最低气压中心得到 | 越小越好（km） |
| `at_km` | 沿路径误差：误差在「路径前进方向」上的投影 | 以**前一个配对时效的实况位置**定向；>0 = 报快了 | 0 为无偏；**首时次为空**（没有前一个位置就无从定向） |
| `ct_km` | 横路径误差：误差在「路径侧向」上的投影 | 同上定向框架 | 0 为无偏 |
| `wind_err_ms` | 最大风速误差 | 诊断中心 ±`intensity_half_deg` 方框内最大 10m 风速（由 `u10m`/`v10m` 现合成）− 实况；给不出风速通道则整列留空 | 0 为无偏（m/s） |
| `pmin_err_hpa` | 中心气压误差 | 诊断中心最低气压 − 实况（hPa） | 0 为无偏 |
| `track_err_km_a` | 同 `track_err_km`，但走**方案 A** | 各成员先各算误差、再对成员平均（集合链专有） | 同 `track_err_km` |
| `at_km_a` / `ct_km_a` | 同 `at_km` / `ct_km`，方案 A | 同上 | 同上 |

长表里**无后缀的是方案 B**（各成员位置先平均成集合平均位置，再与实况求误差），
`_a` 才是方案 A。两者只在路径三项上有差别——三角不等式使然；强度两项
（`wind_err_ms`/`pmin_err_hpa`）两方案代数恒等，只出一行。
**两条链的首时次 `at`/`ct` 口径不同**：确定性链一贯留空，集合链用**起报时刻实况**
当定向基准（各对齐各自的历史归档，别按其中一条的直觉改另一条）。

几个读表要点：

- **长表只放标量。** 列联表计数（`hits`/`misses`/`false_alarms`/`correct_negatives`/`n_pairs`）、
  `BS_ref`/`base_rate`/`n_points` 都不占 `scores.csv` 的列，
  只进明细表 `diagnostics/scores_detail.csv`（声明 `details` writer 时写出），
  且会被 `categorical_wide` / `probability_wide` 各自透视成表头列。
  活跃度的四个量（`activity_ratio`/`activity_forecast`/`activity_observation`/`activity_bias`）
  都是标量，**都在长表里**。
- **球谐带功率直接进长表**（每样本每带 3 行，带名在 `group` 列，对标老仓
  `spherical_bands_<date>_<var>.csv` 的列契约），不需要 `spectrum` writer；
  分带边界在 `metric_options["spherical_bands"]["bands"]` 配（元组列表）。
  它只对全球含极网格有定义——区域/非全球网格在指标层报 `MetricError`，
  配了 `regions` 分带时它恒全球。
- **逐波数谱曲线长表和明细表都不进**，由 `spectrum` writer 出成两张宽 CSV
  （对标参考实现 `det_summary_spectrum_{var}.csv` / `spectrum_{init}_{var}.csv`）：
  `diagnostics/spectrum_{var}.csv` 是全体样本均值曲线，
  `diagnostics/spectrum_by_init.csv` 是逐起报曲线（在其 60 个时效上平均）。
  曲线有 720 个波数，塞进 `value` 会按「一个波数 × 三个字段」展开成明细行——
  逐日起报的全年段就是四千多万行，所以它走 `MetricResult.curve` 的原始数组，
  不经过长表。
- **同名不同口径靠 `aggregation` 区分。** 例如集合场检验里 `spread_error` 顺带输出的 `rmse`
  是**集合平均场的域加权**口径（`area_weighted`），`rmse` 指标是逐样本平均口径（`mean_over_samples`）。
- **空 `value` 不是 0。** NaN/Inf 一律写空字符串，该档有没有数看 `status` 列
  （`success` / `partial` / `no_valid_data` / `undefined`）。

一条流程模板只有**一个时间窗口**（`station_valid_time` 的 `window_hours` 同时决定观测累积长度和有效时效的筛选），所以集合降水检验按口径拆成了两条：24h 的 `weather_ts_ens` 出 TS 系列，6h 的 `weather_ts_ens_prob` 出概率评分。`weather_ts_ens_fuxi` 配置用 `pipeline=["weather_ts_ens", "weather_ts_ens_prob"]` 一条命令跑完两段，结果合并落同一个 `output_dir`（`scores.csv` 里靠 `window_h` 列区分，两个宽表各取自己那一段）。

台风链的两条硬约束（配置里已写死，改错会直接报错、不会出假数）：

- **`lead_chunk_days` 必须是 0。** 中心诊断是**链式**的——每一时效的搜索框中心是上一
  时效**诊断出的位置**，不是实况位置；时效被切开后后一块接不上，只能退回实况位置重起，
  结果会静默偏掉。所以一个场次（台风 × 起报）的整段时效必须落在同一个工作块里。
- **观测的 BABJ 只能 `slice` 装载。** 报文是北京时、一条报文跨台风的**整条生命史**，
  按起报时刻切会把实况切没；`resident`/`window` 会被计划层直接拦下。

集合链**逐成员串行读**（一次只驻留一个成员的场），峰值内存与确定性链同级，代价是每块
耗时 ×成员数。旧实现见 `ref/tiqnqi/xmetai_model_verification_xu/run_tc.py`。

输出文件口径：

| 文件 | 说明 |
|---|---|
| `scores.csv` | 统一评分长表（始终写出），每行 = 一个 变量×指标×阈值×时效×样本 的评分 |
| `coverage.csv` | 请求/有效样本覆盖率（显式声明 `coverage` writer 时写出） |
| `diagnostics/scores_detail.csv` | 诊断明细：列联表计数、`BS_ref`/`base_rate`/`n_points`（声明 `details` writer 时写出） |
| `diagnostics/spectrum_{var}.csv` | 逐波数功率谱的全体样本均值曲线（声明 `spectrum` writer 时写出） |
| `diagnostics/spectrum_by_init.csv` | 逐波数功率谱的逐起报曲线（同上） |
| `diagnostics/categorical_wide.csv` | 分类检验宽表（阈值 × 时效：TS/POD/FAR/漏报率/BIAS + `hits`/`misses`/`false_alarms`/`n_pairs` 计数） |
| `diagnostics/probability_wide.csv` | 概率评分宽表（阈值 × 时效：AROC/BS/BSS + `BS_ref`/`base_rate`/`n_points`） |
| `typhoon/tc<编号>_<起报>.csv` | 台风逐时效路径表（15 列：起止时刻、预报/实况经纬度与强度、五项误差；集合链另加 `n_members`/`n_valid_members` 与方案 A 三列）（声明 `typhoon_cases` writer 时写出） |
| `typhoon/tc<编号>_<起报>_meta.json` | 该场次的元信息：台风编号/中文名/起报时刻、起报点实况位置、种子偏移、搜索参数、成员名与成员数 |
| `typhoon/typhoon.csv` | 全部场次拼接总表（渲染路径图与人工核对用） |
| `scores.json` | 评分 JSON 快照 |

当前已接好的内置任务配置（`configs/`）：

| 配置 | 干什么 |
|---|---|
| `weather_ts_single_fgvp` | FGVP 确定性降水分类检验 |
| `weather_ts_ens_fuxi` | FuXi 集合降水：24h TS + 6h 概率两段一趟跑完 |
| `weather_rmse_single_fuxi` / `_fgvp` | 确定性连续量（RMSE/谱/ACC/活跃度） |
| `weather_rmse_ens_fuxi` / `_fgvp` | 集合场（RMSE/CRPS/Spread/ACC/活跃度/谱/球谐带） |
| `weather_typhoon_single_fuxi` | 台风路径与强度（确定性 × BABJ） |
| `weather_typhoon_ens_fuxi` | 台风路径与强度（集合，方案 A/B 一次算清） |
| `fdp_rmse_single_fengqing` / `fdp_rmse_ens_fengqing` | FDP 要素检验（确定性 / 集合） |
| `fdp_precip_single_fengqing` / `fdp_precip_ens_fengqing` | FDP 降水站点检验（确定性 TS/频率偏差 / 集合概率 BS/AROC/BSS） |
| `rainstorm_ts_single_fuxi` / `_fgvp_ctrl` | 暴雨过程分档检验：走 `weather_ts_det`，逐过程展开起报，过程编目（43 个过程，编号/起止/等级）的唯一真值在 `rainstorm_catalog.py` |

批量多模型见上「批量评测」——写一份 `cfgs` 列表即可，无内置示例。

## 评测数据与格式

每个评测吃三个角色的数据：**预报**（驱动集）、**观测**（配对集）、**参考**（辅助集，
只有距平类/概率类指标要）。这一节按角色说明各 reader 吃什么格式的数据、目录怎么摆、
变量与单位是什么。路径都相对 `root_dir`。

### 各能力的数据需求

| 评测能力 | 预报 | 观测 | 参考 |
|---|---|---|---|
| `weather_ts_det` | `fuxi`（`tp`） | `station` Diamond 站点降水 | — |
| `weather_ts_ens` | `fuxi_ens`（`tp`） | `station` | — |
| `weather_ts_ens_prob` | `fuxi_ens`（`tp`，逐成员） | `station` | `ref_probability`（BSS 必需，缺则降级） |
| `weather_field_scores` | 格点预报（`fuxi` 等） | 格点实况（`cra` / `era5_zarr`） | 气候态（ACC / 活跃度必需；纬向谱不需要） |
| `weather_ens_crps` | `fuxi_ens` | `cra` / `era5_zarr` | — |
| `weather_ens_field_scores` | 集合预报（`fuxi_ens` / `fengqing`） | `era5_zarr` | 日气候态（ACC / 活跃度必需） |
| `fdp_ens_crps` | `fengqing`（集合） | `cra` | — |
| `fdp_field_scores` | `fengqing` | `cra` | `climatology`（ACC 需要） |
| `weather_typhoon_det` | 格点预报（`msl` 必给；`u10m`/`v10m` 可选） | `babj` 报文 | — |
| `weather_typhoon_ens` | 集合格点预报（**日期目录下带 `member_*/`**） | `babj` 报文 | — |
| `fdp_precip_ts` | 格点降水预报 | `station`（中国区） | — |
| `fdp_precip_fss` | 格点降水预报 | **格点**降水实况（当前内置 `cra`） | — |
| `fdp_activity_spectrum` | `z500` 格点预报 | 格点实况 | `climatology`（活跃度比必需） |

表里点名的默认源来自流程模板的 `description`，不是硬绑定：换成别的具名布局
（如 `fuxi` ↔ `fengqing_phys`）照样跑，只要变量名对得上。`fdp_precip_fss` 不吃站点观测。

### 预报产品（格点预报布局）

预报源只需要三件事：目录模板、文件名正则、时效从哪来。三件事都写在
`io/layouts.py` 的 `GriddedLayout` 里。

| reader | 目录 / 文件名 | 时效来源 |
|---|---|---|
| `fuxi` / `fuxi_phys` | `{root}/20250101/001.nc` | **文件序号**（`lead_from="index"`） |
| `fuxi_ens` / `fuxi_ens_phys` | `{root}/20250101/member_000/001.nc` … `member_050/` | 同上；成员号取子目录名 |
| `fengqing` / `fengqing_phys` | `{root}/Fengqing_1.0_GLB_PLEVELS_OP25_6HOR_ENS_FCST_2025010100_024.nc` | **文件名里的 3 位时效（小时）** |

- 起报时刻默认取所在日期目录（`root_template` 为 `{root}/{init:%Y%m%d}`）；fengqing
  的起报时刻在文件名里，10 位 `YYYYMMDDHH`。
- **`lead_from="index"` 的时效 = 该起报目录内文件序号（1-based）× `step_hours`**
  （默认 6h）：`001.nc` → +6h、`002.nc` → +12h。所以文件必须按时效升序命名且不缺号，
  否则后面的时效会整体前移。
- fengqing 每个时效是两个文件：`PLEVELS`（`z500` 等）与 `SURFACE`（`tp`/`t2m`/`msl`/
  地面风），`groups` 声明已写好，配置不用管；成员维在文件里，读出后统一转成
  `(member, lat, lon)`。
- 引擎 netcdf，打开失败回退 h5netcdf。

### 观测

**Diamond 站点降水**（`station`，别名 `diamond_station`）

```
{root}/20250101/00.000        # 分层：日期目录/HH.000
{root}/2025010100.000         # 扁平：YYYYMMDDHH.000（两种都认）
```

文件是 GBK 文本，**前 2 行是文件头**，之后每行一个站：

```
diamond  3 2025年1月1日0时1小时降水(逐时)
25 1 1 0 1000 0 0 0 0 1 77017
站号 经度 纬度 高度(m) 降水(mm)
45004 114.1728 22.3119 66.4 0.0
```

- 一个文件 = 一个整点、全部站点；第 5 列是**逐小时**降水量（mm），不算成 6h/24h——
  累积由流程模板的 `time_window_accumulator` 做，文件本身不用预先累积。
- 读出为 `(time, station)` 时间序列，变量名统一成 `precipitation`（标准名 `tp`）。
  这个源只出降水，配置里观测侧写 `"variable": "precipitation"`。
- 站点经纬度/高度取该站**首次出现**的时次（不是只看第一个时次，否则首时次缺报的站
  经纬度会是 NaN 被区域筛选误剔除）；某站某时次缺行则该格 NaN。
- 时区：Diamond 是**北京时**，协议默认 `local_utc_offset_hours=8` 对齐；
  `fdp_precip_ts` 用 0（按 UTC 对齐）是模板里的显式选择。
- 可选 `station_whitelist`（站号列表）或 `station_list`（清单文件路径，取每行第一个
  字段当站号）；不写 = 全部站点。区域筛选在协议层另配。

**CRA40 再分析**（`cra`）——GRIB，两个分组各一个文件：

```
{root}/ART_ATM_GLB_0P25_6HOR_ANAL_{YYYYMMDDHH}.grib2              # ATM
{root}/CRA40LAND_SURFACE_{YYYYMMDDHH}_GLB_0P25_HOUR_V1_0_0.grib   # SURFACE
```

变量靠 cfgrib 过滤器挑（`z500` → `shortName=gh, level=500`；`t2m` → `2t`；`msl` →
`meanSea`；`u10`/`v10` → `10u`/`10v`；`tp` → `stepType=accum`），所以一个文件里有多少层
都不影响。配置侧写标准名，输出 `(valid_time, lat, lon)`。

**ERA5 zarr**（`era5_zarr`）

```python
observation_reader={
    "type": "era5_zarr",
    "stores": {"pl":  ".../era5_pl_...c84.p25.h6.zarr",
               "sfc": ".../era5_sfc_...c15.p25.h6.zarr"},
    "variables": [...],
}
```

- 与上面两类不同：**一个 store 装下全部时次**，不靠文件名发现——Catalog 只校验路径
  存在。所以这个 reader 要的是 `stores` 而不是 `root_dir`。
- 通道名旁挂在 store 同级目录的 `channel_names.json` 里；**没有层次轴**，层次编码在
  通道名里（`z_500`、`u_200`），配置侧仍写标准名。
- 气压层走 `pl`、地面走 `sfc`（`groups` 已声明）；GRIB 缺测哨兵 `1e30` 当缺测。

**BABJ 台风路径报文**（`babj`）——记录台风**最佳路径**的业务分析报文：

```
{root}/babj2501.dat        # 编号 2501 台风，一个文件 = 一整条路径
```

- diamond7 格式、**GBK** 编码、空白分隔，时刻是**北京时**。一个文件横跨多天
  （台风的整条生命史），不是「一个文件一个时刻」——所以本 reader **不做时间筛选**，
  `discover` 一把全给、`read` 一次全读（报文以十计、体量 KB 级），由协议按编号取用。
  `root_dir` 也可以直接指到单个 `.dat`。
- 只取报文里**时效 000** 的行当分析实况，输出 `(time, storm)` 的
  `lat`/`lon`/`pmin`/`vmax`；坐标 `storm` 是编号字符串、`tcname` 是中文名。
- **各台风的实况时刻不对齐**：`time` 取并集，缺的位置是 NaN（取用时按
  `.sel(storm=…, time=…)`，NaN 即「该时刻无实况」）。预报侧对不上实况的时效
  **不外推不插值，留空**。

### 参考

**整场气候态**（`climatology`）——目录里一堆文件，按「月日 + 时次」索引，**与年份无关**：

```
{root}/ART_ATM_GLB_0P25_CLI_ANAL_{MMDD}{HH}.grib2
# 2026-08-20T06 对应 ART_ATM_GLB_0P25_CLI_ANAL_082006.grib2
```

读取过滤器与 CRA 实况一致（同一套 cfgrib filters），输出 `(valid_time, lat, lon)`，
`valid_time` 的年月日是查表时那个时刻。可选 `engine`（默认 `cfgrib`）。

**日序气候态**（`daily_climatology`）——**单个文件**，时间轴是年内日序：

```
{root}/era5_clim_phys_14.nc
```

- 时间轴 0…364（365 天文件）或 0…365（366 天文件）：365 天的把闰年 2/29 并进 2/28；
  366 天的保留真日序、平年跳过 2/29。
- `time:units` **只用单位**（days / hours / …），起算时刻被忽略——索引的是"年内位置"，
  不是绝对时刻。
- `smooth_days`（默认 15 天）是**环形平滑天数**：取场前对年内日序做中心对齐的滑动
  平均（±7.5 天，跨年首尾相接），抹平单日气候态的天气尺度噪声，与参考实现的
  `--climo-window` 同口径。取 1 = 不平滑。这是气象参数，跟 `execution.loads` 的
  `window:W`（IO 窗块滚动缓存）**不是一回事**。可选 `scales` / `units` 覆盖换算。
- 整年 × 全球很大，按纬度分块惰性读，峰值与分块有关、不整块载入。

⚠️ 两种气候态**不能混用**：把日序单文件接到按 `MMDDHH` 找文件的 Catalog 上，会把整份
文件当成每一个有效时刻，**静默算错且不报错**。

**逐站气候概率**（`ref_probability`）——BSS 的外部基准：

```
{root}/082006.000
```

- 文件名 `MMDDHH.000`，是**北京时**，`HH` 是 6h 窗口的**结束时刻**。
- 每行一个站：`站号 p≥0.1 p≥4 p≥13 p≥25`，4 列概率都在 [0,1]，顺序固定（对应
  `PROB_THRESHOLDS`；改阈值要同步改参考文件的列）。
- 缺某站则该站 4 列 NaN；**缺整个时次直接报错**，不静默跳过。
- 不走 Catalog / DataBundle，也不进加载策略层——协议按验证时刻逐样本查表，所以配置里
  **不能**给它配 `loads`。

### 变量与单位

标准名是配置与指标层唯一认的名字；源变量名与单位换算烘焙在布局声明里
（`io/layouts.py`），下游全程是报告单位。

| 标准名 | `fuxi` / `fuxi_ens` | `fuxi_phys` / `fuxi_ens_phys` | `fengqing` | `fengqing_phys` |
|---|---|---|---|---|
| `z500` | `z500`，m²/s² **÷g** → m | `z500`，**m²/s²**（不除 g） | `Z500`，÷g → m | `Z500`，m²/s² |
| `q700` / `q2m` | `q700` / `q2m`，g/kg | 同 | — | `Q700` / `Q2M`，g/kg |
| `t2m` | `t2m`，K | 同 | `T2M`，K | 同 |
| `msl` | — | `msl`，Pa | `MSL`，Pa | 同 |
| `t700` / `t850` | — | `t700` / `t850`，K | — | `T700` / `T850`，K |
| `u200` / `v200` / `u850` / `v850` | — | `u200` 等，m/s | — | `U200` 等，m/s |
| `u10m` / `v10m` | — | `u10m` / `v10m`，m/s | `U10` / `V10`，标准名是 `u10` / `v10` | `U10` / `V10`，标准名 `u10m` / `v10m` |
| `tp` | `tp`，mm（逐 6h 间隔累积） | 同 | `TP`，mm | 同 |

观测侧：

| 标准名 | `cra` | `era5_zarr` |
|---|---|---|
| `z500` | m（cfgrib `gh`） | **m²/s²** |
| `t2m` | K | K |
| `t700` / `t850` | — | K |
| `msl` | Pa | Pa |
| `u200` / `v200` / `u850` / `v850` | — | m/s |
| `u10` / `v10`（或 `u10m` / `v10m`） | m/s | m/s |
| `q700` / `q2m` | — | **kg/kg ×1000 → g/kg** |
| `tp` | mm | **m ×1000 → mm** |
| `ws10m` | — | store 自带通道，m/s |

`cra` 与按 `MMDDHH` 索引的 `climatology` 都只声明了 6 个变量（`z500`/`gh`/`t2m`/`msl`/
`u10`/`v10`，单位口径一致）——要评各层风温湿得换 `era5_zarr` 或日序气候态。日序气候态的
要素表**随文件走**：请求了文件里没有的变量会被跳过并在日志里打 WARNING，这不是错误，
但对应的距平类指标就没有参考场了。

**派生变量**：`ws10m` / `ws850` / `ws200` 多数文件里都没有，由 `sqrt(u²+v²)` 在**预报、
实况、气候态三侧各自现合成**（名字被请求、且两个分量齐备时才合成）。唯一例外是
`era5_zarr` 自带 `ws10m` 通道，实况侧直接取用。所以日志里"气候态没有 ws850/ws200"
不是错误——它们的分量在就够了，合成的结果照用。

> `z500` 的两种口径差一个 g，跨源对拍前先确认两边用的是哪个布局：`*_phys` 系列是 xu
> 报告口径（m²/s²），`fuxi` / `fengqing` 是 ÷g 换算成 m。

### reader 参数速查

| reader | 必需 | 可选 |
|---|---|---|
| `fuxi` / `fuxi_ens` / `fuxi_phys` / `fuxi_ens_phys` | `root_dir` | `step_hours`（默认 6）、`variable` 或 `variables`、`source_id` |
| `fengqing` / `fengqing_phys` | `root_dir` | `variable` 或 `variables`、`source_id` |
| `cra` | `root_dir` | `variable` 或 `variables`、`source_id` |
| `era5_zarr` | `stores`（`{"pl": …, "sfc": …}`） | `variable` 或 `variables`、`source_id` |
| `station` / `diamond_station` | `root_dir` | `station_whitelist` 或 `station_list`、`source_id` |
| `climatology` | `root_dir` | `engine`（默认 `cfgrib`）、`source_id` |
| `daily_climatology` | `root_dir` | `smooth_days`（默认 15）、`scales`、`units`、`source_id` |
| `ref_probability` | `root_dir` | `source_id`（不可配 `loads`） |

`variable` 给单个变量名、`variables` 给列表，两者都认；不写则用流程模板声明的。

## 如何扩展

- **新增模型 / 数据集**：已支持的文件布局 → 写一份数据源配置即可；全新格式 → 在 `io/` 实现 Reader，再到 `components.py` 注册。
- **新增指标**：在 `metrics/` 实现 `Metric`（`accumulate` / `merge` / `finalize`），到 `components.py` 注册。
- **新增流程**：在 `pipeline/pipelines.py` 加一个 `PipelineTemplate`。
- **新增数据源/算法** = 加注册项；**新增评测** = 加配置，代码不动。

