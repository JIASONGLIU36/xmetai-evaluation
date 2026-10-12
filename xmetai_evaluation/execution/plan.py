# -*- coding: utf-8 -*-
"""执行计划：一段评测怎么切成工作块、每块读什么。

计划层在父进程里跑一次，只做便宜的事：catalog 探测（列文件，不读数据）、
构建组件拿画像、按起报日切块、给观测/参考角色准备加载声明。真正贵的
物化推迟到块执行时（resident 角色会在进程池 fork 前由执行器预热）。

切块的口径：**预报是驱动集**——工作块永远从预报侧按起报时刻切；观测和
参考需要什么，由这一块的起报推出的有效时刻跨度决定。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from xmetai_evaluation.components import register_builtin_components
from xmetai_evaluation.core.contracts import DataIndex, DataRequest
from xmetai_evaluation.core.errors import ConfigError, EvaluationError
from xmetai_evaluation.core.registry import ComponentType, get_registry
from xmetai_evaluation.execution.executor import needs_reference
from xmetai_evaluation.execution.loader import RoleSpec, RunLoader
from xmetai_evaluation.execution.profiles import ResourceProfile, union_profile
from xmetai_evaluation.execution.strategy import ExecutionStrategy, derive_strategy
from xmetai_evaluation.pipeline.protocols import (
    TYPHOON_PROTOCOLS,
    babj_init_times,
    sample_leads,
    station_observation_span,
    select_observation_files,
    typhoon_storm_filter,
)
from xmetai_evaluation.pipeline.spec import PipelineSpec

log = logging.getLogger(__name__)


@dataclass
class WorkChunk:
    """一个工作块：一段起报时刻 × 一段时效 + 它自己的子声明。"""

    chunk_id: str  # 形如 "20250101_L24-48" 或 "20250101-20250103_L0-120"
    init_times: List[datetime]
    #: 本块要读的时效集（采样窗 + 累积预热）；给 reader 当文件过滤条件
    read_leads: List[float] = field(default_factory=list)
    #: 本块要出分的采样时效集（只有时效窗本身，不含预热）
    sample_leads: List[float] = field(default_factory=list)


@dataclass
class WorkPlan:
    """一段评测的完整执行计划。"""

    spec: PipelineSpec  # 段的原始声明（父进程口径，收尾/落盘用）
    chunks: List[WorkChunk]
    sub_specs: List[PipelineSpec]  # 每块一份：init_times 注入、limit 清空
    strategy: ExecutionStrategy
    loader: RunLoader  # serial/threads 直接共享；processes+fork 预热后经全局共享
    handles: Dict[str, Any]  # forecast / observation / reference 的 SourceHandle
    metrics: List[Any]  # 段级指标实例（画像与收尾展开用；块内各自重建）
    profile: ResourceProfile


def build_plan(
    spec: PipelineSpec,
    execution: Optional[Dict[str, Any]] = None,
    num_workers: int = 1,
    order: int = 1,
    total: int = 1,
) -> WorkPlan:
    """把一段声明变成可执行的工作块计划。"""
    register_builtin_components()
    registry = get_registry()

    # 1) 构建组件拿联合画像（父进程一次；块内 worker 会按同一声明重建）
    forecast = registry.build(
        ComponentType.READER, spec.forecast.reader, **spec.forecast.params
    )
    observation = registry.build(
        ComponentType.READER, spec.observation.reader, **spec.observation.params
    )
    metrics = [
        registry.build(ComponentType.METRIC, item.name, **item.params)
        for item in spec.metrics
    ]
    reference = None
    if spec.reference is not None and needs_reference(metrics):
        reference = registry.build(
            ComponentType.READER, spec.reference.reader, **spec.reference.params
        )
    elif spec.reference is not None:
        log.info("本段指标都不需要参考源，跳过构建 %s", spec.reference.reader)
    profile = union_profile(metrics)

    # 2) 起报发现：显式声明优先，否则按评测时段探测 catalog（只列文件）
    init_times = _discover_init_times(spec, forecast)
    if not init_times:
        raise EvaluationError(
            f"流程 {spec.pipeline or spec.name} 在评测时段内没有可用预报起报"
        )
    if spec.protocol in TYPHOON_PROTOCOLS:
        # 起报时刻由报文**反推**（旧链路同口径）：预报目录是逐日的，台风却不是
        # 天天有——不筛的话空档期的每一天都会切出一个 0 样本的块，在 manifest
        # 里堆成一片"工作块失败"。放在 limit 之前筛，limit=1 才真的是"第一个
        # 有台风的起报日"。
        #
        # seed_min_offset_hours 与协议读的是同一个配置项：计划层与协议必须按同一
        # 口径判断"这个起报能不能起链"，否则能跑的场次在切块前就被剔掉了。
        # 时效步长（种子须落在时效网格上）同样从预报布局取，两边同源。
        init_times = babj_init_times(
            observation,
            init_times,
            spec.local_utc_offset_hours,
            typhoon_storm_filter(spec.options),
            float(spec.options.get("seed_min_offset_hours", 6.0)),
            _layout_step_hours(forecast),
        )
        if not init_times:
            # 走 period() 而不是直接读 start_date/end_date：多时段配置下那两个是
            # None，直接插进消息里会打出「None–None」。包络区间在这里正好合适
            # （"整个评测时段里没有一条分析场"）。
            period_start, period_end = spec.period()
            raise EvaluationError(
                f"评测时段 {period_start:%Y%m%d}–{period_end:%Y%m%d} 内没有任何起报时刻"
                f"对得上 BABJ 报文的分析场（报文里没有在评台风的分析记录？）"
            )
    if spec.limit:
        init_times = init_times[: spec.limit]

    # 3) 时效：显式声明优先，否则从索引拿最大时效按步长展开（不读数据）
    leads = _discover_leads(spec, forecast, init_times)

    # 4) 推导执行策略（驻留策略按数据源形态定，配置可覆盖）
    strategy = derive_strategy(
        profile,
        observation_reader=spec.observation.reader,
        reference_reader=spec.reference.reader if spec.reference else None,
        execution=execution,
        num_workers=num_workers,
    )

    # 5) 切块：(chunk_days 个起报日) × (lead_chunk_days 天时效)，每块一份子声明
    if 0 < strategy.lead_chunk_days * 24 < spec.window_hours:
        log.warning(
            "lead_chunk_days=%d 的时效窗（%dh）比累积窗口 %dh 还窄："
            "每个块都要往前读预热、窗内样本稀疏，建议不小于 %d 天",
            strategy.lead_chunk_days,
            strategy.lead_chunk_days * 24,
            spec.window_hours,
            math.ceil(spec.window_hours / 24),
        )
    warmup = _warmup_hours(spec, _layout_step_hours(forecast))
    chunks = _chunk_work(
        spec,
        init_times,
        strategy.chunk_days,
        leads,
        strategy.lead_chunk_days,
        warmup,
    )
    sub_specs = [_sub_spec(spec, chunk) for chunk in chunks]

    # 6) 观测/参考角色的加载声明（预报恒为 slice，不进加载层）
    roles = _build_roles(
        spec, strategy, forecast, observation, reference, init_times, leads,
        n_chunks=len(chunks), profile=profile, warmup_hours=warmup,
    )
    # 预取只在"一个 loader 一个使用者"的形态下开：threads 是多个块并发读同一个
    # loader，各自排一个预取只会互相顶掉、白读一遍（threads 本来就已经有 N 路
    # 并发读，也轮不到预取来补空档）。
    mode = strategy.resolve_mode(len(chunks), profile)
    loader = RunLoader(roles, prefetch=mode != "threads")

    log.info(
        "第 %d/%d 段 %s：%d 个起报切成 %d 个工作块（chunk_days=%d，"
        "lead_chunk_days=%d，单块读时效 %d 个）",
        order,
        total,
        spec.pipeline or spec.name,
        len(init_times),
        len(chunks),
        strategy.chunk_days,
        strategy.lead_chunk_days,
        len(chunks[0].read_leads) if chunks else 0,
    )
    log.info("画像 %s，驻留策略 %s", profile.as_dict(), strategy.loads or "（无缓存角色）")
    if any(spec_.policy == "window" for spec_ in roles.values()):
        log.info(
            "窗块预取：%s（后台线程读下一个窗块；待用槽只放一块）",
            "关（threads 形态，并发读会互相顶掉）" if mode == "threads" else "开",
        )
    return WorkPlan(
        spec=spec,
        chunks=chunks,
        sub_specs=sub_specs,
        strategy=strategy,
        loader=loader,
        handles={
            "forecast": forecast,
            "observation": observation,
            "reference": reference,
        },
        metrics=metrics,
        profile=profile,
    )


def _forecast_variables(spec: PipelineSpec) -> List[str]:
    """预报侧变量（两个协议的取法并起来：variables / variable / variables 映射）。"""
    names = spec.forecast.params.get("variables")
    if names:
        return [str(name) for name in names]
    if spec.forecast.params.get("variable"):
        return [str(spec.forecast.params["variable"])]
    if spec.variables.get("forecast"):
        return [str(spec.variables["forecast"])]
    raise ConfigError(
        "预报数据源必须声明 variable 或 variables（计划层要先探测可用起报）"
    )


def _observation_variables(spec: PipelineSpec) -> List[str]:
    """观测侧变量（grid 协议口径：观测请求用的就是这组变量）。"""
    names = spec.observation.params.get("variables")
    if names:
        return [str(name) for name in names]
    if spec.observation.params.get("variable"):
        return [str(spec.observation.params["variable"])]
    if spec.variables.get("observation"):
        return [str(spec.variables["observation"])]
    raise ConfigError("观测数据源必须声明 variable 或 variables")


def _discover_init_times(spec: PipelineSpec, forecast: Any) -> List[datetime]:
    explicit = [
        datetime.fromisoformat(str(value))
        for value in spec.forecast.params.get("init_times", [])
    ]
    if explicit:
        return sorted(explicit)
    discovered = forecast.catalog.discover(
        DataRequest(
            source_id=forecast.source_id,
            variables=_forecast_variables(spec),
            init_times=spec.init_candidate_times(),
        )
    )
    return list(forecast.reader.available_init_times(discovered))


def _discover_leads(
    spec: PipelineSpec, forecast: Any, init_times: List[datetime]
) -> List[float]:
    """这段预报的时效列表（小时）。

    显式声明的 lead_times 原样用；否则从 catalog 索引拿最大时效，按布局
    步长展开成 0..max。展开结果只会是真实时效的超集——块执行时协议按
    实际读到的时效发请求，多出来的时刻只是留在驻留缓存里没人用。
    """
    # ``or []``：键写了但值是 None 时也要当"没声明"。只写 ``.get(..., [])``
    # 的话，键在、值是 None 拿到的是 None 而不是那个缺省，下一句遍历就 TypeError。
    declared = [float(value) for value in spec.forecast.params.get("lead_times") or []]
    if declared:
        return declared
    request = DataRequest(
        source_id=forecast.source_id,
        variables=_forecast_variables(spec),
        init_times=init_times,
    )
    index = forecast.catalog.discover(request)
    lead_max = float(forecast.reader.max_lead_hours(index))
    step = _layout_step_hours(forecast)
    if lead_max <= 0:
        return [0.0]
    # 从 0 铺起：有的布局带 0 时效文件（valid = 起报时刻本身），超集漏了
    # 它的话驻留缓存会静默缺那个有效时刻。多出来的时刻只是没人用，无害。
    return [step * i for i in range(0, int(lead_max / step) + 1)]


def _layout_step_hours(forecast: Any) -> float:
    """预报源的时效步长（小时）。切块的预热按它算：累积窗要的是连续步。"""
    layout = getattr(forecast.reader, "layout", None)
    return float(getattr(layout, "step_hours", None) or 6.0)


def _warmup_hours(spec: PipelineSpec, step_hours: float) -> float:
    """累积变换需要的预热时效（小时）。

    ``TimeWindowAccumulator`` 是滑动窗，窗口不完整的开头时刻会被直接跳过
    （``transforms/temporal.py``），所以一个时效窗要出第一个样本，读取集必须
    往前多带 ``(窗口步数 - 1)`` 个时效。没有累积变换的流程（预报文件本身就是
    窗口累积量，如 6h 降水）不需要预热。

    **至少带 1 步**，即使 ``窗口步数 == 1``（窗口长度恰好等于时效步长，如 6h
    窗配 6h 预报）。那种流程数学上不需要累积，但累加器要先拿 ``np.diff`` 反推
    步长才能算出窗口要几步——读取集只有一个时效时它推不出来，直接抛
    "Need at least 2 time steps"，于是**每段时效的最后一个窗永远出不了样本**
    （实测 6h 段顶边块全灭）。多带一步给的是步长这个信息，不是精度：那个多出
    来的窗不在 ``sample_leads`` 里，协议会跳过，不会多出样本。
    """
    if spec.transform("time_window_accumulator") is None:
        return 0.0
    steps = int(round(spec.window_hours / float(step_hours)))
    return max(1, steps - 1) * float(step_hours)


def _sample_values(spec: PipelineSpec, leads: List[float]) -> List[float]:
    """本流程可评的采样时效集。

    站点协议是滑动累积窗——只有完整窗口的末端才配得上一个累积实况，取
    ``sample_leads``；格点协议逐有效时刻配对，读到的每个时效都是样本。
    分派与协议的 ``duplicate_key_policy`` 一样按协议身份走，两边同一口径。
    """
    if spec.protocol == "station_valid_time":
        return sample_leads(leads, spec.window_hours)
    return sorted({float(lead) for lead in leads})


def _sample_window_groups(
    sample_values: List[float], lead_chunk_days: int
) -> List[List[float]]:
    """把采样时效切成若干时效窗；``lead_chunk_days=0`` 表示不切（整段一组）。

    分组锚在 lead=0 上、按采样点归属：采样时效是「完整窗口的末端」，固定累积窗
    的流程（如 24h 降水）采样点本来就稀疏，空组天然不出现——不会切出
    「一个采样点都没有」的块（那种块 processed==0，会被判失败块）。
    """
    if not sample_values:
        return []
    if lead_chunk_days <= 0:
        return [sorted(float(lead) for lead in sample_values)]
    span = float(lead_chunk_days) * 24.0
    groups: Dict[int, List[float]] = {}
    for lead in sample_values:
        groups.setdefault(int(float(lead) // span), []).append(float(lead))
    return [groups[index] for index in sorted(groups)]


def _chunk_work(
    spec: PipelineSpec,
    init_times: List[datetime],
    chunk_days: int,
    leads: List[float],
    lead_chunk_days: int,
    warmup_hours: float,
) -> List[WorkChunk]:
    """工作块 = (起报日组) × (采样时效窗)，两维的叉积。

    时效维是内存的主要旋钮：不切时效时一个块要把整个时效跨度 × 成员数全部
    物化（15 天 × 51 成员约 25GB）。切窗落在**采样**时效上而不是原始时效跨度
    （理由见 ``_sample_window_groups``）。

    读取集 = 窗内采样时效 ∪ 往前 ``warmup_hours`` 的预热时效。预热只从 ``leads``
    里取——时效是显式稀疏声明（如 ``[24, 48, 72]``）时中间步不在声明里，累积窗
    本来就拼不出来，不是切块引入的。读取集可以跨窗重叠，但采样集是
    (起报, 时效) 的纯划分——指标只对采样到的样本累加，重复读不会重复计数。

    块序按起报组升序、组内时效窗升序（只为可读性与 resume 稳定；归并的正确性
    不依赖块序，见 ``executor.merge_outcomes`` 的 replace 分支）。
    """
    window_hours = spec.window_hours
    sample_values = _sample_values(spec, leads)
    if not sample_values:
        span = max(leads) if leads else 0.0
        raise EvaluationError(
            f"评测时段内最大时效 {span:g}h 小于累积窗口 {window_hours}h，"
            f"没有任何完整窗口可评（放长时效，或调小 window_hours）"
        )
    windows = _sample_window_groups(sample_values, lead_chunk_days)
    ordered_leads = sorted({float(lead) for lead in leads})

    by_date: Dict[Any, List[datetime]] = {}
    for value in sorted(init_times):
        by_date.setdefault(value.date(), []).append(value)
    dates = sorted(by_date)

    chunks: List[WorkChunk] = []
    for start in range(0, len(dates), chunk_days):
        group = dates[start : start + chunk_days]
        times = [t for date in group for t in by_date[date]]
        init_id = group[0].strftime("%Y%m%d")
        if len(group) > 1:
            init_id += "-" + group[-1].strftime("%Y%m%d")
        for window in windows:
            low, high = min(window), max(window)
            lead_id = f"L{low:g}" if low == high else f"L{low:g}-{high:g}"
            chunks.append(
                WorkChunk(
                    chunk_id=f"{init_id}_{lead_id}",
                    init_times=times,
                    read_leads=[
                        lead
                        for lead in ordered_leads
                        if low - warmup_hours <= lead <= high
                    ],
                    sample_leads=list(window),
                )
            )
    return chunks


def _sub_spec(spec: PipelineSpec, chunk: WorkChunk) -> PipelineSpec:
    """块的子声明：起报与时效换成这一块的，limit 清空（计划层已截断过）。

    ``lead_times`` 是**覆盖**不是合并：``_discover_leads`` 把同名键当用户声明，
    留下一份全时效声明会让块把整个时效跨度读回来，时效切分就白做了。
    """
    params = dict(spec.forecast.params)
    params["init_times"] = [value.isoformat() for value in chunk.init_times]
    params["lead_times"] = list(chunk.read_leads)
    params["sample_leads"] = list(chunk.sample_leads)
    return replace(spec, forecast=replace(spec.forecast, params=params), limit=None)


def _grid_valid_times(init_times: List[datetime], leads: List[float]) -> List[datetime]:
    return sorted(
        {
            init + timedelta(hours=float(lead))
            for init in init_times
            for lead in leads
        }
    )


def _lead_window_days(leads: List[float]) -> int:
    """整段时效跨度（天）——自动定窗的基数。

    这里是**整个时效跨度**（``max(leads)``），不是单块的时效窗宽
    （``lead_chunk_days``）。块序是「起报日外层、时效窗内层」：一个起报日组
    要顺次扫完所有时效窗，用到的观测日跨度就是整个时效跨度；组与组之间还要
    回跳（扫完 L 天、回跳 L-2 天）。窗宽只有 ≥ 这个跨度，回跳时上一轮读过的
    块才还留在缓存里（配合 loader 的 ±1 块保留），一个观测日整 run 只读一次。
    ``lead_chunk_days=0``（不切时效）时两种口径本就相同。
    """
    return max(1, math.ceil(max(leads) / 24.0)) if leads else 1


def _auto_window_days(
    strategy: ExecutionStrategy,
    n_chunks: int,
    profile: ResourceProfile,
    leads: List[float],
    warmup_hours: float = 0.0,
) -> int:
    """自动窗宽（同时也是配置窗宽的**上限**）：观测日跨度 + 并发跨度。

    需要的最小窗宽 = 一个起报日组扫完所有时效窗所碰到的观测日跨度
    = 整段时效跨度 + 预热 + 块内起报跨度（``chunk_days`` 天）。

    threads 下并发块时间相邻（队列按序领块），再加 (并发数-1)*chunk_days；
    processes 下每进程段内串行、一次只跑一块，这项为 0。目的是让并发中的
    所有块的观测跨度都落进**相邻两个**窗块（loader 保留 ±1 块）。

    再宽没有收益：读量只取决于"装不装得下一个组的观测日跨度"，装得下之后
    加宽只多占内存——而且进程模式下窗块是每个子进程各一份，加宽的钱按
    worker 数翻。
    """
    mode = strategy.resolve_mode(n_chunks, profile)
    workers = strategy.resolve_workers(mode)
    spread = (workers - 1) * strategy.chunk_days if mode == "threads" else 0
    return (
        _lead_window_days(leads)
        + math.ceil(warmup_hours / 24.0)
        + max(1, strategy.chunk_days)
        + spread
    )


def _resolve_window(
    role: str,
    strategy: ExecutionStrategy,
    n_chunks: int,
    profile: ResourceProfile,
    leads: List[float],
    warmup_hours: float = 0.0,
) -> Tuple[str, int]:
    """拿角色的 (驻留策略, 窗宽)；window 的最终窗宽在这里落定并回写声明。

    最终窗宽 = ``min(算出来的最小窗宽, 配置里的值)``：

        ``window``      不写数字 —— 直接用算出来的最小值；
        ``window:W``    配置值当**上限**用 —— 比最小值小就照用小值（读会变多、
                        缓存变小，是拿内存换 IO 的主动选择）；比最小值大就收到
                        最小值（再宽不减少重读，只多占内存，而进程模式下窗块是
                        每个子进程各一份，加宽的钱按 worker 数翻）。

    回写 ``strategy.loads`` 是为了 manifest 的 execution 段显示**实际用的**值
    （如 ``"window:16"``），跑完可核对当时算的是多少——配置写了 32 而实际用了
    16 时，manifest 里看到的是 16，收到的动作在日志里另有一行。
    """
    policy, window = strategy.load_policy(role)
    if policy != "window":
        return policy, int(window or 0)
    minimum = _auto_window_days(strategy, n_chunks, profile, leads, warmup_hours)
    requested = None if window is None else int(window)
    final = minimum if requested is None else min(requested, minimum)
    if requested is None:
        log.info(
            "角色 %s 自动定窗：%d 天（整段时效跨度 %d 天 + 起报跨度 %d 天 + 预热/并发）",
            role,
            final,
            _lead_window_days(leads),
            max(1, strategy.chunk_days),
        )
    elif final != requested:
        log.info(
            "角色 %s 配置窗宽 %d 天，收到 %d 天（整段时效跨度 %d 天；再宽只多占内存，不减少重读）",
            role,
            requested,
            final,
            _lead_window_days(leads),
        )
    strategy.loads[role] = f"window:{final}"
    return policy, final


def _build_roles(
    spec: PipelineSpec,
    strategy: ExecutionStrategy,
    forecast: Any,
    observation: Any,
    reference: Any,
    init_times: List[datetime],
    leads: List[float],
    n_chunks: int,
    profile: ResourceProfile,
    warmup_hours: float = 0.0,
) -> Dict[str, RoleSpec]:
    """观测 / 参考角色的加载声明（按协议形态给 builder 和全量时刻）。"""
    roles: Dict[str, RoleSpec] = {}
    obs_policy, obs_window = _resolve_window(
        "observation", strategy, n_chunks, profile, leads, warmup_hours
    )

    if spec.protocol in TYPHOON_PROTOCOLS:
        # BABJ 报文一个文件就是一号台风的**整条路径**（跨越多天），没有"按跨度
        # 挑文件"这回事；整包 KB 级，一次读完就是全部。角色必须是 slice：
        # resident 命中缓存后要按请求跨度切时间轴，而报文时间轴是北京时、覆盖
        # 各号台风的完整生命史，按预报起报时刻（UTC）去切会把实况切没。
        if obs_policy != "slice":
            raise ConfigError(
                f"台风协议的观测角色只能是 slice，当前是 {obs_policy!r}："
                f"驻留/分窗都会按请求跨度切 BABJ 的时间轴，而报文时间是北京时、"
                f"跨度是台风的整条生命史，切完实况就对不上了。"
            )
        roles["observation"] = RoleSpec(
            role="observation",
            policy=obs_policy,
            window_days=obs_window,
            builder=partial(_babj_builder, observation),
            times=[],
        )
    elif spec.protocol == "grid_valid_time":
        variables = _observation_variables(spec)
        valid_times = _grid_valid_times(init_times, leads)
        roles["observation"] = _grid_role(
            "observation", observation, variables, valid_times, obs_policy, obs_window
        )
        if reference is not None and reference.catalog is not None:
            ref_policy, ref_window = _resolve_window(
                "reference", strategy, n_chunks, profile, leads, warmup_hours
            )
            roles["reference"] = _grid_role(
                "reference", reference, variables, valid_times, ref_policy, ref_window
            )
    else:
        # 站点协议：观测按跨度选文件（与协议直读同一口径）
        obs_var = spec.variables.get("observation") or spec.observation.params.get(
            "variable"
        )
        if not obs_var:
            raise ConfigError("station_valid_time 协议需要声明 observation 的 variable")
        lead_max = max(leads) if leads else 0.0
        span = station_observation_span(
            init_times,
            spec.local_utc_offset_hours,
            spec.window_hours,
            int(lead_max),
        )
        roles["observation"] = _station_role(
            observation, obs_var, span, obs_policy, obs_window
        )
        # 站点协议的参考源是逐站气候概率（ref_probability 一类），协议在
        # build_batch 里逐样本直读，没有整包可缓存，不进加载层。
    return roles


def _grid_builder(
    role: str,
    handle: Any,
    variables: List[str],
    times: List[datetime],
    span: Optional[Tuple[datetime, datetime]],
):
    """格点角色的 builder：按跨度从全量时刻里筛出请求再读。

    写成模块级函数 + ``partial`` 绑参（而不是闭包）：闭包 pickle 不了，会让
    整个 loader 无法序列化，spawn 平台的子进程就只能退化成逐块直读。
    """
    if span is None:
        selected = list(times)
    else:
        selected = [t for t in times if span[0] <= t <= span[1]]
    if not selected:
        raise ValueError(f"角色 {role} 在跨度 {span} 内没有可请求的时刻")
    request = DataRequest(
        source_id=handle.source_id,
        variables=list(variables),
        init_times=selected,
    )
    return handle.reader.read(request, handle.catalog.discover(request))


def _grid_role(
    role: str,
    handle: Any,
    variables: List[str],
    times: List[datetime],
    policy: str,
    window_days: int,
) -> RoleSpec:
    """格点角色：builder 按跨度从全量时刻里筛出请求再读。"""
    return RoleSpec(
        role=role,
        policy=policy,
        window_days=window_days,
        builder=partial(_grid_builder, role, handle, variables, times),
        times=times,
    )


def _station_builder(
    observation: Any,
    observation_var: str,
    span: Tuple[datetime, datetime],
    block_span: Optional[Tuple[datetime, datetime]],
):
    """站点观测角色的 builder：按跨度挑观测文件（与协议直读同一选法）。"""
    start, end = block_span if block_span is not None else span
    request, selected = select_observation_files(
        observation, observation_var, start, end
    )
    if not selected:
        raise ValueError(f"观测窗口 {start} 到 {end} 内没有找到评估所需的观测文件")
    return observation.reader.read(
        request, DataIndex(source_id=observation.source_id, available=selected)
    )


def _babj_builder(observation: Any, block_span):
    """BABJ 观测角色：不看跨度，整个目录读一次（理由见 ``_build_roles``）。

    模块级函数 + partial 绑参而不是闭包：闭包 pickle 不了，进程池模式下会让
    整个 loader 无法序列化。
    """
    request = DataRequest(source_id=observation.source_id, variables=["storm"])
    return observation.reader.read(request, observation.catalog.discover(request))


def _station_role(
    observation: Any,
    observation_var: str,
    span: Tuple[datetime, datetime],
    policy: str,
    window_days: int,
) -> RoleSpec:
    """站点观测角色：builder 按跨度挑观测文件（与协议直读同一选法）。"""
    # 全量时刻：按小时铺满整个观测窗（window 策略分块用）
    times: List[datetime] = []
    current = span[0]
    while current <= span[1]:
        times.append(current)
        current += timedelta(hours=1)

    return RoleSpec(
        role="observation",
        policy=policy,
        window_days=window_days,
        builder=partial(_station_builder, observation, observation_var, span),
        times=times,
    )
