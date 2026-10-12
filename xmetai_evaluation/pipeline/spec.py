# -*- coding: utf-8 -*-
"""Pipeline 声明。

一套流程 = 数据源 + 协议 + 变换链 + 指标 + 输出，全部是声明：
没有循环、没有 I/O、没有指标公式。真正执行它的是 ``pipeline/runner.py``。

新增一个评测 = 新增一份声明（配置）；只有出现新的物理算法时才新增代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from xmetai_evaluation.core.errors import ConfigError


@dataclass
class SourceSpec:
    """一个数据源的声明：读什么、用哪个 Reader。"""

    reader: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SourceSpec":
        """从 ``{"type": "<reader>", "root_dir": ...}`` 形式的配置构造。

        数据源类型必须由配置显式声明，框架不猜默认模型。
        """
        config = dict(config or {})
        reader = config.get("type") or config.get("reader")
        if not reader:
            raise ConfigError("数据源配置必须声明 type（或 reader）")
        params = {
            key: value
            for key, value in config.items()
            if key not in ("type", "reader")
        }
        params.setdefault("root_dir", config.get("root"))
        return cls(reader=reader, params=params)


@dataclass
class TransformSpec:
    """一个变换的声明。"""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSpec:
    """一个指标的声明。"""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTemplate:
    """一套流程模板：只声明"怎么算"。

    protocol / transforms / metrics / writers 与协议参数（区域、权重、窗口口径）
    都属于流程；"数据从哪来、评哪段时间"属于配置（``--config``）。
    """

    name: str
    protocol: str
    description: str = ""
    transforms: List[TransformSpec] = field(default_factory=list)
    metrics: List[MetricSpec] = field(default_factory=list)
    writers: List[str] = field(default_factory=lambda: ["csv_long"])
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSpec:
    """一次评测 = 流程模板（怎么算）+ 数据与时段（算什么）。

    流程来自 ``pipeline/templates`` 里的具名模板（``--pipeline``）；数据来自配置
    （``--config``）。两者正交，可以自由组合。
    """

    name: str
    forecast: SourceSpec
    observation: SourceSpec
    #: 可选参考场（气候态等），供 ACC / 活跃度 / 功率谱等距平类指标使用
    reference: Optional[SourceSpec] = None
    description: str = ""
    #: 流程模板名（注册在 pipeline/pipelines.py）
    pipeline: str = ""
    #: 解析后的流程模板；不显式给流程时由配置内联生成
    template: Optional[PipelineTemplate] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    #: 多时段评测：[("20250101","20250215"), ("20250601","20250715")]。
    #: 与 start_date/end_date **二选一**（同时给直接报错，理由见 declared_periods）。
    #: 多段共用同一个 output_dir、汇进同一套统计量——不是每段各出一套结果。
    periods: Optional[List[Tuple[str, str]]] = None
    limit: Optional[int] = None
    output_dir: str = "evaluation_results"
    #: 覆盖模板的输出视图（None 表示用模板的）
    writers: Optional[List[str]] = None
    log_level: str = "INFO"
    #: 参与评测的变量：{"forecast": "tp", "observation": "precipitation"}
    variables: Dict[str, str] = field(default_factory=dict)
    #: 配置侧的协议参数覆盖（如观测是北京时 local_utc_offset_hours=8）
    config_options: Dict[str, Any] = field(default_factory=dict)
    #: 模板里变换参数的覆盖：{变换名: {参数: 值}}
    transform_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: 模板里指标参数的覆盖：{指标名: {参数: 值}}
    metric_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def protocol(self) -> str:
        return self.template.protocol if self.template else "station_valid_time"

    @property
    def transforms(self) -> List[TransformSpec]:
        if self.template is None:
            return []
        return [
            TransformSpec(
                name=item.name,
                params={**item.params, **self.transform_options.get(item.name, {})},
            )
            for item in self.template.transforms
        ]

    @property
    def metrics(self) -> List[MetricSpec]:
        if self.template is None:
            return []
        return [
            MetricSpec(
                name=item.name,
                params={**item.params, **self.metric_options.get(item.name, {})},
            )
            for item in self.template.metrics
        ]

    @property
    def output_writers(self) -> List[str]:
        if self.writers is not None:
            return list(self.writers)
        if self.template is not None:
            return list(self.template.writers)
        return ["csv_long"]

    @property
    def options(self) -> Dict[str, Any]:
        """协议参数：模板默认口径 + 配置覆盖。"""
        merged = dict(self.template.options) if self.template else {}
        merged.update(self.config_options)
        return merged

    def use_pipeline(self, name: str) -> None:
        """切换到具名流程模板（``--pipeline``）。"""
        from xmetai_evaluation.pipeline.pipelines import get_template

        self.pipeline = name
        self.template = get_template(name)

    def transform(self, name: str) -> Optional[TransformSpec]:
        for item in self.transforms:
            if item.name == name:
                return item
        return None

    @property
    def window_hours(self) -> int:
        """时间窗口长度（小时）；优先取变换参数。"""
        override = self.options.get("window_hours")
        if override is not None:
            return int(override)
        item = self.transform("time_window_accumulator")
        if item is not None:
            return int(item.params.get("window_hours", 24))
        return 24

    @property
    def local_utc_offset_hours(self) -> float:
        """观测时刻相对 UTC 的偏移（北京时 = +8）。"""
        return float(self.options.get("local_utc_offset_hours", 8.0))

    def declared_periods(self) -> List[Tuple[datetime, datetime]]:
        """本段要评的时段列表（解析成 datetime 对）。

        ``periods`` 与 ``start_date``/``end_date`` **二选一**：同时给直接报错，
        不让 ``periods`` 静默盖掉 start/end——配置里躺着两套时段、没人知道哪套
        生效，是最难查的一类问题。

        单时段走 start_date/end_date，返回长度 1 的列表，下游不用分支。
        """
        if self.periods:
            if self.start_date or self.end_date:
                raise ConfigError(
                    "start_date/end_date 与 periods 不能同时给：多时段请删掉 "
                    "start_date/end_date，单时段请删掉 periods"
                )
            parsed: List[Tuple[datetime, datetime]] = []
            for item in self.periods:
                pair = tuple(item or ())
                if len(pair) != 2:
                    raise ConfigError(
                        f"periods 的每一项应是 (起始, 结束) 两个日期，收到 {item!r}"
                    )
                try:
                    start = _parse_date(pair[0], "periods 起始")
                    end = _parse_date(pair[1], "periods 结束")
                except ValueError as error:
                    # 统一成 ConfigError：periods 的所有毛病都是配置毛病，
                    # 别让"少写个日期"和"日期写错"报出两种异常类型。
                    raise ConfigError(
                        f"periods 的时段 {item!r} 日期解析失败：{error}"
                    ) from error
                if end < start:
                    raise ConfigError(f"periods 的时段 {item!r} 起止颠倒")
                parsed.append((start, end))
            return parsed
        if not self.start_date:
            raise ValueError("评测配置必须提供 start_date 或 periods")
        start = _parse_date(self.start_date, "start_date")
        end = _parse_date(self.end_date, "end_date") if self.end_date else start
        return [(start, end)]

    def period(self) -> Tuple[datetime, datetime]:
        """整体时段的 (最早, 最晚)。

        多段时是一个**覆盖全部段**的包络区间，只用于报错信息与展示——真正驱动
        切块的是 ``init_candidate_times()``，时段本身不参与别的计算。
        """
        periods = self.declared_periods()
        return min(start for start, _ in periods), max(end for _, end in periods)

    def init_candidate_times(self) -> List[datetime]:
        """候选起报时刻：逐段按天铺开后并集去重。

        **这是"多时段"唯一的落点。** 计划层与各协议 ``prepare`` 都只拿这个列表
        去 catalog 探测实际可用的起报，之后切块、时效窗、观测跨度、指标全部只认
        探测出来的 ``init_times``——所以多段与"跑一个覆盖全部段的大区间再手工剔掉
        空档期起报"完全等价，省下的只是空档期那些块的算力。
        """
        values: set = set()
        for start, end in self.declared_periods():
            values.update(daily_times(start, end))
        return sorted(values)

    @classmethod
    def from_config(cls, cfg: Any, pipeline: Optional[str] = None) -> "PipelineSpec":
        """从 ``configs/*.py`` 的 ``EvalConfig`` 构造声明。

        Args:
            cfg: 配置对象。
            pipeline: 指定流程模板名；不给就用配置里声明的那个。配置里的
                ``pipeline`` 可以是列表（多段），那时由 ``specs_from_config``
                逐段调用本方法并把名字传进来。

        Raises:
            ConfigError: 配置没有声明任何流程模板。
        """
        forecast_config = dict(cfg.forecast_reader)
        observation_config = dict(cfg.observation_reader)
        variables: Dict[str, str] = {}
        for key, config in (("forecast", forecast_config), ("observation", observation_config)):
            names = config.get("variables")
            if names:
                variables[key] = list(names)[0]
            elif config.get("variable"):
                variables[key] = config["variable"]

        from xmetai_evaluation.pipeline.pipelines import get_template

        pipeline_name = str(pipeline or "").strip() or pipeline_names(cfg)[0]
        template = get_template(pipeline_name)

        return cls(
            name=cfg.name,
            description=cfg.description,
            forecast=SourceSpec.from_config(forecast_config),
            observation=SourceSpec.from_config(observation_config),
            reference=(
                SourceSpec.from_config(cfg.reference_reader)
                if getattr(cfg, "reference_reader", None)
                else None
            ),
            pipeline=pipeline_name,
            template=template,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            periods=getattr(cfg, "periods", None),
            limit=cfg.limit,
            output_dir=cfg.output_dir,
            writers=(
                list(cfg.writers) if getattr(cfg, "writers", None) is not None else None
            ),
            log_level=getattr(cfg, "log_level", "INFO"),
            variables=variables,
            config_options=dict(getattr(cfg, "options", None) or {}),
            transform_options=dict(getattr(cfg, "transform_options", None) or {}),
            metric_options=dict(getattr(cfg, "metric_options", None) or {}),
        )


def pipeline_names(cfg: Any) -> List[str]:
    """配置声明的流程模板名列表。

    ``pipeline`` 写一个名字就是一段；写一串就是按顺序跑多段（比如集合降水
    检验要 24h 的 TS 和 6h 的概率评分两套窗口，见 ``pipelines.py``）。

    Raises:
        ConfigError: 一个流程名都没声明。
    """
    raw = getattr(cfg, "pipeline", "") or ""
    if isinstance(raw, str):
        names = [raw.strip()] if raw.strip() else []
    else:
        names = [str(name).strip() for name in raw if str(name).strip()]
    if not names:
        from xmetai_evaluation.pipeline.pipelines import list_pipelines

        raise ConfigError(
            f"配置 {cfg.name} 没有声明 pipeline（要走哪套流程）。"
            f"可用流程: {', '.join(sorted(list_pipelines()))}"
        )
    return names


def specs_from_config(cfg: Any) -> List[PipelineSpec]:
    """把一份配置展开成要跑的若干段（每段一个流程模板）。

    多段共用一个 ``output_dir``：结果合并后由 Runner 落盘一次，长表里靠
    ``window_h`` 之类的列区分是哪一段算的。
    """
    return [PipelineSpec.from_config(cfg, name) for name in pipeline_names(cfg)]


def _parse_date(value: str, field_name: str) -> datetime:
    for fmt in ("%Y%m%d", "%Y%m%d%H", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    raise ValueError(f"无效的 {field_name}: {value}")


def daily_times(start: datetime, end: datetime) -> List[datetime]:
    """[start, end] 区间内逐日时刻列表。"""
    times: List[datetime] = []
    current = start
    while current <= end:
        times.append(current)
        current += timedelta(days=1)
    return times


def hourly_times(start: datetime, end: datetime) -> List[datetime]:
    """[start, end] 区间内逐小时时刻列表（站点观测的请求跨度用）。"""
    times: List[datetime] = []
    current = start
    while current <= end:
        times.append(current)
        current += timedelta(hours=1)
    return times
