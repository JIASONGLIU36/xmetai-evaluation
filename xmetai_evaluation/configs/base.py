# -*- coding: utf-8 -*-
"""评估配置基础设施：EvalConfig + load_config()"""
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class EvalConfig:
    """一次评估的完整配置（由 cli.py 加载并执行）"""

    name: str  # 配置名称
    description: str  # 配置描述

    # 数据源
    forecast_reader: Dict[str, Any]  # 预报数据读取器配置
    observation_reader: Dict[str, Any]  # 观测数据读取器配置
    # 可选参考场（气候态等）：ACC / 活跃度 / 功率谱等距平类指标需要
    reference_reader: Optional[Dict[str, Any]] = None

    # 时间范围
    start_date: Optional[str] = None  # YYYYMMDD or YYYYMMDDHH
    end_date: Optional[str] = None  # YYYYMMDD or YYYYMMDDHH
    # 多时段评测：[(起始, 结束), ...]，如
    #     periods=[("20250101", "20250215"), ("20250601", "20250715")]
    # 与 start_date/end_date **二选一**（同时给会报配置错）。
    # 多段共用同一个 output_dir、汇进同一套统计量（每个 变量×时效×阈值 只有
    # 一行，不是每段各一行）；段与段之间的空档不切块、不出样本。
    periods: Optional[List[Tuple[str, str]]] = None
    limit: Optional[int] = None  # 限制处理的初始化时间数（用于调试）

    # 输出
    output_dir: str = "evaluation_results"  # 输出目录

    # 结果输出视图：统一长表（csv_long）始终写出；
    # None = 用流程模板的视图，显式给列表则覆盖模板
    writers: Optional[List[str]] = None

    # 流程名（pipeline/pipelines.py 里的模板）；给了就用它的协议/变换/指标。
    # 写成列表就是按顺序跑多段（如集合降水检验两次窗口），结果合并落同一个 output_dir
    pipeline: Union[str, List[str]] = ""

    # 协议口径覆盖（如观测为北京时：local_utc_offset_hours=8）
    options: Dict[str, Any] = field(default_factory=dict)

    # 流程模板里"变换参数"的覆盖：{"time_window_accumulator": {"window_hours": 6}}
    transform_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 流程模板里"指标参数"的覆盖：{"ts_score": {"thresholds": [0.1, 10, 25]}}
    metric_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 运行配置
    log_level: str = "INFO"  # 日志级别
    num_workers: int = 1  # 并行 worker 数（execution 没写 n_workers 且它 > 1 时生效）

    # 执行策略（并发与数据加载）：不写就按指标族与数据源形态推导缺省。
    # 全部键字面量，只覆盖写到的键：
    #   mode: "serial" | "threads" | "processes" | "auto"（默认 auto：
    #         单块=serial，重指标=processes，轻指标=threads）
    #   n_workers: 进程/线程数（缺省 threads=4、processes=CPU 数）
    #   chunk_days: 一个工作块装几个起报日（默认 1）——管起报跨度
    #   lead_chunk_days: 一个工作块装几天时效（默认 1）——管时效跨度，是单块
    #         内存的主旋钮；0 = 不切（整段时效一块，与改造前逐位一致，逃生口）
    #   loads: {"observation": "slice"|"resident"|"window:30"|"window",
    #           "reference": ...}（缺省：站点观测/气候态驻留，其余逐块直读；
    #           裸 "window" = 按单块观测跨度自动定窗）
    #   resume: True 时已完成块的指标状态落 output_dir/.states/，重跑跳过
    execution: Dict[str, Any] = field(default_factory=dict)

    # 内部字段
    _source_path: str = field(init=False, repr=False, default="")

def metric_options_from_var_metrics(
    var_metrics: Dict[str, List[str]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """把「变量 -> 指标」翻译成 ``EvalConfig.metric_options``。

    参考实现用 ``--var-metrics z500:rmse,acc`` 声明"哪个变量算哪些指标"；
    框架里的对应声明是 ``metric_options[指标名]["variables"] = [变量...]``。
    这里做一次反转，配置就能直接照抄参考实现的形状：

        VAR_METRICS = {"z500": ["rmse", "acc"], "t2m": ["rmse"]}
        metric_options = metric_options_from_var_metrics(VAR_METRICS)

    没被任何变量点名的指标不会出现在结果里；想要某个指标评全部变量，
    就在 ``var_metrics`` 里给每个变量都写上它。

    Args:
        var_metrics: 变量 -> 指标名列表（顺序即结果里的变量顺序）。
        metrics: 要输出哪些指标；默认取 ``var_metrics`` 里出现过的全部。
    """
    names = (
        list(metrics)
        if metrics
        else [name for items in var_metrics.values() for name in items]
    )
    ordered: List[str] = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    return {
        name: {
            "variables": [
                variable for variable, items in var_metrics.items() if name in items
            ]
        }
        for name in ordered
    }


def _resolve_config_path(value: str) -> Path:
    """解析外部配置路径或包内配置名"""
    direct = Path(value).resolve()
    if direct.is_file():
        return direct

    # 尝试在 configs/ 目录下查找
    name = value
    if name.startswith("configs/") or name.startswith("configs\\"):
        name = name.replace("\\", "/").rsplit("/", 1)[-1]

    if not name.endswith(".py"):
        name += ".py"

    packaged = Path(__file__).parent / name
    if packaged.is_file():
        return packaged

    raise FileNotFoundError(f"配置文件不存在: {value}")


def load_config(path: str) -> Union[EvalConfig, List[EvalConfig]]:
    """加载外部配置文件或包内配置名。

    模块里定义 ``cfg = EvalConfig(...)`` 返回单个；定义
    ``cfgs = [EvalConfig(...), ...]``（复数，非空列表）返回列表，
    由 cli 顺序逐个执行——一份文件批量评多个模型用这种。
    两个都定义是配置错误，二选一。
    """
    config_path = _resolve_config_path(path)

    config_dir = str(config_path.parent)
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)

    module_name = f"_xmetai_eval_config_{abs(hash(str(config_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载配置文件: {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    cfg = getattr(module, "cfg", None)
    cfgs = getattr(module, "cfgs", None)
    if cfg is not None and cfgs is not None:
        raise ValueError(f"配置文件 {config_path} 同时定义了 `cfg` 和 `cfgs`，二选一")

    if cfgs is not None:
        if not isinstance(cfgs, (list, tuple)) or not cfgs:
            raise ValueError(
                f"配置文件 {config_path} 的 `cfgs` 应是非空的 EvalConfig 列表"
            )
        for item in cfgs:
            if not isinstance(item, EvalConfig):
                raise TypeError(
                    f"配置文件 {config_path} 的 cfgs 里混有 "
                    f"{type(item).__name__}，应为 EvalConfig"
                )
        for item in cfgs:
            item._source_path = str(config_path)
        return list(cfgs)

    if cfg is None:
        raise ValueError(
            f"配置文件 {config_path} 没有定义 `cfg` 或 `cfgs`"
            "（应写 cfg = EvalConfig(...) 或 cfgs = [EvalConfig(...), ...]）"
        )

    if not isinstance(cfg, EvalConfig):
        raise TypeError(
            f"配置文件 {config_path} 的 cfg 类型是 {type(cfg).__name__}，应为 EvalConfig"
        )

    cfg._source_path = str(config_path)
    return cfg
