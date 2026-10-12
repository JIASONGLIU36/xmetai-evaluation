# -*- coding: utf-8 -*-
"""2025 年暴雨过程个例检验（FuXi 确定性 × Diamond 站点）。

口径与 ``weather_ts_single_fgvp`` 完全一致——08–08 北京时 24h 窗、同一份站表
（``zd_sta_10285.dat``）、同一套阈值（含 ≥250 补档）；只是把"连续时段"
换成纪要表的"逐过程"：43 个过程 × 2 种口径。**不分纬度带，只出全球行。**

两种口径（纪要表过程记为 S 日至 E 日，0–0 日历口径）：

口径 A（逐日 24h，``*_24h/{起报日}/``）：[S-1, E] 逐日 00UTC 起报，只评
    lead 24——每个 08–08 窗由当日起报的 day-1 预报评一次，不重复计数
    （窗 = 起报日 08 时 → 次日 08 时）。框架对多起报会把列联计数合并成
    一行，"每天一个结果"因此拆成单起报段：每个起报日单独一段、单独
    一个输出目录，互不合并。

口径 B（过程累积，``*_total``）：S-1 日 00UTC 单起报，窗长 = 过程全长
    = (E-S+2) 天 × 24h，窗为 [S-1 日 08:00, E+1 日 08:00]。比纪要表的
    0–0 日历口径头尾各多 8 小时，是统一 08–08 日界的代价（先统一不调整）。

过程表来自 ``xmetai_evaluation/rainstorm_catalog.py``（源头是
《2025年暴雨过程纪要表（修订）-1.docx》2025-10-08 版，编号 202501–202543）。
注意 202519 与 202520 起止完全相同（纪要表把同一时段的两个侧面分列两条），
各评各的、不合并。

先冒烟再正式：``SMOKE = True`` 只跑 202521（特强过程）一个过程；
跑通后改回 ``False`` 全量。所有路径是字面量，换机器直接改本文件的常量。
"""
from datetime import datetime, timedelta

from xmetai_evaluation.configs.base import EvalConfig
from xmetai_evaluation.rainstorm_catalog import process_tuples

#: 与 weather_ts_single_fgvp 同一套阈值：模板 5 档 + ≥250mm 极端量级补档。
TS_THRESHOLDS = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0]

#: 冒烟开关：True = 只跑 202521 一个过程；False = 全部 43 个。
SMOKE = False

#: (编号, 开始日, 结束日)，日期 YYYYMMDD，纪要表 0–0 日历口径。
#: 编目真值在 ``xmetai_evaluation.rainstorm_catalog``，本文件不再自带一份。
PROCESSES = process_tuples()

#: 数据源与输出根，字面量（换机器直接改这里）。
FUXI_ROOT = "/workspace/data/shenzw/fuxi_single_output"
STATION_ROOT = "/workspace/data/worm/r0/2025"
STATION_LIST = "/workspace/data/worm/r0/zd_sta_10285.dat"
OUTPUT_ROOT = "/workspace/szwCode/evaluation_results/rainstorm_ts_single_fuxi"

def _dense_leads(max_hours):
    """按 6h 密排到 max_hours（24h 窗是 4 个连续 6h 步加出来的，必须密排）。"""
    return [float(h) for h in range(6, int(max_hours) + 1, 6)]


cfgs = []
for _code, _start, _end in PROCESSES:
    _s = datetime.strptime(_start, "%Y%m%d")
    _e = datetime.strptime(_end, "%Y%m%d")
    _prev = (_s - timedelta(days=1)).strftime("%Y%m%d")
    _span_days = (_e - _s).days + 2  # 08–08 口径的全长天数（头尾各多 8 小时）
    _window_hours = _span_days * 24

    if SMOKE and _code != "202521":
        continue

    # ── 口径 A：逐日 24h（每个起报日单独一段、单独一个目录）───────────
    # 框架对多起报会把列联计数合并成一行，"每天一个结果"靠单起报段实现。
    # lead_times 截到 24：默认采样规则（≥窗长 且 窗长整数倍）在
    # {6,12,18,24} 里恰好只筛出 24，无需显式声明 sample_leads。
    _d = _s - timedelta(days=1)
    while _d <= _e:
        _init = _d.strftime("%Y%m%d")
        cfgs.append(
            EvalConfig(
                name=f"rainstorm_fuxi_{_code}_24h_{_init}",
                description=f"{_code} 过程 {_init} 起报 day-1 24h TS（{_init}08时→次日08时）",
                pipeline="weather_ts_det",
                forecast_reader={
                    "type": "fuxi",  # fuxi 目录布局：root/YYYYMMDD/NNN.nc
                    "root_dir": FUXI_ROOT,
                    "variable": "tp",
                    "step_hours": 6.0,
                    "lead_times": [6, 12, 18, 24],
                },
                observation_reader={
                    "type": "station",
                    "root_dir": STATION_ROOT,
                    "variable": "precipitation",
                    "station_list": STATION_LIST,
                },
                transform_options={"time_window_accumulator": {"window_hours": 24}},
                metric_options={"ts_score": {"thresholds": TS_THRESHOLDS}},
                writers=["csv_long", "categorical_wide"],
                options={"local_utc_offset_hours": 8},  # 不分带，只出全球行
                start_date=_init,
                end_date=_init,  # 单起报：这个 08–08 窗只由这一根预报评
                limit=None,  # 限起报数，直接改这里；None = 不限
                output_dir=f"{OUTPUT_ROOT}/{_code}_24h/{_init}",
                log_level="INFO",
                execution={
                    "mode": "threads",  # 轻指标 → threads（同 weather_ts_single_fgvp）
                    "n_workers": 24,
                    "chunk_days": 1,
                    "lead_chunk_days": 1,  # 1 天时效 = 1 个 24h 采样点
                    "loads": {"observation": "resident"},
                    "resume": True,
                },
            )
        )
        _d += timedelta(days=1)

    # ── 口径 B：过程累积（S-1 日单起报，窗长 = 过程全长）───────────────
    # lead_times 密排到窗长，默认采样规则恰好只筛出窗长本身。
    # lead_chunk_days 给到窗长天数，一个块装下整窗（小了只是告警不报错，
    # 但整窗切块没有意义）。
    cfgs.append(
        EvalConfig(
            name=f"rainstorm_fuxi_{_code}_total",
            description=f"{_code} 过程累积 TS（{_prev} 08时 起 {_span_days} 天）",
            pipeline="weather_ts_det",
            forecast_reader={
                "type": "fuxi",
                "root_dir": FUXI_ROOT,
                "variable": "tp",
                "step_hours": 6.0,
                "lead_times": _dense_leads(_window_hours),
            },
            observation_reader={
                "type": "station",
                "root_dir": STATION_ROOT,
                "variable": "precipitation",
                "station_list": STATION_LIST,
            },
            transform_options={
                "time_window_accumulator": {"window_hours": _window_hours}
            },
            metric_options={"ts_score": {"thresholds": TS_THRESHOLDS}},
            writers=["csv_long", "categorical_wide"],
            options={"local_utc_offset_hours": 8},  # 不分带，只出全球行
            start_date=_prev,
            end_date=_prev,  # 单起报日
            limit=None,
            output_dir=f"{OUTPUT_ROOT}/{_code}_total",
            log_level="INFO",
            execution={
                "mode": "threads",
                "n_workers": 24,
                "chunk_days": 1,
                "lead_chunk_days": _span_days,  # 一块装下整个累积窗
                "loads": {"observation": "resident"},
                "resume": True,
            },
        )
    )
