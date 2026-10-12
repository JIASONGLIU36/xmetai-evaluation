"""
时间累积 Transform

支持滑动窗口累积（如 6h/24h 累积降水）
"""

import numpy as np
import xarray as xr
import pandas as pd
import logging
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


def window_sum_at(
    dataset: xr.Dataset,
    variable: str,
    end_time: datetime,
    window_hours: int,
    time_dim: str = "time",
    require_complete: bool = True,
) -> xr.DataArray:
    """取闭合窗口 [end_time - window_hours + 1h, end_time] 内的累计量。

    用于把逐时实况聚合到预报窗口（如 24h 累积降水），保证预报与观测口径一致。
    窗口缺任一时次即报错；站级缺测的处置看 require_complete；0 是有效记录，不参与缺测判定。

    Args:
        dataset: 含时间维的观测数据。
        variable: 需要累计的变量名。
        end_time: 窗口结束时刻。
        window_hours: 窗口长度（小时）。
        time_dim: 时间维度名。
        require_complete: True 时站级缺测即把该站置为缺测；
            False 时站级缺测按 0 参与累计，但只保留窗口末端时次存在的样本
            （与参考实现 ``read_rain_obs`` 的 fillna(0) 口径一致）。
            两种"缺"不是一回事：整时次不在观测索引里（文件缺失 / 解析失败 /
            白名单过滤后为空）**两种口径都直接报错**，与这个开关无关。

    Returns:
        窗口累计值，缺失窗口为 NaN。

    Raises:
        ValueError: 观测缺少窗口内的任一时次。
    """
    expected = [end_time - timedelta(hours=i) for i in range(window_hours - 1, -1, -1)]
    available = pd.DatetimeIndex(dataset[time_dim].values)
    positions = available.get_indexer(np.asarray(expected, dtype="datetime64[ns]"))
    if np.any(positions < 0):
        missing = [str(expected[i]) for i, position in enumerate(positions) if position < 0]
        raise ValueError(
            f"{end_time} 的 {window_hours}h 观测窗口不完整，"
            f"缺少 {len(missing)} 个时次（如 {missing[:3]}）"
        )

    values = dataset[variable].isel({time_dim: positions})
    complete = np.isfinite(values).all(dim=time_dim)
    if require_complete:
        return values.sum(dim=time_dim, skipna=False).where(complete)
    present = np.isfinite(values.isel({time_dim: -1}))
    # 与参考实现同序累加（从有效时刻往前）：浮点求和顺序会影响恰好落在阈值上的站点，
    # 参考口径下必须同序才能逐位一致。
    ordered = values.isel({time_dim: slice(None, None, -1)})
    # 站级缺测按 0 累计是**配置选择**（observation_window=reference），不是数据异常：
    # 它写在配置里、也进了 manifest.json 的 resolved_config，所以这里不再对每次调用
    # 刷警告——正式跑会按「起报 x 时效」喊几万次，把日志淹掉。只有真的把 NaN 折成 0
    # 时留一条 DEBUG，排障时开 log_level=DEBUG 就能看到有多少个站的窗口不完整。
    n_incomplete = int(np.count_nonzero(~complete.values))
    if n_incomplete:
        log.debug(
            "%s 的 %dh 观测窗口：%d 个站的窗口内含缺测，已按 0 累计（reference 口径）",
            end_time,
            window_hours,
            n_incomplete,
        )
    return ordered.sum(dim=time_dim, skipna=True).where(present)


class TimeWindowAccumulator:
    """
    时间窗口累积

    将逐时效数据累积为固定窗口（如 6h/24h）。

    示例：
        输入：逐 6h 时效降水 [6h, 12h, 18h, 24h, ...]
        窗口：24h
        输出：[0-24h累积, 6-30h累积, 12-36h累积, ...]
    """

    def __init__(self, window_hours: float, time_dim: str = "lead_time"):
        """
        Args:
            window_hours: 累积窗口（小时）
            time_dim: 时间维度名称，默认 'lead_time'
        """
        self.window_hours = window_hours
        self.time_dim = time_dim

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """
        累积时间窗口

        Args:
            data: 输入数据，必须包含时间维度（lead_time 或 time）

        Returns:
            累积后的数据，保留原始时间坐标但值为累积值
        """
        if self.time_dim not in data.dims:
            raise ValueError(f"Data must have '{self.time_dim}' dimension")

        # 获取时间坐标（单位：小时）
        time_coord = data.coords[self.time_dim].values

        # 计算步长
        if len(time_coord) < 2:
            raise ValueError("Need at least 2 time steps for accumulation")

        time_diffs = np.diff(time_coord)
        if not np.allclose(time_diffs, time_diffs[0], rtol=1e-6):
            raise ValueError("Time steps must be uniform")

        step_hours = float(time_diffs[0])

        # 计算窗口包含多少个时间步
        n_steps_per_window = int(np.round(self.window_hours / step_hours))

        if n_steps_per_window < 1:
            raise ValueError(f"Window ({self.window_hours}h) smaller than step ({step_hours}h)")

        # 使用滑动窗口累积
        # 对于每个时刻 t，累积 [t-window+step : t] 的数据
        n_times = len(time_coord)
        accumulated = []

        for i in range(n_times):
            # 累积窗口：从 max(0, i-n_steps_per_window+1) 到 i（包含）
            start_idx = max(0, i - n_steps_per_window + 1)
            end_idx = i + 1  # 不包含，所以是 i+1

            # 如果窗口不完整（开头几个时刻），跳过
            if end_idx - start_idx < n_steps_per_window:
                continue

            # 累积该窗口
            window_slice = {self.time_dim: slice(start_idx, end_idx)}
            window_sum = data.isel(window_slice).sum(dim=self.time_dim, skipna=False)

            # 添加时间坐标（累积窗口的结束时刻）
            window_sum = window_sum.expand_dims({self.time_dim: [time_coord[i]]})
            accumulated.append(window_sum)

        if not accumulated:
            raise ValueError(
                f"No complete windows: need at least {n_steps_per_window} time steps "
                f"for {self.window_hours}h window with {step_hours}h step"
            )

        # 拼接所有累积结果
        # coords/compat 双双写死：xarray 的弃用预告成对出现，只写一个它会接着
        # 问下一个；两个值都是当前默认，行为不变。
        result = xr.concat(
            accumulated, dim=self.time_dim, coords="different", compat="equals"
        )

        # 添加属性标记这是累积值
        result.attrs = data.attrs.copy()
        result.attrs["accumulation_window_hours"] = self.window_hours
        result.attrs["accumulation_method"] = "sliding_window"

        return result
