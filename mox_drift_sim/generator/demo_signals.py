# mox_drift_sim/generator/demo_signals.py
import numpy as np
import pandas as pd


def _time_index(start: str, periods: int, freq: str):
    """简单时间索引工具函数。"""
    return pd.date_range(start=start, periods=periods, freq=freq)


def make_warmup_and_field():
    """
    生成一个“好看版”的 warm-up + field 信号，用于画图展示。

    - warm-up: 从 0.2 * baseline 指数式上升到 baseline
    - field: 以很小的斜率漂移（模拟 aging / humidity drift）
    """
    start = "2025-01-01 00:00:00"
    freq = "1min"

    n_warm = 60          # 1 小时 warm-up
    n_field = 4 * 60     # 4 小时 field

    idx_warm = _time_index(start, n_warm, freq)
    idx_field = _time_index(idx_warm[-1] + pd.Timedelta(minutes=1), n_field, freq)

    rng = np.random.default_rng(42)

    baseline = 1.0
    tau = 20.0  # warm-up 时间常数（越小上升越快）
    alpha = 1.0 - np.exp(-1.0 / tau)

    warm = np.zeros(n_warm, dtype=float)
    warm[0] = 0.2 * baseline

    for t in range(1, n_warm):
        warm[t] = warm[t - 1] + alpha * (baseline - warm[t - 1])

    warm += rng.normal(0.0, 0.01, size=n_warm)

    # field 段：在 baseline 附近缓慢上升（drift）
    drift_per_min = 0.0003  # 每分钟的微小漂移
    field = baseline + drift_per_min * np.arange(n_field)
    field += rng.normal(0.0, 0.01, size=n_field)

    df_warm = pd.DataFrame({"timestamp": idx_warm, "sensor_value": warm})
    df_field = pd.DataFrame({"timestamp": idx_field, "sensor_value": field})
    return df_warm, df_field


def make_humidity_coupling():
    """
    生成一个带有“日周期 + 噪声”的湿度信号，以及与之耦合的传感器输出。
    """
    start = "2025-01-01 01:00:00"
    freq = "1min"
    n = 4 * 60  # 4 小时

    idx = _time_index(start, n, freq)
    rng = np.random.default_rng(123)

    base_h = 50.0
    amplitude = 10.0

    t = np.arange(n)
    # 近似日周期（这里只截取 4 小时）
    humidity = base_h + amplitude * np.sin(2 * np.pi * t / (24 * 60))
    humidity += rng.normal(0.0, 1.0, size=n)

    # 传感器对湿度的线性耦合
    sensor = 1.0 + 0.01 * (humidity - base_h)
    sensor += rng.normal(0.0, 0.01, size=n)

    return pd.DataFrame(
        {
            "timestamp": idx,
            "sensor_value": sensor,
            "humidity": humidity,
        }
    )


def make_stress_drift():
    """
    生成一个带有“baseline 突然跳变”的 stress drift 示例。
    """
    start = "2025-01-01 00:00:00"
    freq = "5min"
    n = 12 * 12  # 12 小时，5 分钟一个点

    idx = _time_index(start, n, freq)
    rng = np.random.default_rng(7)

    baseline = 1.0
    sensor = baseline + rng.normal(0.0, 0.01, size=n)

    step1 = int(n / 3)
    step2 = int(2 * n / 3)

    # 第一次应力事件：整体上移 0.15
    sensor[step1:] += 0.15
    # 第二次应力事件：整体下降 0.10
    sensor[step2:] -= 0.10

    return pd.DataFrame({"timestamp": idx, "sensor_value": sensor})
