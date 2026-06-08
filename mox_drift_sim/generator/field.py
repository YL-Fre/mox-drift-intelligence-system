from __future__ import annotations
import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from datetime import timedelta
from typing import Dict
from ..config import LifecycleConfig
from ..schema import SIMULATION_SCHEMA
from . import utils


def time_index_field(cfg, start):
    """
    为 FIELD 段生成时间索引。

    这里强制：
    - periods 转成整数
    - periods 至少为 1
    这样可以避免 idx 为空导致 idx[0] 抛 IndexError，
    同时也消除 pandas 对非整数 periods 的 FutureWarning。
    """
    # 原本的 periods 计算逻辑（按你的代码写法保留）：
    # 假设是按小时 * 60 分钟 * 采样率来算
    periods = cfg.field_hours * 60 * cfg.sampling_rate_hz

    # 强制转成整数，并确保至少为 1
    periods_int = max(1, int(periods))

    return pd.date_range(start=start, periods=periods_int, freq=cfg.freq)



def generate_field_modes(idx: DatetimeIndex, cfg: LifecycleConfig) -> pd.Series:
    rng = np.random.default_rng(cfg.random_seed + 1)
    modes = np.full(len(idx), "NORM", dtype=object)

    df = pd.DataFrame({"time": idx})
    df["date"] = df["time"].dt.date
    unique_dates = df["date"].unique()

    for d in unique_dates:
        mask = df["date"].values == d
        day_idx = np.where(mask)[0]
        if not len(day_idx):
            continue

        cal_len = min(120, len(day_idx))  # 2 hours
        modes[day_idx[:cal_len]] = "CAL"

        if rng.random() < cfg.prob_stress_day:
            day_len = len(day_idx)
            stress_hours = rng.integers(
                cfg.stress_duration_hours_min,
                cfg.stress_duration_hours_max + 1,
            )
            stress_len = stress_hours * 60
            if stress_len < day_len - cal_len:
                start_offset = rng.integers(cal_len, day_len - stress_len)
                stress_idx = day_idx[start_offset : start_offset + stress_len]
                modes[stress_idx] = "STRESS"

    return pd.Series(modes, index=idx, name="mode")


def simulate_field_segment(
    cfg: LifecycleConfig,
    warmup_end: pd.Timestamp,
    initial_baseline: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed + 2)
    idx = time_index_field(cfg, warmup_end)
    n = len(idx)

    modes = generate_field_modes(idx, cfg)

    minutes = (idx - idx[0]).total_seconds() / 60.0
    temp = (
        cfg.temp_mean
        + cfg.temp_amplitude
        * np.sin(2 * np.pi * (minutes % (24 * 60)) / (24 * 60))
        + rng.normal(0.0, cfg.temp_noise_std, size=n)
    )

    rh = utils.random_walk_around(rng, n, cfg.rh_mean, cfg.rh_rw_std)
    rh[modes.values == "STRESS"] += cfg.rh_stress_boost
    rh = np.clip(rh, 10.0, 95.0)

    gas_1 = utils.generate_gas_profile(rng, n, modes, base_level=5.0)

    baseline = np.zeros(n)
    baseline[0] = initial_baseline

    drift_flag = np.zeros(n, dtype=int)
    fault_flag = np.zeros(n, dtype=int)

    df_dates = pd.DataFrame({"time": idx})
    df_dates["date"] = df_dates["time"].dt.date
    unique_dates = df_dates["date"].unique()

    stress_offsets_by_day: Dict = {}
    for d in unique_dates:
        if rng.random() < cfg.prob_stress_drift_day:
            stress_offsets_by_day[d] = cfg.stress_step

    current_stress_offset = 0.0
    sensor_dyn = 0.0
    sensor_values = np.zeros(n)

    for t in range(1, n):
        mode = modes.iloc[t]
        prev_baseline = baseline[t - 1]

        drift_rate = cfg.drift_rate_field_base
        if mode == "STRESS":
            drift_rate += cfg.drift_rate_field_stress
        rw = rng.normal(0.0, cfg.baseline_rw_std_field)
        humidity_effect = cfg.k_h * (rh[t] - cfg.humidity_ref)

        new_baseline = prev_baseline + drift_rate + rw + humidity_effect

        current_date = idx[t].date()
        if current_date in stress_offsets_by_day and mode == "STRESS":
            step = stress_offsets_by_day.pop(current_date)
            current_stress_offset += step

        baseline[t] = new_baseline + current_stress_offset

        if current_stress_offset != 0.0:
            drift_flag[t] = 1

        sensor_eq = utils.gas_response(
            gas_ppm=gas_1[t],
            temp=temp[t],
            gas_base_level=cfg.gas_base_level,
            gas_exponent=cfg.gas_exponent,
            temp_mean=cfg.temp_mean,
        )
        sensor_dyn = sensor_dyn + (sensor_eq - sensor_dyn) / cfg.tau_field

        noise = rng.normal(0.0, cfg.noise_std_field)
        sensor_values[t] = baseline[t] + sensor_dyn + noise

    sensor_values[0] = baseline[0] + rng.normal(0.0, cfg.noise_std_field)

    df = pd.DataFrame(
        {
            "sensor_id": 1,
            "phase": "FIELD",
            "mode": modes.values,
            "time": idx,
            "sensor_value": sensor_values,
            "temperature": temp,
            "humidity": rh,
            "gas_1_ppm": gas_1,
            "gas_2_ppm": 0.0,
            "gas_3_ppm": 0.0,
            "drift_flag": drift_flag,
            "fault_flag": fault_flag,
        }
    )

    df = utils.cast_schema(df, SIMULATION_SCHEMA)
    return df
