from __future__ import annotations
import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from ..config import LifecycleConfig
from ..schema import SIMULATION_SCHEMA
from . import utils


def time_index_warmup(cfg: LifecycleConfig) -> DatetimeIndex:
    periods = cfg.warmup_hours * 60  # 1 min freq
    return pd.date_range(start=cfg.start_time, periods=periods, freq=cfg.freq)


def simulate_warmup_segment(cfg: LifecycleConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    idx = time_index_warmup(cfg)
    n = len(idx)

    minutes = (idx - idx[0]).total_seconds() / 60.0

    temp = (
        cfg.temp_mean
        + cfg.temp_amplitude * np.sin(2 * np.pi * minutes / (24 * 60))
        + rng.normal(0.0, cfg.temp_noise_std, size=n)
    )

    rh = utils.random_walk_around(rng, n, cfg.rh_mean, cfg.rh_rw_std)

    gas_1 = np.zeros(n, dtype=float)  # warm-up: assume no target gas

    baseline = np.zeros(n)
    baseline[0] = cfg.baseline_start

    sensor_dyn = 0.0
    sensor_values = np.zeros(n)

    drift_flag = np.zeros(n, dtype=int)
    fault_flag = np.zeros(n, dtype=int)

    for t in range(1, n):
        drift = cfg.drift_rate_warmup + rng.normal(0.0, cfg.baseline_rw_std_warmup)
        humidity_effect = cfg.k_h * (rh[t] - cfg.humidity_ref)
        baseline[t] = baseline[t - 1] + drift + humidity_effect

        sensor_eq = utils.gas_response(
            gas_ppm=gas_1[t],
            temp=temp[t],
            gas_base_level=cfg.gas_base_level,
            gas_exponent=cfg.gas_exponent,
            temp_mean=cfg.temp_mean,
        )

        sensor_dyn = sensor_dyn + (sensor_eq - sensor_dyn) / cfg.tau_warmup

        noise = rng.normal(0.0, cfg.noise_std_warmup)
        sensor_values[t] = baseline[t] + sensor_dyn + noise

    sensor_values[0] = baseline[0] + rng.normal(0.0, cfg.noise_std_warmup)

    df = pd.DataFrame(
        {
            "sensor_id": 1,
            "phase": "WARMUP",
            "mode": "WARMUP",
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
