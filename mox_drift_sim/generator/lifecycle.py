from __future__ import annotations
import pandas as pd
from ..config import LifecycleConfig
from .warmup import simulate_warmup_segment
from .field import simulate_field_segment


def simulate_lifecycle(cfg: LifecycleConfig | None = None) -> pd.DataFrame:
    """Simulate full lifecycle: WARMUP → FIELD, returning v1.0 schema."""
    cfg = cfg or LifecycleConfig()

    df_warm = simulate_warmup_segment(cfg)
    warmup_end = df_warm["time"].iloc[-1]
    initial_baseline = df_warm["sensor_value"].iloc[-1]

    df_field = simulate_field_segment(cfg, warmup_end, initial_baseline)
    df = pd.concat([df_warm, df_field], ignore_index=True)

    cols = [
        "sensor_id",
        "phase",
        "mode",
        "time",
        "sensor_value",
        "temperature",
        "humidity",
        "gas_1_ppm",
        "gas_2_ppm",
        "gas_3_ppm",
        "drift_flag",
        "fault_flag",
    ]
    return df[cols]
