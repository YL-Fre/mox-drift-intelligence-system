from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from ..schema import SIMULATION_SCHEMA


def random_walk_around(
    rng: np.random.Generator,
    n: int,
    mean: float,
    step_std: float,
) -> np.ndarray:
    """Random walk fluctuating around a mean value."""
    steps = rng.normal(0.0, step_std, size=n)
    walk = np.cumsum(steps)
    return mean + walk


def gas_response(
    gas_ppm: float | np.ndarray,
    temp: float | np.ndarray,
    gas_base_level: float,
    gas_exponent: float,
    temp_mean: float,
) -> np.ndarray:
    """Simple MOx-like gas response with temperature dependence."""
    gas_ppm = np.asarray(gas_ppm, dtype=float)
    temp = np.asarray(temp, dtype=float)
    response_f = gas_base_level * np.power(np.maximum(gas_ppm, 0.0), gas_exponent)
    temp_factor = 1.0 + 0.01 * (temp - temp_mean)
    return response_f * temp_factor


def cast_schema(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """Cast DataFrame columns to the specified schema if present."""
    for col, dtype in schema.items():
        if col not in df.columns:
            continue
        if dtype == "datetime64[ns]":
            df[col] = pd.to_datetime(df[col])
        elif dtype == "category":
            df[col] = df[col].astype("category")
        else:
            df[col] = df[col].astype(dtype)
    return df


def generate_gas_profile(
    rng: np.random.Generator,
    n: int,
    modes: pd.Series,
    base_level: float = 5.0,
) -> np.ndarray:
    """Generate a simple gas profile with pulses during NORM/STRESS modes."""
    gas = np.zeros(n, dtype=float)
    mask = modes.isin(["NORM", "STRESS"]).to_numpy()
    gas[mask] = base_level
    return gas
