import os
import pandas as pd
from mox_drift_sim.config import LifecycleConfig
from mox_drift_sim.generator import simulate_lifecycle


if __name__ == "__main__":
    cfg = LifecycleConfig()
    df = simulate_lifecycle(cfg)

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/mox_drift_v1.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved synthetic dataset to {out_path} ({len(df)} rows)")

