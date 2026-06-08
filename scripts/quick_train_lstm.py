# scripts/quick_train_lstm.py
import numpy as np
import pandas as pd
import torch

from mox_drift_sim.config import TrainingConfig
from training.lstm_pipeline import train_lstm


def main():
    # 1. 读取数据
    df = pd.read_csv("data/processed/mox_drift_v1.csv")

    # 2. 取出传感器数值序列
    series = df["sensor_value"].values.astype(np.float32)

    # 3. 清洗 NaN / inf，避免训练时出现 NaN loss
    series = np.where(np.isfinite(series), series, np.nan)  # 把 inf/-inf 替换成 NaN
    series = pd.Series(series).interpolate().bfill().ffill().values  # 插值 + 前后填充

    # 4. 简单归一化（零均值单位方差），防止数值过大/过小
    mean = series.mean()
    std = series.std() if series.std() > 0 else 1.0
    series = (series - mean) / std

    # 5. 配置训练参数
    cfg = TrainingConfig(
        window_size=30,
        batch_size=16,
        num_epochs=3,
        lr=1e-3,
    )

    # 6. 为了可复现，设置随机种子（可选）
    torch.manual_seed(42)
    np.random.seed(42)

    # 7. 训练 LSTM
    train_lstm(series, cfg)


if __name__ == "__main__":
    main()
