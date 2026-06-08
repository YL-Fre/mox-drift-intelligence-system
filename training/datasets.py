# training/datasets.py
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesWindowDataset(Dataset):
    """
    用于 LSTM 时间序列预测的窗口数据集。
    输入：window_size 长度的序列
    输出：下一个时间点的值（单步预测）
    """

    def __init__(self, series: np.ndarray, window_size: int = 30):
        self.series = series.astype(np.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.window_size]
        y = self.series[idx + self.window_size]
        return torch.tensor(x), torch.tensor(y)
