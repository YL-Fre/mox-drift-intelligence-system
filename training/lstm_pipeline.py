# training/lstm_pipeline.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.datasets import TimeSeriesWindowDataset
from mox_drift_sim.config import TrainingConfig


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out


def train_lstm(series: np.ndarray, cfg: TrainingConfig):
    dataset = TimeSeriesWindowDataset(series, window_size=cfg.window_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = SimpleLSTM()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(cfg.num_epochs):
        losses = []
        for x, y in loader:
            x = x.unsqueeze(-1)   # shape: (B, T, 1)
            pred = model(x)
            loss = loss_fn(pred.squeeze(), y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{cfg.num_epochs} — Loss: {np.mean(losses):.4f}")

    return model
