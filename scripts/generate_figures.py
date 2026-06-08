"""
Generate demonstration figures for MOx drift simulation.

This script produces three industrial-style figures:

1) warmup_vs_field.png
   - N-type MOx sensor
   - Warm-up transient vs. field operation (stable regime)
   - Particle size & warm-up time constant included as model assumptions

2) humidity_coupling.png
   - Baseline drift due to humidity coupling
   - Shows sensor signal and relative humidity profile

3) stress_drift_example.png
   - Long-term stress / aging drift with occasional shocks
   - Typical for fielded MOx sensors

You can later replace the `simulate_*` functions with calls to your
`demo_signals.py` if you want a single source of synthetic data.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Global paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Style helpers
# ----------------------------------------------------------------------
def apply_mox_style():
    """Apply a clean, industrial plotting style."""
    plt.style.use("default")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.grid"] = False


def _to_hours(t_seconds: np.ndarray) -> np.ndarray:
    """Convert seconds to hours."""
    return np.asarray(t_seconds, dtype=float) / 3600.0


# ----------------------------------------------------------------------
# Synthetic signal generators
# 你之后可以把这些替换成 demo_signals.py 里的函数
# ----------------------------------------------------------------------
def simulate_warmup_and_field() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate warm-up + early field operation for an N-type MOx sensor.

    Returns
    -------
    t_warmup_s : seconds
    warmup : a.u.
    t_field_s : seconds
    field : a.u.
    """
    rng = np.random.default_rng(42)

    # Warm-up: 0–300 s
    t_warmup_s = np.linspace(0, 300, 200)
    tau = 90.0  # warm-up time constant (s), ~15–30 nm particles
    warmup = 0.2 + 0.8 * (1.0 - np.exp(-t_warmup_s / tau))
    warmup += rng.normal(scale=0.005, size=warmup.shape)

    # Field operation: next 5 hours, starting near the warm-up end value
    t_field_s = np.linspace(300, 300 + 5 * 3600, 500)
    baseline = warmup[-1]
    # small slow drift + noise
    drift = 0.01 * (1.0 - np.exp(-(t_field_s - t_field_s[0]) / (3 * 3600.0)))
    noise = rng.normal(scale=0.004, size=t_field_s.shape)
    field = baseline + drift + noise

    return t_warmup_s, warmup, t_field_s, field


def simulate_humidity_coupling():
    """
    Simulate humidity-induced baseline modulation.

    Returns
    -------
    t_s : seconds
    signal : a.u. (sensor response with humidity coupling)
    rh : %RH (relative humidity profile)
    """
    rng = np.random.default_rng(123)

    # 0–8 h, sample every 2 min
    t_s = np.linspace(0, 8 * 3600, 240)

    # Humidity: step-like pattern between 40% and 80%
    rh = 40 + 20 * (np.sign(np.sin(2 * np.pi * t_s / (2 * 3600))) + 1)
    # so we get approx 40 % ↔ 80 % steps every ~2 hours

    # Baseline: N-type sensor, humidity increases conductance a bit
    base = 1.0 + 0.02 * (t_s / (8 * 3600))  # slight long-term drift
    coupling = 0.08 * (rh - 40) / 40.0      # relative change from 40 % RH
    noise = rng.normal(scale=0.005, size=t_s.shape)

    signal = base + coupling + noise
    return t_s, signal, rh


def simulate_stress_drift_example():
    """
    Simulate long-term stress / aging drift over 30 days.

    Returns
    -------
    t_days : days
    signal : a.u.
    """
    rng = np.random.default_rng(456)

    # 0–30 days, sample every 0.1 day
    t_days = np.linspace(0, 30, 300)

    # Aging drift: logarithmic + small linear
    drift = 0.04 * np.log1p(t_days) + 0.005 * t_days

    # Occasional stress events (e.g. thermal shock, poisoning)
    signal = 1.0 + drift
    event_days = [5, 12, 20, 27]
    event_magnitudes = [0.015, -0.02, 0.01, -0.015]

    for d, mag in zip(event_days, event_magnitudes):
        signal += mag * np.exp(-0.5 * ((t_days - d) / 0.4) ** 2)

    # Noise
    signal += rng.normal(scale=0.006, size=t_days.shape)

    return t_days, signal


# ----------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------
def plot_warmup_vs_field(t_warmup_s, warmup, t_field_s, field, save_path: Path):
    """
    Plot MOx warm-up vs field operation with industrial-style visualization.

    Model assumptions:
    - Sensor type: N-type MOx (SnO2-like behaviour)
    - Typical particle size: ~15–30 nm
    - Warm-up time constant: tau ≈ 90 s
    - Time axis: hours since power-on
    """
    apply_mox_style()

    # convert time axis to hours since power-on
    t_warm_h = _to_hours(t_warmup_s)
    t_field_h = _to_hours(t_field_s)

    fig, ax = plt.subplots(figsize=(12, 4))

    # warm-up region shading
    ax.axvspan(t_warm_h[0], t_warm_h[-1], color="#D0E4F7", alpha=0.25)

    # boundary line between warm-up & field
    ax.axvline(t_warm_h[-1], color="gray", linestyle="--", linewidth=1)

    # signals
    ax.plot(t_warm_h, warmup,
            color="#0B69A8", linewidth=2.0, label="Warm-up (transient)")
    ax.plot(t_field_h, field,
            color="#D57A1F", linewidth=2.0, label="Field (day 1, stable)")

    # annotation
    ax.annotate(
        "Sensor stabilizes\n(field operation begins)",
        xy=(t_warm_h[-1], field[0]),
        xytext=(t_warm_h[-1] + 0.2 * (t_field_h[-1] - t_field_h[0]),
                field[0] - 0.06),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
    )

    # model assumptions text
    ax.text(0.01, 0.96, "Model: N-type MOx", transform=ax.transAxes,
            fontsize=9, color="gray", va="top")
    ax.text(0.01, 0.86, "Assumed particle size: 15–30 nm",
            transform=ax.transAxes, fontsize=8, color="gray", va="top")
    ax.text(0.01, 0.79, "Warm-up time constant: τ ≈ 90 s",
            transform=ax.transAxes, fontsize=8, color="gray", va="top")

    ax.set_title(
        "MOx Sensor Dynamics (N-type)\n"
        "Warm-up (transient) vs Field Operation (stable regime)"
    )
    ax.set_xlabel("Time since power-on (h)")
    ax.set_ylabel("Sensor value / conductance (a.u.)")

    ax.set_xlim(left=0.0)
    ax.grid(alpha=0.15)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_humidity_coupling(t_s, signal, rh, save_path: Path):
    """
    Plot humidity coupling: sensor signal vs. relative humidity.
    """
    apply_mox_style()

    t_h = _to_hours(t_s)

    fig, ax1 = plt.subplots(figsize=(12, 4))

    # sensor signal
    ax1.plot(t_h, signal, color="#0B69A8", linewidth=2.0,
             label="Sensor signal (with humidity coupling)")
    ax1.set_xlabel("Time in field (h)")
    ax1.set_ylabel("Sensor value (a.u.)")
    ax1.grid(alpha=0.15)

    # relative humidity on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(t_h, rh, color="#6C757D", linewidth=1.6,
             linestyle="--", label="Relative humidity")
    ax2.set_ylabel("Relative humidity (%RH)")

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title(
        "Humidity Coupling\n"
        "Baseline modulation of N-type MOx sensor by relative humidity"
    )

    # small note about coupling model
    ax1.text(0.01, 0.02,
             "Model: humidity increases conductance by up to ~8 % between 40–80 %RH",
             transform=ax1.transAxes, fontsize=8, color="gray", va="bottom")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_stress_drift_example(t_days, signal, save_path: Path):
    """
    Plot long-term stress / aging drift with occasional events.
    """
    apply_mox_style()

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t_days, signal, color="#0B69A8", linewidth=2.0,
            label="Sensor baseline (aging + stress)")

    ax.set_xlabel("Time in field (days)")
    ax.set_ylabel("Sensor value (a.u.)")
    ax.set_title(
        "Long-term Stress / Aging Drift\n"
        "Slow baseline increase with occasional shock events"
    )

    ax.grid(alpha=0.15)
    ax.legend(loc="upper left")

    # annotate that this is a conceptual model
    ax.text(0.01, 0.02,
            "Conceptual model: logarithmic aging drift + small linear trend\n"
            "with Gaussian-shaped shock events (e.g. thermal cycling, poisoning)",
            transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    print("[generate_figures] Project root:", PROJECT_ROOT)
    print("[generate_figures] Saving figures to:", FIG_DIR)

    # 1) Warm-up vs field
    t_warm_s, warmup, t_field_s, field = simulate_warmup_and_field()
    warmup_field_path = FIG_DIR / "warmup_vs_field.png"
    print("[generate_figures] Creating:", warmup_field_path)
    plot_warmup_vs_field(t_warm_s, warmup, t_field_s, field, warmup_field_path)

    # 2) Humidity coupling
    t_h_s, sig_h, rh = simulate_humidity_coupling()
    humidity_path = FIG_DIR / "humidity_coupling.png"
    print("[generate_figures] Creating:", humidity_path)
    plot_humidity_coupling(t_h_s, sig_h, rh, humidity_path)

    # 3) Stress drift example
    t_days, sig_stress = simulate_stress_drift_example()
    stress_path = FIG_DIR / "stress_drift_example.png"
    print("[generate_figures] Creating:", stress_path)
    plot_stress_drift_example(t_days, sig_stress, stress_path)

    print("[generate_figures] Done.")


if __name__ == "__main__":
    main()
