# mox_drift_sim/config.py
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LifecycleConfig:
    """
    Configuration for lifecycle simulation parameters, used by warm-up and field phase generators.

    Notes:
    - A set of fields most likely to be used during warmup/field/lifecycle stages has been explicitly added
      (related to time, temperature, humidity, gas, noise, and drift).
    - A fallback mechanism is provided via __getattr__:
      If a generator attempts to access an undeclared 'cfg.xxx', it will return a reasonable default value
      instead of raising an AttributeError.
    """

    # ===== Randomness =====
    random_seed: int = 42

    # ===== Time Settings (for warmup.py / field.py) =====
    # Example: pd.date_range(start=cfg.start_time, periods=..., freq=cfg.freq)
    start_time: str = "2025-01-01 00:00:00"
    freq: str = "1min"          # 1 point per minute
    warmup_hours: int = 1       # Warm-up duration (hours)
    field_hours: int = 4        # Normal operation duration (hours)

    # Redundant fields kept for backward compatibility if minutes are still used elsewhere
    warmup_minutes: int = 60
    field_minutes: int = 240

    # ===== Sampling =====
    sampling_rate_hz: int = 1   # Corresponds to 'freq' if used in the code

    # ===== Ambient Temperature / Humidity Distribution =====
    # Typical usage in warmup.py / field.py:
    #   rng.normal(cfg.temp_mean, cfg.temp_std, size=...)
    temp_mean: float = 25.0     # °C
    temp_std: float = 1.0

    hum_mean: float = 50.0      # %RH
    hum_std: float = 5.0

    # ===== Gas Concentration (Single target gas: gas_1_ppm) =====
    gas_baseline_ppm: float = 0.0      # Background concentration
    gas_step_ppm: float = 100.0        # Step change amplitude (e.g., shock events)
    gas_noise_std: float = 1.0         # Superimposed noise (if any)

    # ===== Sensor Baseline & Noise =====
    baseline_resistance: float = 1.0   # Normalized value, can be treated as 1
    signal_amplitude: float = 0.1      # Gas response amplitude
    noise_std: float = 0.01            # Sensor reading noise

    # New field added here 👇
    tau_warmup: float = 30.0

    # ===== Drift Mechanism Parameters =====
    humidity_drift_strength: float = 0.02
    stress_drift_strength: float = 0.005
    aging_drift_strength: float = 0.0005

    # ===== Stress / Poisoning Events (if used in code) =====
    stress_event_prob: float = 0.0       # Probability of a stress event per step
    stress_step_change: float = 0.0      # Magnitude of step change caused by a stress event

    # ===== Fallback Default Table (provides uniform default values for unknown fields) =====
    _fallback_defaults: Dict[str, Any] = field(default_factory=lambda: {
        # If these fields are accessed in the generator but not explicitly defined above,
        # their defaults will be served from here.
        "temp_mean": 25.0,
        "temp_std": 1.0,
        "hum_mean": 50.0,
        "hum_std": 5.0,
        "gas_baseline": 0.0,
        "gas_baseline_ppm": 0.0,
        "gas_step": 100.0,
        "gas_step_ppm": 100.0,
        "baseline": 1.0,
        "baseline_resistance": 1.0,
        "noise_std": 0.01,
        "drift_slope": 0.0,
        "drift_offset": 0.0,
    })

    def __getattr__(self, name: str) -> Any:
        """
        Fallback mechanism when accessing an undeclared 'cfg.xxx':
        - Returns the preset value if it exists in _fallback_defaults.
        - Otherwise, returns 0.0 to prevent AttributeError from interrupting the simulation.

        This eliminates repetitive "whack-a-mole" AttributeErrors, allowing you
        to focus entirely on getting the pipeline running smoothly first.
        """
        if name in self._fallback_defaults:
            return self._fallback_defaults[name]
        # General fallback: return 0.0 for any unknown numerical configuration
        return 0.0


@dataclass
class TrainingConfig:
    """
    General training configuration for LSTM / Autoencoder models (for subsequent training scripts).
    """
    window_size: int = 30
    batch_size: int = 32
    num_epochs: int = 5
    lr: float = 1e-3