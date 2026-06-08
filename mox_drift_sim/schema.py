# mox_drift_sim/schema.py
"""
Compatibility shim so that imports like:
    from ..schema import SIMULATION_SCHEMA
work even though the actual definition lives in data_schema.py
"""

from .data_schema import SIMULATION_SCHEMA

__all__ = ["SIMULATION_SCHEMA"]
