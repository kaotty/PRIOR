"""Activate opt-in TIR compatibility hooks from a portable PYTHONPATH."""
import os

if os.environ.get("TIR_AGENT_HORIZON_COMPAT") == "1":
    import tir_agent_horizon_compat

    tir_agent_horizon_compat.install()

if os.environ.get("TIR_MROPE_LAYOUT_COMPAT") == "1":
    import tir_mrope_compat

    tir_mrope_compat.install()
