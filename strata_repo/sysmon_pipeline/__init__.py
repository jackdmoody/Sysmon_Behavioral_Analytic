"""
STRATA-E: Structural and Temporal Role-Aware Threat Analytics for Endpoint Telemetry.

Unified pipeline combining all three development iterations.
"""
from .config import StrataConfig, AblationConfig
from .pipeline import StrataPipeline, StrataArtifacts, FittedArtifacts
from .loaders import load_darpa_tc, load_sysmon_csv, split_time_windows

__all__ = [
    "StrataConfig",
    "AblationConfig",
    "StrataPipeline",
    "StrataArtifacts",
    "FittedArtifacts",
]
