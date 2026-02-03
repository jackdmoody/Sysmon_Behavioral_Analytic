from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

@dataclass(frozen=True)
class SysmonConfig:
    # Core behavior-pair settings
    window_seconds: int = 60
    min_events_per_host: int = 25

    # Baseline selection
    baseline_host_allowlist: Optional[Sequence[str]] = None  # if None, infer from data

    # Isolation Forest settings
    iforest_contamination: float = 0.02
    random_state: int = 42

    # Column names (make ingestion portable across schemas)
    col_timestamp: str = "_timestamp"
    col_host: str = "host.fqdn"
    col_event_id: str = "winlog.event_id"
    col_severity_label: str = "severity_label"

    # Optional: drop noisy event IDs
    drop_event_ids: Sequence[int] = field(default_factory=tuple)

    # Export / caching
    cache_dir: Optional[str] = None  # e.g. "./.cache_sysmon"
    debug: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
