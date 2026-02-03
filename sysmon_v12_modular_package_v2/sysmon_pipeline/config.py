from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

@dataclass(frozen=True)
class SysmonConfig:
    # --- Required base columns ---
    col_timestamp: str = "_timestamp"
    col_host: str = "host.fqdn"
    col_event_id: str = "winlog.event_id"

    # --- Enrichment columns ---
    col_severity_score: str = "severity_score"
    col_severity_label: str = "severity_label"
    col_pair: str = "pair"

    # --- Pair correlation ---
    window_seconds: int = 60
    critical_labels: Sequence[str] = ("critical", "high")

    # --- Baseline selection ---
    baseline_host_allowlist: Optional[Sequence[str]] = None
    baseline_top_n_hosts: int = 20

    # --- Isolation Forest ---
    iforest_contamination: float = 0.02
    random_state: int = 42

    # --- Filtering / hygiene ---
    drop_event_ids: Sequence[int] = field(default_factory=tuple)
    min_events_per_host: int = 25

    # --- Debug / caching ---
    debug: bool = False
    cache_dir: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
