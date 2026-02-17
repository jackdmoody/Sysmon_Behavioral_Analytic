"""Central configuration for the Sysmon Behavioral Analytic MVP.

This MVP is intentionally small:
  - Load Sysmon-like events from CSV/JSONL
  - Normalize into a standard schema
  - Build host-level features (counts & simple rates)
  - Score hosts using either z-score (default) or IsolationForest
  - Write triage outputs + simple plots
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    # --- I/O ---
    input_path: Path = Path("data/sample_sysmon.csv")
    output_dir: Path = Path("output")

    # Input format: inferred by suffix. Supported: .csv, .jsonl
    # If your timestamp column differs, set timestamp_col accordingly.
    timestamp_col: str = "UtcTime"

    # Host column candidates (first found will be used).
    host_cols: tuple[str, ...] = ("Computer", "Host", "Hostname", "host")

    # Event ID column candidates.
    event_id_cols: tuple[str, ...] = ("EventID", "EventId", "event_id", "EventID_")

    # Process image column candidates.
    image_cols: tuple[str, ...] = ("Image", "ProcessImage", "process_image", "ProcessName")

    # Command line column candidates.
    cmd_cols: tuple[str, ...] = ("CommandLine", "CmdLine", "command_line")

    # User column candidates.
    user_cols: tuple[str, ...] = ("User", "UserName", "SubjectUserName", "user")

    # Optional filtering
    # Provide ISO-like strings (e.g., "2026-02-01T00:00:00") to filter window.
    time_min: Optional[str] = None
    time_max: Optional[str] = None

    # --- Features ---
    top_event_ids: int = 20     # include top-N event IDs as features
    top_images: int = 20        # include top-N process images as features

    # --- Scoring ---
    scoring_method: str = "zscore"  # "zscore" or "iforest"
    random_seed: int = 42

    # Isolation Forest params (only used if scoring_method == "iforest")
    iforest_contamination: float = 0.02

    # --- Plotting ---
    make_plots: bool = True
