"""I/O helpers for the MVP."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from .config import PipelineConfig


def load_events(cfg: PipelineConfig) -> pd.DataFrame:
    """Load Sysmon-like events from CSV or JSONL into a pandas DataFrame."""
    path = Path(cfg.input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}.\n"
            "Put your data at data/sample_sysmon.csv (default) or set input_path in config."
        )

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported input type: {path.suffix}. Use .csv or .jsonl")

    return df


def ensure_output_dir(cfg: PipelineConfig) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out
