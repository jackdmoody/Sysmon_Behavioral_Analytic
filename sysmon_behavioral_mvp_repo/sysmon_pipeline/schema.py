"""Schema normalization and validation.

Goal: take messy Sysmon-like input and produce a consistent dataframe with columns:
  - timestamp (datetime64[ns, UTC] naive acceptable but normalized to UTC-aware)
  - host (string)
  - event_id (int)
  - image (string, optional)
  - command_line (string, optional)
  - user (string, optional)

The MVP is tolerant: missing optional columns are created as empty strings.
"""

from __future__ import annotations

import pandas as pd
from typing import Iterable
from .config import PipelineConfig


REQUIRED_OUTPUT_COLS = ("timestamp", "host", "event_id", "image", "command_line", "user")


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_schema(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Normalize raw events into the MVP schema."""
    df = df.copy()

    # Timestamp
    if cfg.timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{cfg.timestamp_col}' not found. Available: {list(df.columns)[:50]}")
    ts = pd.to_datetime(df[cfg.timestamp_col], errors="coerce", utc=True)
    df["timestamp"] = ts

    # Host
    host_col = _first_present(df, cfg.host_cols)
    if host_col is None:
        raise ValueError(f"No host column found. Candidates: {cfg.host_cols}. Available: {list(df.columns)[:50]}")
    df["host"] = df[host_col].astype("string")

    # Event ID
    eid_col = _first_present(df, cfg.event_id_cols)
    if eid_col is None:
        raise ValueError(f"No event_id column found. Candidates: {cfg.event_id_cols}. Available: {list(df.columns)[:50]}")
    df["event_id"] = pd.to_numeric(df[eid_col], errors="coerce").astype("Int64")

    # Optional columns
    image_col = _first_present(df, cfg.image_cols)
    df["image"] = df[image_col].astype("string") if image_col else pd.Series(["" for _ in range(len(df))], dtype="string")

    cmd_col = _first_present(df, cfg.cmd_cols)
    df["command_line"] = df[cmd_col].astype("string") if cmd_col else pd.Series(["" for _ in range(len(df))], dtype="string")

    user_col = _first_present(df, cfg.user_cols)
    df["user"] = df[user_col].astype("string") if user_col else pd.Series(["" for _ in range(len(df))], dtype="string")

    # Drop rows with unusable timestamps or event IDs
    df = df.dropna(subset=["timestamp", "event_id"]).copy()
    df["event_id"] = df["event_id"].astype(int)

    # Filter window (optional)
    if cfg.time_min:
        tmin = pd.to_datetime(cfg.time_min, utc=True)
        df = df[df["timestamp"] >= tmin]
    if cfg.time_max:
        tmax = pd.to_datetime(cfg.time_max, utc=True)
        df = df[df["timestamp"] < tmax]

    # Sort for determinism
    df = df.sort_values(["host", "timestamp"]).reset_index(drop=True)

    # Ensure required cols exist in final view
    out = df[list(REQUIRED_OUTPUT_COLS)].copy()
    return out


def validate_schema(df: pd.DataFrame) -> None:
    """Raise if required columns are missing or obviously wrong."""
    missing = [c for c in REQUIRED_OUTPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    if df["timestamp"].isna().any():
        raise ValueError("Found null timestamps after normalization.")

    if df["host"].isna().any():
        raise ValueError("Found null hosts after normalization.")

    if df["event_id"].isna().any():
        raise ValueError("Found null event IDs after normalization.")
