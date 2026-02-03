from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd

@dataclass(frozen=True)
class SchemaSpec:
    required: Tuple[str, ...]
    optional: Tuple[str, ...] = ()

def validate_schema(df: pd.DataFrame, spec: SchemaSpec) -> None:
    missing = [c for c in spec.required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nAvailable: "
            + ", ".join(map(str, df.columns))
        )

def coerce_types(df: pd.DataFrame, *, ts_col: str, host_col: str, event_id_col: str) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    out[host_col] = out[host_col].astype(str)
    out[event_id_col] = pd.to_numeric(out[event_id_col], errors="coerce").astype("Int64")
    return out

def split_by_host(df: pd.DataFrame, host_col: str) -> Dict[str, pd.DataFrame]:
    # Canonical implementation
    return {str(h): g for h, g in df.groupby(host_col, dropna=False)}
