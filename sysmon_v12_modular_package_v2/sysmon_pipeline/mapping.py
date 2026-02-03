from __future__ import annotations
from typing import Dict, Optional
import pandas as pd

def score_to_label(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"

def grade_event(event_id: int) -> float:
    # Replace with your notebook's full mapping as needed.
    defaults = {
        1: 0.2,   # Process Create
        3: 0.5,   # Network Connect
        11: 0.7,  # File Create
        13: 0.8,  # Registry Value Set
        22: 0.85, # DNS Query
    }
    try:
        return float(defaults.get(int(event_id), 0.1))
    except Exception:
        return 0.1

def grade_events(
    df: pd.DataFrame,
    *,
    event_id_col: str,
    out_score_col: str="severity_score",
    out_label_col: str="severity_label",
) -> pd.DataFrame:
    out = df.copy()
    out[out_score_col] = out[event_id_col].apply(lambda x: grade_event(x) if pd.notna(x) else 0.0)
    out[out_label_col] = out[out_score_col].apply(score_to_label)
    return out
