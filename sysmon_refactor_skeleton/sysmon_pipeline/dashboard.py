from __future__ import annotations
import pandas as pd

def launch_dashboard(*, triage: pd.DataFrame) -> None:
    # Minimal placeholder: keep dashboard separate so pipeline is testable.
    # In your notebook, port the ipywidgets + plotly dashboard here.
    display(triage.head(25))
