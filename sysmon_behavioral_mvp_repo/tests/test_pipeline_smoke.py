import pandas as pd
from pathlib import Path
from sysmon_pipeline.config import PipelineConfig
from sysmon_pipeline.pipeline import run


def test_pipeline_smoke(tmp_path: Path):
    # Minimal dataset
    df = pd.DataFrame({
        "UtcTime": ["2026-02-16T12:00:00Z", "2026-02-16T12:01:00Z", "2026-02-16T12:02:00Z"],
        "Computer": ["HOST1", "HOST1", "HOST2"],
        "EventID": [1, 4688, 1],
        "Image": ["a.exe", "b.exe", "a.exe"],
    })

    data_path = tmp_path / "sample.csv"
    df.to_csv(data_path, index=False)

    cfg = PipelineConfig(input_path=data_path, output_dir=tmp_path / "out", make_plots=False)
    artifacts = run(cfg)

    assert artifacts["triage"].exists()
    triage = pd.read_csv(artifacts["triage"])
    assert "anomaly_score" in triage.columns
