import pandas as pd
from sysmon_pipeline.config import PipelineConfig
from sysmon_pipeline.schema import normalize_schema, validate_schema


def test_normalize_and_validate_schema():
    cfg = PipelineConfig()

    raw = pd.DataFrame({
        "UtcTime": ["2026-02-16T12:00:00Z", "2026-02-16T12:01:00Z"],
        "Computer": ["HOST1", "HOST1"],
        "EventID": [1, 4688],
        "Image": ["C:\\Windows\\System32\\cmd.exe", "C:\\Windows\\System32\\powershell.exe"],
        "CommandLine": ["cmd.exe /c whoami", "powershell -enc AAAA"],
        "User": ["DOMAIN\\user1", "DOMAIN\\user1"],
    })

    events = normalize_schema(raw, cfg)
    validate_schema(events)

    assert set(events.columns) == {"timestamp", "host", "event_id", "image", "command_line", "user"}
    assert len(events) == 2
    assert events["event_id"].dtype.kind in ("i", "u")
