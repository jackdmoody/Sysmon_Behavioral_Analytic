import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from sysmon_pipeline import SysmonConfig, SysmonPipeline

def make_synth(n_hosts=5, n_events_per_host=200):
    rows = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    event_ids = [1, 3, 11, 13, 22]
    for h in range(n_hosts):
        host = f"host{h}.lab"
        t = base
        for i in range(n_events_per_host):
            t = t + timedelta(seconds=int(rng.integers(1, 10)))
            # Make one host weird
            if h == n_hosts - 1 and i % 10 == 0:
                eid = 22  # heavy DNS
            else:
                eid = int(rng.choice(event_ids))
            rows.append({"_timestamp": t, "host.fqdn": host, "winlog.event_id": eid})
    return pd.DataFrame(rows)

def test_pipeline_runs():
    df = make_synth()
    cfg = SysmonConfig(window_seconds=30, iforest_contamination=0.1, baseline_top_n_hosts=3)
    pipe = SysmonPipeline(cfg).fit(df)
    art = pipe.run(df, stages=("enrich","pairs","pair_stats","markov","triage"))
    assert art.triage is not None
    assert len(art.triage) > 0
    assert "combined_score" in art.triage.columns
