from run_experiments import make_synthetic
from sysmon_pipeline.loaders import split_time_windows
from sysmon_pipeline.schema import normalize_schema
from sysmon_pipeline import StrataConfig

df, labels = make_synthetic(n_hosts=60, n_attack_hosts=12, n_roles=3, seed=42)
cfg = StrataConfig()
events = normalize_schema(df, cfg)
baseline, score = split_time_windows(events, baseline_days=1, score_days=1)

attack_hosts = set(labels[labels.is_compromised].host)
attack_in_score = score[score["host"].isin(attack_hosts)]
benign_in_score = score[~score["host"].isin(attack_hosts)]

print(f"Total events in scoring window:        {len(score)}")
print(f"Attack events in scoring window:       {len(attack_in_score)}")
print(f"Unique attack hosts in scoring window: {attack_in_score['host'].nunique()} / {len(attack_hosts)}")
print(f"Benign events in scoring window:       {len(benign_in_score)}")
print()
print("Attack host event counts in scoring window:")
print(attack_in_score.groupby("host").size().to_string())
