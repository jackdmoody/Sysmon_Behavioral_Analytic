from run_experiments import make_synthetic
from sysmon_pipeline.loaders import split_time_windows
from sysmon_pipeline.schema import normalize_schema
from sysmon_pipeline import StrataConfig, AblationConfig, StrataPipeline

df, labels = make_synthetic(n_hosts=60, n_attack_hosts=12, n_roles=3, seed=42)
cfg = StrataConfig(ablation=AblationConfig.full_pipeline())
events = normalize_schema(df, cfg)
baseline, score = split_time_windows(events, baseline_days=1, score_days=1)

pipe = StrataPipeline(cfg)
fitted = pipe.fit(baseline)
art = pipe.score(score, fitted, prior_window_df=baseline)

print("=== TRIAGE hosts (first 5) ===")
print(art.triage["host"].head().tolist())

print("\n=== LABELS hosts (first 5) ===")
print(labels["host"].head().tolist())

print("\n=== ATTACK hosts in labels ===")
print(labels[labels.is_compromised]["host"].tolist())

print("\n=== OVERLAP (what run_h4 sees) ===")
triage_hosts = set(art.triage["host"])
label_attack  = set(labels[labels.is_compromised]["host"])
print(f"Triage hosts:        {sorted(list(triage_hosts))[:5]} ...")
print(f"Label attack hosts:  {sorted(list(label_attack))}")
print(f"Intersection:        {triage_hosts & label_attack}")
print(f"Overlap count:       {len(triage_hosts & label_attack)} / {len(label_attack)}")
