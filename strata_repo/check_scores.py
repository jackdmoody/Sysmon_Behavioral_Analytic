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

attack_hosts = set(labels[labels.is_compromised].host)

triage = art.triage.copy()
triage["is_attack"] = triage["host"].isin(attack_hosts)
triage = triage.sort_values("triage_rank")

print("=== TRIAGE TABLE (top 20) ===")
cols = ["triage_rank", "host", "score", "is_attack", "gate_pass", "S_seq", "S_freq", "S_ctx", "S_drift"]
cols = [c for c in cols if c in triage.columns]
print(triage[cols].head(20).to_string(index=False))

print()
print("=== ATTACK HOST SCORES ===")
attack_rows = triage[triage["is_attack"]].sort_values("triage_rank")
print(attack_rows[cols].to_string(index=False))

print()
print("=== SCORE STATS ===")
print(f"Attack hosts   — mean rank: {attack_rows['triage_rank'].mean():.1f},  mean score: {attack_rows['score'].mean():.3f}")
benign_rows = triage[~triage["is_attack"]]
print(f"Benign hosts   — mean rank: {benign_rows['triage_rank'].mean():.1f},  mean score: {benign_rows['score'].mean():.3f}")

print()
print("=== CHANNEL SCORE DISTRIBUTIONS ===")
for col, source in [("S_seq", art.seq_scores), ("S_freq", art.freq_scores),
                    ("S_ctx", art.ctx_scores), ("S_drift", art.drift_scores)]:
    if source is not None and col in source.columns:
        merged = source.merge(labels[["host","is_compromised"]], on="host", how="left")
        atk = merged[merged.is_compromised][col]
        ben = merged[~merged.is_compromised][col]
        print(f"{col}:  attack mean={atk.mean():.4f}  benign mean={ben.mean():.4f}  "
              f"attack max={atk.max():.4f}  benign max={ben.max():.4f}")
