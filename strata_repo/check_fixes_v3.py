"""
Quick diagnostic script to verify the three bug fixes before running experiments.

Run with:
    python check_fixes.py

Each check prints PASS or FAIL with a brief explanation.
No pytest required — just run it directly.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
INFO = "\033[94m  INFO\033[0m"

def header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ---------------------------------------------------------------------------
# Setup: build a small synthetic dataset we can reuse across checks
# ---------------------------------------------------------------------------

def make_data():
    """Minimal synthetic data: 3 clearly distinct roles, attacks in one role."""
    from run_experiments import make_synthetic
    df, labels = make_synthetic(
        n_hosts=30,
        n_events_per_host=200,
        n_attack_hosts=6,
        n_roles=3,
        seed=42,
    )
    return df, labels

# ---------------------------------------------------------------------------
# CHECK 1: H2 — KMeans actually produces multiple distinct roles
# ---------------------------------------------------------------------------

def check_h2_role_separation(df, labels):
    header("CHECK 1: H2 — Role inference produces distinct clusters")

    from sysmon_pipeline.schema import normalize_schema
    from sysmon_pipeline.mapping import build_tokens
    from sysmon_pipeline.pairs import compute_rate_features, build_role_features, infer_roles
    from sysmon_pipeline import StrataConfig

    cfg = StrataConfig()
    events = normalize_schema(df, cfg)
    events = build_tokens(events)
    rates  = compute_rate_features(events, cfg)
    role_feats = build_role_features(rates, cfg)
    roles  = infer_roles(role_feats, cfg)

    counts = roles["role_id"].value_counts()
    n_distinct = len(counts)

    print(f"  Role distribution:\n{counts.to_string()}")

    if n_distinct >= 2:
        print(f"{PASS}  {n_distinct} distinct roles found")
    else:
        print(f"{FAIL}  Only {n_distinct} role — KMeans not separating hosts")
        print("       Fix: make role profiles in make_synthetic() more distinct")
        return False

    # Warn if KMeans over-fragmented (too many small roles)
    if n_distinct > 4:
        print(f"{FAIL}  {n_distinct} roles from {len(roles)} hosts — KMeans over-fragmenting")
        print("       Fix: increase min_hosts_per_role in config, or reduce n_clusters cap in infer_roles()")
        return False

    minority_role = counts.idxmin()
    minority_pct  = counts.min() / counts.sum()
    min_hosts     = counts.min()
    if min_hosts < 3:
        print(f"{FAIL}  Smallest role '{minority_role}' has only {min_hosts} host(s) — "
              "too few to build a meaningful peer baseline")
        print("       Fix: increase n_hosts in make_synthetic(), or cap n_clusters in infer_roles()")
        return False

    print(f"{PASS}  Smallest role has {min_hosts} hosts ({minority_pct:.1%}) — enough for peer baseline")
    return True


# ---------------------------------------------------------------------------
# CHECK 2: H4 — All four channels have real variance
# ---------------------------------------------------------------------------

def check_h4_channel_variance(df, labels):
    header("CHECK 2: H4 — All channels have non-zero score variance")

    from sysmon_pipeline import StrataConfig, AblationConfig, StrataPipeline
    from sysmon_pipeline.loaders import split_time_windows
    from sysmon_pipeline.schema import normalize_schema

    cfg  = StrataConfig(ablation=AblationConfig.full_pipeline())

    # Normalize first so ts column exists, then split
    # Use short windows to fit synthetic data (~2 day span)
    events = normalize_schema(df, cfg)
    baseline_df, score_df = split_time_windows(events, baseline_days=1, score_days=1)
    pipe = StrataPipeline(cfg)
    fitted = pipe.fit(baseline_df)
    # Pass baseline_df as prior_window so drift channel has something to compare against
    art    = pipe.score(score_df, fitted, prior_window_df=baseline_df)

    triage = art.triage

    channel_map = {
        "S_seq":   art.seq_scores,
        "S_freq":  art.freq_scores,
        "S_ctx":   art.ctx_scores,
        "S_drift": art.drift_scores,
        "score":   triage,
    }

    all_ok = True
    for col, source_df in channel_map.items():
        if source_df is None or col not in source_df.columns:
            print(f"{FAIL}  {col}: MISSING — channel not producing output")
            all_ok = False
            continue

        s = source_df[col].dropna()
        std = s.std()
        nonzero = (s != 0).sum()
        print(f"  {col:12s}  std={std:.4f}  nonzero={nonzero}/{len(s)}")

        if std < 1e-6 or nonzero == 0:
            print(f"{FAIL}  {col} has no variance — channel is dead")
            all_ok = False
        else:
            print(f"{PASS}  {col} has real variance")

    # Check fused != any single channel (Borda is actually combining)
    if art.seq_scores is not None and "S_seq" in art.seq_scores.columns and triage is not None:
        seq_col = "S_seq_seq" if "S_seq_seq" in triage.columns else "S_seq"
        merged  = triage.merge(art.seq_scores[["host","S_seq"]], on="host", how="left")
        # After merge, S_seq may have been renamed due to suffix collision
        seq_col = [c for c in merged.columns if "S_seq" in c and c != "score"]
        if seq_col:
            corr = merged["score"].corr(merged[seq_col[0]])
            if corr > 0.999:
                print(f"{FAIL}  fused score is perfectly correlated with S_seq (r={corr:.4f}) — fusion not working")
                all_ok = False
            else:
                print(f"{PASS}  fused score differs from S_seq (r={corr:.3f})")

    return all_ok


# ---------------------------------------------------------------------------
# CHECK 3: H3 — Bootstrap null gives p-values near uniform (mean ~0.5)
# ---------------------------------------------------------------------------

def check_h3_pvalue_calibration(df, labels):
    header("CHECK 3: H3 — Bootstrap null distribution is correctly calibrated")

    from sysmon_pipeline import StrataConfig, AblationConfig, StrataPipeline
    from sysmon_pipeline.divergence import calibrate_jsd_null_distribution
    from sysmon_pipeline.loaders import split_time_windows
    from scipy.stats import kstest

    # Sub-check A: null distribution mean should be > 0 (bug was n_state always tiny)
    cfg = StrataConfig(ablation=AblationConfig.full_pipeline())
    cfg.baseline.bootstrap_samples = 200  # fast for diagnostic

    from sysmon_pipeline.schema import normalize_schema
    from sysmon_pipeline.mapping import build_tokens
    from sysmon_pipeline.pairs import compute_rate_features, build_role_features, infer_roles
    from sysmon_pipeline.sequence import assign_sessions, bucket_deltas, build_transition_counts
    from sysmon_pipeline.divergence import fit_peer_baselines

    events = normalize_schema(df, cfg)
    events = build_tokens(events)
    rates  = compute_rate_features(events, cfg)
    role_feats = build_role_features(rates, cfg)
    roles  = infer_roles(role_feats, cfg)
    events["role_id"] = events["host"].map(roles.set_index("host")["role_id"])

    events = assign_sessions(events, cfg, {"default": float(cfg.time.session_gap_seconds)})
    events = bucket_deltas(events, cfg)
    token_col = f"token_{cfg.token_resolution}"
    trans = build_transition_counts(events, cfg, level=token_col)
    baselines = fit_peer_baselines(trans, roles, cfg)

    if not baselines:
        print(f"{FAIL}  No baselines built — check role inference and transition counting")
        return False

    role_key  = next(iter(baselines))
    baseline  = baselines[role_key]

    rng = np.random.default_rng(0)
    null_jsds = []
    for n_h in [50, 100, 200]:
        mu, sigma = calibrate_jsd_null_distribution(baseline, n_h, cfg, rng=rng)
        null_jsds.append(mu)
        print(f"  n_h={n_h:4d}  null_mean={mu:.5f}  null_std={sigma:.5f}")
        if mu < 1e-6:
            print(f"{FAIL}  null mean is ~0 for n_h={n_h} — n_state bug still present")
            return False

    print(f"{PASS}  null distribution mean > 0 for all n_h values")

    # Sub-check B: run H3 on benign-only data and check p-value mean
    clean_hosts = set(labels.loc[~labels["is_compromised"], "host"])
    benign_df   = df[df["host"].isin(clean_hosts)]
    benign_norm = normalize_schema(benign_df, cfg)

    pipe   = StrataPipeline(cfg)
    fitted = pipe.fit(benign_norm)
    art    = pipe.score(benign_norm, fitted)

    if art.seq_scores is None or "S_seq_pvalue" not in art.seq_scores.columns:
        print(f"{FAIL}  S_seq_pvalue column missing — calibration not running")
        return False

    pvals = art.seq_scores["S_seq_pvalue"].dropna().to_numpy()
    mean_pval = pvals.mean()
    ks_stat, ks_p = kstest(pvals, "uniform")

    print(f"  p-value mean={mean_pval:.3f}  (ideal ~0.5)")
    print(f"  KS stat={ks_stat:.4f}  KS p={ks_p:.4f}")

    if abs(mean_pval - 0.5) < 0.15:
        print(f"{PASS}  p-value mean is close to 0.5")
    else:
        print(f"{FAIL}  p-value mean={mean_pval:.3f} is far from 0.5 — null still miscalibrated")
        return False

    if ks_p > 0.05:
        print(f"{PASS}  KS test passes — p-values are approximately uniform")
    else:
        print(f"{INFO}  KS test p={ks_p:.4f} (still failing, but mean is better)")
        print("       May need more hosts or events for KS to pass — check mean first")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nBuilding synthetic data...")
    try:
        df, labels = make_data()
        print(f"  {len(df)} events, {df['host'].nunique()} hosts, "
              f"{labels['is_compromised'].sum()} attack hosts")
    except Exception as e:
        print(f"{FAIL}  Could not build synthetic data: {e}")
        sys.exit(1)

    results = {}
    for name, fn in [
        ("H2 role separation", check_h2_role_separation),
        ("H4 channel variance", check_h4_channel_variance),
        ("H3 p-value calibration", check_h3_pvalue_calibration),
    ]:
        try:
            results[name] = fn(df, labels)
        except Exception as e:
            print(f"{FAIL}  {name} crashed: {e}")
            import traceback; traceback.print_exc()
            results[name] = False

    header("SUMMARY")
    all_passed = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"{status}  {name}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("  All checks passed — safe to run experiments.\n")
    else:
        print("  Some checks failed — review output above before running experiments.\n")
        sys.exit(1)
