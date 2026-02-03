## Module & Function Breakdown

This section documents what each module contains and what each function is responsible for. Use it as a quick “where do I add/change things?” reference.

---

### `sysmon_pipeline/config.py`
**Purpose:** Central configuration (column names + knobs for each analytic stage)

- **`SysmonConfig`**  
  Dataclass holding: required column names, correlation window, baseline selection settings, Isolation Forest settings, and pipeline hygiene options (drops, minimum events, etc.).

---

### `sysmon_pipeline/schema.py`
**Purpose:** Input validation and normalization (fail fast, consistent types)

- **`SchemaSpec`**  
  Simple container for required/optional columns.

- **`validate_schema(df, spec)`**  
  Ensures required columns exist; raises a clear error message if not.

- **`coerce_types(df, ts_col, host_col, event_id_col)`**  
  Converts timestamp to UTC datetime, host to string, event_id to nullable integer.

- **`split_by_host(df, host_col)`**  
  Returns `{host: dataframe}` partitions for host-centric processing.

---

### `sysmon_pipeline/mapping.py`
**Purpose:** Severity enrichment (turn event IDs into analyst-meaningful severity labels)

- **`score_to_label(score)`**  
  Maps a numeric score into a severity label (e.g., low/medium/high/critical).

- **`grade_event(event_id)`**  
  Returns a severity score for an event ID (placeholder mapping by default; replace/extend with your full v12 table).

- **`grade_events(df, event_id_col, out_score_col="severity_score", out_label_col="severity_label")`**  
  Adds `severity_score` and `severity_label` columns to the dataframe.

---

### `sysmon_pipeline/pairs.py`
**Purpose:** Behavioral correlation (time-windowed event-pairs) + host summary stats + anomaly scoring

- **`correlate_critical_events_single_host(host_df, ts_col, event_id_col, severity_col, window_seconds, critical_labels)`**  
  For one host: builds directed event pairs occurring within a time window among “critical/high” (or configured) events.

- **`correlate_critical_events_by_host(df, host_col, ts_col, event_id_col, severity_col, window_seconds, critical_labels)`**  
  Applies the single-host correlation across all hosts and returns a unified pairs table.

- **`compute_pair_stats(pairs_df, host_col, count_col="count")`**  
  Aggregates per-host statistics over pair counts (sum/mean/std/max/n_pairs).

- **`run_isolation_forest_on_hosts(stats_df, feature_cols, contamination, random_state)`**  
  Fits Isolation Forest on host summary stats and adds `iforest_score` and `iforest_label`.

---

### `sysmon_pipeline/sequence.py`
**Purpose:** Markov chain construction (event sequence → transition matrices)

- **`ensure_sorted_events(df, ts_col, host_col)`**  
  Sorts events by host then timestamp (required for sequence modeling).

- **`build_state_map(df, event_id_col)`**  
  Creates a stable mapping `{event_id: state_index}` for Markov matrix indexing.

- **`build_transition_counts(df, host_col, ts_col, event_id_col, state_map)`**  
  Builds per-host transition count matrices (counts of event_i → event_j).

- **`normalize_rows(mat, eps=1e-12)`**  
  Row-normalizes a matrix into a valid transition probability matrix.

- **`build_host_markov_matrix(transition_counts)`**  
  Converts a host’s transition counts into a Markov transition probability matrix.

- **`compute_baseline_markov_matrix(host_markov, baseline_hosts)`**  
  Produces a baseline Markov matrix by averaging baseline hosts’ matrices and normalizing.

---

### `sysmon_pipeline/divergence.py`
**Purpose:** Drift scoring (distance between host behavior and baseline)

- **`kl_divergence_matrix(P, Q, eps=1e-12)`**  
  KL divergence between two transition matrices.

- **`js_divergence_matrix(P, Q, eps=1e-12)`**  
  Jensen–Shannon divergence between two transition matrices (symmetric, bounded).

---

### `sysmon_pipeline/scoring.py`
**Purpose:** Host-level Markov scoring + evidence fusion into a ranked triage table

- **`compute_host_markov_scores(host_markov, baseline_markov)`**  
  Computes per-host `markov_kl` and `markov_js` against the baseline.

- **`build_ranked_triage(pair_stats, markov_scores, host_col)`**  
  Merges pair-based stats with Markov drift scores and computes a simple `combined_score` for ranking.

---

### `sysmon_pipeline/pipeline.py`
**Purpose:** Orchestrates stages; supports partial runs for “Hunter vs Research mode”

- **`SysmonArtifacts`**  
  Container for outputs (events, pairs, pair_stats, markov matrices, scores, triage).

- **`SysmonPipeline.fit(df)`**  
  Learns baseline assets:
  - chooses baseline hosts (allowlist or top-N)
  - builds `state_map`
  - computes `baseline_markov`

- **`SysmonPipeline.stage_enrich(df)`**  
  Validates schema, coerces types, applies severity enrichment.

- **`SysmonPipeline.stage_pairs(events)`**  
  Computes event pairs by host within the configured time window.

- **`SysmonPipeline.stage_pair_stats(pairs)`**  
  Aggregates pair stats and runs Isolation Forest for anomaly scoring.

- **`SysmonPipeline.stage_markov(events)`**  
  Builds per-host Markov matrices and computes divergence scores vs baseline.

- **`SysmonPipeline.stage_triage(pair_stats, markov_scores)`**  
  Merges evidence layers and ranks hosts.

- **`SysmonPipeline.run(df, stages=("enrich","pairs","pair_stats","markov","triage"))`**  
  Runs a selected subset of stages and returns `SysmonArtifacts`.

  Valid stages:
  - `enrich`
  - `pairs`
  - `pair_stats`
  - `markov`
  - `triage`

---

### `tests/`
**Purpose:** Ensure the package is importable and runs end-to-end

- **`test_imports.py`**  
  Verifies package imports cleanly.

- **`test_end_to_end_synthetic.py`**  
  Generates synthetic Sysmon-like events, fits baseline, runs full pipeline, asserts triage output exists.
