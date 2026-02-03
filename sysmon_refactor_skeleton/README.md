# Sysmon Refactor Skeleton

This is a streamlined package structure for your monolithic Sysmon notebook.

## What you get
- `sysmon_pipeline/` : a testable, modular pipeline with clear boundaries
- `notebooks/01_run_pipeline.ipynb` : a slim runner notebook
- `scripts/extract_from_notebook.py` : helper to extract and detect duplicate function defs

## Next steps
1) Put `Sysmon_v12_enhanced.ipynb` in this folder (or reference its path).
2) Run:
   ```bash
   python scripts/extract_from_notebook.py Sysmon_v12_enhanced.ipynb
   ```
3) Copy the *canonical* V12 implementations into the matching modules:
   - Markov + divergence -> `sequence.py`, `divergence.py`, `scoring.py`
   - Pair correlation + spikes -> `pairs.py`
   - Dashboard -> `dashboard.py`

## Notes
- I provided safe defaults and placeholder mappings in `mapping.py`. Replace with your exact scoring table.
- This package is intentionally minimal so you can iterate without being blocked.
