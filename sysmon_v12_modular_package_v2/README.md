# Sysmon v12 Modular Package

Generated from Sysmon_v12_enhanced.ipynb.


## Stage-based execution

You can run only parts of the pipeline:

```python
pipe.fit(df_baseline)
art = pipe.run(df_new, stages=("enrich","pairs","pair_stats"))
```

Valid stages: `enrich`, `pairs`, `pair_stats`, `markov`, `triage`.
