# harness/scheduler/ — resource-DAG runtime & replays (import-coupled)

The only import-coupled group: each file does `sys.path.insert(0, parent)` then
`from miss_assignment_replay import ...`, so these MUST stay co-located.

- `spice_shallow_issuer_runtime.py` — MAIN real-CUDA wall-clock runtime. SPICE shallow H2D
  issuer + residual miss scheduler + verified-gate NEGATIVE admission (`--substitute_ranks`).
- `spice_event_scheduler_replay.py` — unified event-queue deadline-aware DAG replay.
- `spice_forecast_pressure_replay.py` — PCIe-pressure scheduler under SPICE draft forecast.
- `prefetch_pressure_scheduler_replay.py` — pressure-aware DP residual-miss scheduler (layer-serial DAG).
- `miss_assignment_replay.py` — shared library: load_costs / popularity / warm_cache / evict_ls + cost-table replay.

CONSUMES: forecast dumps (`--forecast_dir`, from ../datagen) and cost tables (`--cost_json`, from ../microbench).
