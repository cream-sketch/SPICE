# harness/datagen/ — routing traces & SPICE forecast dumps

PRODUCES the data the scheduler/replays consume (`--forecast_dir`, `--trace_dir`).

- `qwen_spice_draft.py` — training-free SPICE draft: routing prediction + forecast dump (true_top, fcast).
- `gen_decode_traces.py` / `gen_decode_traces_ds.py` — Qwen / DeepSeek autoregressive decode routing traces (dec_*.pt).
- `make_forecast_from_dec.py` — build a forecast dump (true_top real, fcast placeholder) from decode traces,
  for depth=0 negative-admission runs on models without a draft-forecast dump (e.g. DeepSeek).
