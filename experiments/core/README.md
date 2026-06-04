# core/ — original SPICE reproduction (verified speculative expert prefetch)

The original SPICE system that reproduces the paper (up to 2.86x TPOT on PCIe-constrained /
weaker GPUs). Stable reference code. For the batch=1 / fast-GPU negative-admission system see `../harness/`.

## Layout
- **root (shared libraries, kept flat so every subdir can import them):**
  - `common.py` — utilities: build_arg_parser, device_from_arg, ensure_dir, set_seed, write_json, etc.
  - `draft_model.py` — the SPICE draft: frozen target attention/router + LoRE expert surrogates +
    routing-history context + route-KL & hidden-alignment losses + routing metrics.
- **`draft/`** — train & evaluate the draft path:
  - `train_draft_model.py` (train the LoRE predictor), `eval_draft_prefetch.py` (anchor-reinit
    lookahead prefetch + verified fallback + online self-correction).
- **`sim/`** — system simulation & correctness:
  - `prefetch_system_sim.py` (Naive/LRU/MoE-Offloading/Pre-gated/SPICE policies + PCIe copy microbench),
    `lossless_correctness.py` (proves fallback affects latency not logits), `eval_hf_trace_prefetch.py`.
- **`data/`** — trace/model acquisition:
  - `collect_hf_moe_traces.py`, `download_hf_snapshot.py`, `real_ppl_smoke.py`.
- **`analysis/`** — summaries & timeline/energy:
  - `summarize_results.py`, `summarize_iccd_system_results.py`, `timeline_replay.py`, `copy_timeline.py`,
    `energy_per_token.py`, `power_trace.py`, `sota_proxy.py`.
- `run_*.sh` — the 4-GPU workstation suites (paths updated to the subdirs above).

## Import contract
Every subdir file bootstraps `sys.path.insert(0, parent.parent)` (the core/ root) then
`from common import ...` (and `draft/` files `from draft_model import ...`). So the two shared
libraries MUST stay at the core/ root. Run scripts from anywhere; the bootstrap is CWD-independent.

Verified post-reorg: `sim/prefetch_system_sim.py` runs (common import resolves via parent.parent).
