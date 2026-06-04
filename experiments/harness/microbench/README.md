# harness/microbench/ — real A800 hardware edges & cost tables

Standalone microbenches that PRODUCE the real-hardware numbers the scheduler consumes.

- `miss_assignment_microbench.py` — CPU‖PCIe capacity-aware miss-assignment cost table (the `--cost_json`).
- `shallow_h2d_issuer_microbench.py` — copy-engine queue probe: deep vs shallow H2D submit depth (24.7ms->1.55ms).
- `spice_resource_microbench.py` — A800 edges: H2D fetch, CPU serve, DRAM contention, PCIe priority, copy overlap.
- `cpu_expert_bench.py` — CPU expert compute latency vs PCIe fetch latency (the batch=1 premise: 0.167ms vs 0.78ms).
- `pcie_topology_microbench.py` — H2D/D2H duplex, tiled prefetch, small-copy isolation.
