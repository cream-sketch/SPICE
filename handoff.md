# SPICE Experiment Handoff

This handoff records the current SPICE experiment state, the pitfalls we hit,
the comparison methods we used, and the recommended path for continuing the
memory-limited experiments on L4 / RTX 4060 / other single-GPU servers.

Current GitHub branch:

```bash
https://github.com/cream-sketch/SPICE
branch: exp/nonincremental-miss-recovery
latest pushed commit used for this handoff: 6572c7f3709299dba54b00c4423d89f3c4d8dcc6
```

Current main server:

```bash
host: moe-server-248
code: /data/ziheng/spice
model: /data2/ziheng/models/Qwen2-57B-A14B-Instruct
runs: /data2/ziheng/spice_runs/
```

## 1. Goal

The current experiment is the memory-limited MoE inference setting. The model is
Qwen2-57B-A14B-Instruct in BF16. The target hardware has only one visible GPU,
and routed expert weights are too large to keep fully resident on GPU.

The main metrics are:

- TPOT: time per output token.
- TTFT: time to first token, mainly from actual offloaded HF generation.
- Optional quality metrics should be handled separately; current Qwen2 memory
  limited comparison mainly focuses on inference speed.

The methods to compare are:

- SPICE exact CPU residual orchestration.
- SPICE with low-confidence substitution by shared expert / LoRE surrogate.
- AdapMoE-style active-expert replay baseline.
- HybriMoE-style hybrid CPU-GPU replay baseline.

Internal SPICE ablations such as `deep_fetch_all`, `deep_cpu`,
`shallow_cpu`, `shallow_scheduler`, and `gos_cpu` should not all be reported as
external paper baselines. They are diagnostic policies unless explicitly mapped
to a prior method.

## 2. Current Implemented Code

Important files now pushed to GitHub:

```bash
experiments/harness/run_qwen2_memory_limited_compare.sh
experiments/harness/QWEN2_MEMORY_LIMITED.md
experiments/harness/summarize_qwen2_compare.py
experiments/harness/datagen/gen_decode_traces_offload.py
experiments/harness/datagen/decode_to_real_lore_traces.py
experiments/harness/datagen/make_forecast_from_real_lore.py
experiments/harness/datagen/make_single_decode_manifest.py
experiments/harness/scheduler/adapmoe_trace_replay.py
experiments/harness/run_qwen2_57b_full8_compare.sh
experiments/harness/run_qwen2_57b_full8_tail.sh
```

The portable entrypoint for new servers is:

```bash
MODEL_DIR=/path/to/Qwen2-57B-A14B-Instruct \
RUN_DIR=/path/to/spice_runs/qwen2_57b_l4 \
GPU=0 \
HW_TAG=l4 \
GPU_MEM=22GiB \
CPU_MEM=256GiB \
bash experiments/harness/run_qwen2_memory_limited_compare.sh
```

For RTX 4060 16GB, start more conservatively:

```bash
MODEL_DIR=/path/to/Qwen2-57B-A14B-Instruct \
RUN_DIR=/path/to/spice_runs/qwen2_57b_4060 \
GPU=0 \
HW_TAG=rtx4060 \
GPU_MEM=13GiB \
CPU_MEM=192GiB \
N_PROMPTS=4 \
GEN=32 \
bash experiments/harness/run_qwen2_memory_limited_compare.sh
```

The output summary is written to:

```bash
$RUN_DIR/summary.md
```

## 3. Current A800 Qwen2-57B Full8 Result

Run directory:

```bash
/data2/ziheng/spice_runs/qwen2_57b_full8
```

This is a small full8 run:

- 8 prompts.
- 32 decode tokens per prompt.
- 256 decode trace steps total.
- 128 test decode tokens used in runtime replay.
- Single visible A800 GPU for each runtime process.
- Qwen2-57B-A14B-Instruct BF16.
- Residency budget: `0.1`.

Measured Qwen2 expert shape:

```bash
hidden_size = 3584
moe_intermediate_size = 2560
num_experts = 64
num_experts_per_tok = 8
num_hidden_layers = 28
single routed expert size ~= 55.05 MB in BF16
```

Important hardware measurements on A800:

```bash
resource file: /data2/ziheng/spice_runs/qwen2_57b_bf16/resource_edges_a800_qwen2_57b.json
miss cost file: /data2/ziheng/spice_runs/qwen2_57b_bf16/miss_assign_qwen2_57b_bf16_t16.json

GPU resident expert compute ~= 0.068 ms
H2D fetch one expert ~= 2.478 ms
CPU expert compute ~= 0.646 ms
```

Current comparison table:

| method | variant | TPOT ms/token | relation to SPICE exact |
|---|---:|---:|---:|
| HybriMoE-style | prefetch=4 | 110.93 | 1.69x faster |
| SPICE exact CPU residual | gos_cpu | 187.27 | 1.00x |
| SPICE + low-confidence substitution | rank7 | 192.15 | 1.03x slower |
| AdapMoE-style | active_m=4 | 230.70 | 1.23x slower |
| AdapMoE-style | active_m=6 | 355.56 | 1.90x slower |
| AdapMoE-style | active_m=8 | 485.35 | 2.59x slower |

Interpretation:

- The strongest current method-level baseline result is HybriMoE-style replay,
  which is faster than the current SPICE exact configuration on this A800
  full8 trace.
- The strongest current SPICE configuration is exact CPU residual orchestration.
- LoRE substitution rank7 did not improve TPOT in this small full8 run; it was
  slightly slower than exact CPU residual. Do not claim LoRE improves this
  Qwen2 full8 result unless more tuning or a stronger LoRE checkpoint changes
  the result.
- The current SPICE scheduler is conservative: it sends residual misses to CPU
  and avoids demand fetches. HybriMoE-style replay shows that a more aggressive
  CPU/GPU split can be faster under the measured A800 cost table, so this is now
  a key baseline pressure point to address.

## 4. What SPICE Means in This Experiment

Use this terminology consistently:

1. Speculative prefetch predicts future expert demand with a lightweight draft
   predictor and enqueues predicted experts for asynchronous transfer.

2. Low-confidence substitution applies only when a router-selected expert is
   missed and is judged safe/low-confidence enough to approximate. In the paper
   method, substitution includes two parts:

   - resident shared expert path on GPU;
   - trained LoRE surrogate path.

3. CPU-GPU heterogeneous orchestration handles remaining exact residual misses
   that cannot be safely approximated. The scheduler decides whether to fetch
   the expert to GPU or execute the expert exactly on CPU using host-resident
   weights. The CPU path can overlap with outstanding PCIe transfers and GPU
   work, reducing critical-path stalls.

Avoid saying "exact misses are absorbed by CPU" too absolutely. The correct
claim is that SPICE schedules remaining exact misses between GPU fetch and CPU
execution depending on cost and contention.

## 5. Baseline Clarifications

### AdapMoE

The original AdapMoE repo on the server is:

```bash
/data/ziheng/baselines/AdapMoE
```

Problem:

- Original code is Mixtral/HQQ-oriented.
- It is not a drop-in Qwen2-57B BF16 offload runtime.
- Some scripts try to load the full model on CUDA and would OOM for Qwen2-57B.

Current fair Qwen2 comparison:

```bash
experiments/harness/scheduler/adapmoe_trace_replay.py
```

This is a method-level AdapMoE-style replay on the same Qwen2 route trace and
same hardware cost measurements. It implements:

- finite GPU expert cache;
- active top-m expert gating;
- demand H2D fetch for active misses.

It intentionally does not use:

- SPICE forecast;
- CPU residual execution;
- LoRE substitution.

### Pre-gated MoE

The original Pre-gated MoE repo on the server is:

```bash
/data/ziheng/baselines/Pregated_MoE
```

Problem:

- The artifact targets Switch/T5-style MoE with FasterTransformer.
- It is not a drop-in Qwen2-MoE runtime.
- Directly running it on Qwen2-57B would require a substantial port.

Current status:

The Pre-gated comparison is now abandoned for the main baseline table. Keep the
notes above only as historical context.

### HybriMoE

The HybriMoE repo on the server is:

```bash
/data/ziheng/baselines/HybriMoE
```

Problem:

- The released implementation is based on KTransformers and GGUF/CPUInfer.
- Directly running it would mix policy effects with quantization format,
  custom kernels, and model injection differences.
- The repo has Qwen2-MoE optimize rules, but those rules do not give a clean
  BF16 HuggingFace-runtime comparison against SPICE.

Current fair Qwen2 comparison:

Use a method-level HybriMoE-style replay on the same Qwen2 route trace:

```bash
experiments/harness/scheduler/hybrimoe_trace_replay.py
```

This models:

- per-layer GPU expert cache under the same total residency budget;
- score-aware cache eviction and next-layer prefetch;
- hybrid CPU-GPU scheduling where cache hits run on GPU and residual misses are
  assigned to exact CPU execution or demand fetch + GPU by measured cost.

When writing the paper, phrase it as:

```text
HybriMoE-style hybrid CPU-GPU scheduling replay on the same Qwen2 trace and
hardware cost measurements.
```

Do not claim that the original HybriMoE/KTransformers artifact itself was run
as-is in BF16 Qwen2-57B mode unless that port is completed.

## 6. Pitfalls Already Encountered

### Storage

`/data` was nearly full. Qwen2-57B was downloaded to `/data2`, not `/data`.

Use:

```bash
/data2/ziheng/models/Qwen2-57B-A14B-Instruct
/data2/ziheng/spice_runs/
```

Do not casually delete files. The only model we explicitly deleted was the small
old Qwen1.5-MoE-A2.7B copy after user approval.

### Qwen2-57B Download

Direct HuggingFace download was unstable. The mirror worked:

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HOME=/data2/ziheng/hf_home \
HF_HUB_CACHE=/data2/ziheng/hf_home/hub \
hf download Qwen/Qwen2-57B-A14B-Instruct \
  --local-dir /data2/ziheng/models/Qwen2-57B-A14B-Instruct \
  --cache-dir /data2/ziheng/hf_home/hub \
  --max-workers 8
```

### Accelerate

The remote conda env had a broken `accelerate` symlink. We fixed it by removing
the broken package path and reinstalling:

```bash
/data/ziheng/miniconda3/envs/caproute_vllm/bin/python -m pip install --no-deps accelerate==1.13.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If CPU offload fails with a `device_map` or `accelerate` error, check this first:

```bash
/data/ziheng/miniconda3/envs/caproute_vllm/bin/python - <<'PY'
import accelerate, transformers
print(accelerate.__version__)
print(transformers.utils.is_accelerate_available())
PY
```

### Batch Size

For these runs, batch size effectively stays at 1 because:

- Qwen2-57B BF16 is already memory-heavy;
- CPU offload and expert tracing are fragile under larger batches;
- routing trace collection and hidden-state capture can explode memory.

For final throughput experiments, larger batch sizes can be explored later, but
do not silently compare batch=1 SPICE against a different-batch baseline.

### Full Dataset vs Smoke

Earlier "smoke" runs used small subsets to verify code paths. They should not
be presented as full dataset results.

For the current Qwen2 memory-limited speed study, "full8" means 8 prompts and
256 decode trace steps, not full GSM8K/LongBench/HumanEval. Be explicit.

### LoRE Quality

The current Qwen2 full8 LoRE training log showed low route quality:

```bash
slot hit ~= 0.26
fallback slot ~= 0.74
exact set match ~= 0.0017
mean confidence ~= 0.153
```

This likely explains why rank7 substitution did not improve TPOT. Before making
a strong LoRE claim, retrain/tune on more traces or use a better training setup.

## 7. How To Continue On A New Server

### Step 1: Clone and checkout branch

```bash
git clone https://github.com/cream-sketch/SPICE.git
cd SPICE
git checkout exp/nonincremental-miss-recovery
```

### Step 2: Prepare Python environment

Required Python packages:

```bash
pip install torch transformers accelerate safetensors datasets numpy pandas
```

Use a PyTorch build compatible with the target GPU and CUDA version.

### Step 3: Prepare model

Put Qwen2 model on local fast storage:

```bash
/path/to/Qwen2-57B-A14B-Instruct
```

Make sure these files exist:

```bash
config.json
model.safetensors.index.json
*.safetensors
tokenizer files
```

### Step 4: Run portable pipeline

For L4:

```bash
MODEL_DIR=/path/to/Qwen2-57B-A14B-Instruct \
RUN_DIR=/path/to/spice_runs/qwen2_57b_l4 \
GPU=0 \
HW_TAG=l4 \
GPU_MEM=22GiB \
CPU_MEM=256GiB \
N_PROMPTS=8 \
GEN=32 \
bash experiments/harness/run_qwen2_memory_limited_compare.sh
```

For RTX 4060 16GB:

```bash
MODEL_DIR=/path/to/Qwen2-57B-A14B-Instruct \
RUN_DIR=/path/to/spice_runs/qwen2_57b_4060 \
GPU=0 \
HW_TAG=rtx4060 \
GPU_MEM=13GiB \
CPU_MEM=192GiB \
N_PROMPTS=4 \
GEN=32 \
bash experiments/harness/run_qwen2_memory_limited_compare.sh
```

If the model OOMs during trace collection, reduce:

```bash
GPU_MEM
N_PROMPTS
GEN
PROMPT_LEN
```

Do not reduce only one method's workload. Keep all methods on the same trace.

### Step 5: Summarize existing run

```bash
python experiments/harness/summarize_qwen2_compare.py \
  --run_dir /path/to/run \
  --out /path/to/run/summary.md
```

## 8. What To Control For Fairness

Keep these fixed across SPICE and baselines:

- same model checkpoint;
- same dtype;
- same single visible GPU;
- same prompt set;
- same decode trace;
- same number of evaluated decode tokens;
- same top-k routing;
- same expert residency budget;
- same measured hardware cost table;
- same CPU thread count;
- same GPU filler/compute window settings;
- same train/test split in replay.

Do not compare:

- Qwen2 SPICE against a native artifact with a different model runtime or
  quantization format;
- batch=1 SPICE against a larger-batch baseline;
- smoke subset results against full-dataset baseline results;
- SPICE with LoRE substitution against a baseline using different accuracy
  semantics unless quality impact is separately measured.

## 9. Current Open Issues

1. LoRE substitution needs improvement.

   Current rank7 substitution is slightly slower than exact CPU residual on
   Qwen2 full8. Need better LoRE training, more traces, threshold tuning, or
   quality-aware substitution policy before claiming it improves the final
   result.

2. Accuracy datasets are not yet complete for Qwen2-57B memory-limited runs.

   Earlier discussion included GSM8K, LongBench, HumanEval, and MT-Bench, but
   MT-Bench was dropped because it mainly measures chat quality rather than
   speed/precision in this setting. For future accuracy + speed, choose datasets
   that can report both task score and TTFT/TPOT under the same generation path.

3. AdapMoE and HybriMoE are method-level Qwen2 replays.

   This is currently the fair way to compare on Qwen2 because original artifacts
   are not directly compatible. The paper should state this clearly.

4. Need repeat runs on L4 and RTX 4060.

   These are important because SPICE should be strongest when PCIe/memory
   pressure is severe and GPU compute is not the only bottleneck.

## 10. Quick Commands On Current Server

Check GPU:

```bash
nvidia-smi
```

Check running experiment processes:

```bash
ps -eo pid,stat,etime,cmd | grep -E 'qwen2|spice|adapmoe|pregated' | grep -v grep
screen -ls
```

Summarize current full8 result:

```bash
cd /data/ziheng/spice
/data/ziheng/miniconda3/envs/caproute_vllm/bin/python \
  experiments/harness/summarize_qwen2_compare.py \
  --run_dir /data2/ziheng/spice_runs/qwen2_57b_full8 \
  --out /data2/ziheng/spice_runs/qwen2_57b_full8/summary.md
```

Run current portable pipeline on A800-style settings:

```bash
cd /data/ziheng/spice
MODEL_DIR=/data2/ziheng/models/Qwen2-57B-A14B-Instruct \
RUN_DIR=/data2/ziheng/spice_runs/qwen2_57b_new \
PYTHON=/data/ziheng/miniconda3/envs/caproute_vllm/bin/python \
GPU=2 \
HW_TAG=a800 \
GPU_MEM=70GiB \
CPU_MEM=600GiB \
N_PROMPTS=8 \
GEN=32 \
bash experiments/harness/run_qwen2_memory_limited_compare.sh
```

## 11. Suggested Paper Wording

For the current SPICE method:

```text
After speculative prefetching and low-confidence substitution, a residual
bottleneck remains: some router-selected experts are still absent from the GPU
but cannot be safely approximated. SPICE handles these remaining misses with a
CPU-GPU heterogeneous scheduler, which decides whether each expert should be
fetched to the GPU or executed exactly on the CPU using host-resident weights.
By overlapping CPU execution with outstanding transfers and GPU computation,
SPICE reduces the critical-path latency caused by residual expert misses.
```

For HybriMoE baseline:

```text
Because the released HybriMoE artifact is a KTransformers/GGUF runtime, we
implement a method-level HybriMoE-style replay on the same Qwen2 route trace and
hardware cost measurements to isolate the scheduling and cache policy from
quantization and custom-kernel effects.
```

For current LoRE result:

```text
In the current Qwen2 full8 run, the exact CPU residual path is the strongest
configuration. The low-confidence LoRE substitution path requires additional
training and threshold tuning before being used as the main speed result.
```
