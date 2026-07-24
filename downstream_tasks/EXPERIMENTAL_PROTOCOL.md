# PiKV Experimental Protocol

This document states the **exact settings** used for systems tables in
`downstream_tasks/README.md` and the reproducible ablation harness in
`downstream_tasks/eval/`. For a systems paper, these knobs determine credibility.

## 1. Hardware & software environment

| Setting | Default (artifact) | Notes |
|---------|-------------------|--------|
| GPU | **NVIDIA A100-SXM4-80GB** | Single GPU unless `world_size>1` |
| CUDA | 12.1+ | Match host driver |
| PyTorch | ≥ 2.0 | See `setup.py` / `requirements.txt` |
| Host CPU | AMD EPYC / Intel Xeon (any modern) | Used for dataloader + FPGA host |
| FPGA (optional path) | AMD Alveo U55C (`xcu55c-fsvh2892-2L-e`) | See `core/fpga/README.md` |
| OS | Ubuntu 22.04 | Recommended |

Override GPU in scripts via `CUDA_VISIBLE_DEVICES` and record the value in the run JSON.

## 2. Workload & request shape

| Setting | Default | Rationale |
|---------|---------|-----------|
| Model (tables) | Mistral-7B / Qwen-14B / LLaMA-2-7B | As labeled in each README section |
| Precision | bf16 (A100) | Fair memory comparison |
| Context length | **4096** tokens | Long-context KV pressure |
| Prefill tokens / request | **512** | Shared “token batch” across variants |
| Decode tokens / request | **128** | Fixed generation budget |
| Concurrent requests | **8** | Multi-request fairness (shared GPU) |
| Batch size (microbatch) | **1** per request stream; aggregate concurrency = 8 | Avoids conflating batching with PiKV |
| Warmup | 3 iterations | Discarded from stats |
| Measured runs | **3** | Mean ± range (min–max) in tables |
| Seed | `42 + run_id` | Deterministic routing ties |

**Token budget selection:** keep prefill=512 / decode=128 so every module
combination sees the **same** attention footprint. Do not retune budgets per
compressor or scheduler.

## 2.5 Data download & dataloader (fair prompts)

All evaluation that claims fairness must use the **same frozen prompts**.

```bash
# Download WikiText-2 → data/ (repo root)
python -m data.download_data

# NTP / CE eval on frozen prompts
python -m downstream_tasks.eval.eval_with_data --max-prompts 32

# Module ablation with identical hidden inputs from those prompts
python -m downstream_tasks.eval.ablation_study --preset factor --from-data
```

Details: [`../data/README.md`](../data/README.md).

| Artifact | Role |
|----------|------|
| `data/train.txt` / `data/test.txt` | Sliding-window NTP training |
| `data/prompts_eval.txt` | One prompt/line — **do not regenerate mid-comparison** |
| `data/manifest.json` | Dataset id, seed, sizes |
| `create_eval_dataloader` | Prefill=512 tokenized batches |
| `prompts_to_hidden` | Seeded embed → identical `[B,S,H]` for ablations |

## 3. Fairness controls

1. **Identical prompts** — reuse a frozen prompt file (`data/prompts_eval.txt` from `python -m data.download_data`, or LongBench subset).
2. **Identical generation length** — hard cap `max_new_tokens=128`; no early-exit differences counted as latency wins.
3. **Identical concurrency** — always 8 outstanding requests; no queue draining tricks.
4. **No DVFS surprises** — lock GPU clocks when possible (`nvidia-smi -lgc`); otherwise report variance.
5. **Module isolation** — for single-technique rows, enable **exactly one** of {routing, compression, scheduling, expert-sharding}; others use identity/baseline.
6. **Combined rows** — document the exact triple `(router, compressor, scheduler[, shard])` in the JSON `config` field.

## 4. Metrics

| Metric | Definition |
|--------|------------|
| Latency (ms) | Mean end-to-end time per request (prefill+decode), after warmup |
| KV Mem (GB) | Peak allocated KV / cache tensor bytes (`torch.cuda.max_memory_allocated` proxy or explicit cache nbytes) |
| Compression ↑ | `raw_kv_bytes / stored_kv_bytes` |
| KV Hit ↑ (%) | `cache_hits / (cache_hits + cache_misses) * 100` |
| Accuracy-Drop ↓ (%) | Task metric drop vs dense baseline on the same prompts |
| Load imbalance | Variance / mean of per-expert token counts (routing ablations) |

**Statistical variance:** tables report **mean over 3 runs** and **min–max range**.
Ablation JSON also stores per-run samples and sample std.

## 5. Module combination matrix

| Experiment class | Enabled modules | Script |
|------------------|-----------------|--------|
| Baseline | none (identity) | `eval/ablation_study.py --preset baseline` |
| Routing only | router ∈ {base, adaptive, eplb, pikv, …} | `--preset routing` |
| Compression only | compressor ∈ {pyramid, svd, lora, …} | `--preset compression` |
| Scheduling only | scheduler ∈ {h2o, lru, quest, …} | `--preset scheduling` |
| Sharding only | expert-sharded KV on/off | `--preset sharding` |
| Pairwise / triple | combinations from README matrix | `--preset combined` |
| Full factorial (local) | all singles + key pairs + full stack | `--preset factor` |

**PiKV additional gain over “just combining existing methods”:**
compare `full_stack` against (a) best single module, (b) best pairwise
non-PiKV combo, and (c) additive expectation `sum(Δ_single)`. Report
`synergy = Δ_full - sum(Δ_single)` in the ablation output.

## 6. How to run

```bash
# From repo root
python -m downstream_tasks.eval.ablation_study --preset factor --device cpu
python -m downstream_tasks.eval.ablation_study --preset factor --device cuda --runs 3

# Component smoke (shape checks)
python -m downstream_tasks.eval.routing_experiment
python -m downstream_tasks.eval.compression_experiment
python -m downstream_tasks.eval.scheduling_experiment

# FPGA / CXL characterization (separate from LLM tables)
python -m core.fpga.benchmark_hw --json
```

Artifacts write to `downstream_tasks/eval/results/ablation_<timestamp>.json`.

## 7. Mapping to paper tables

| Paper claim location | Artifact |
|----------------------|----------|
| §5 efficiency tables | `downstream_tasks/README.md` numbers + this protocol |
| Single / double / triple technique | README sections 1–3 + `ablation_study.py` |
| FPGA §3.5 | `core/fpga/README.md` + `core/fpga/benchmark_hw.py` |

If a number cannot be regenerated from the scripts above on the stated GPU,
treat it as **not reproducible from this artifact** and exclude it from claims.
