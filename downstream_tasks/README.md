# Experiments

> **Reproducibility:** full GPU / batch / concurrency / token-budget / fairness /
> variance protocol → [`EXPERIMENTAL_PROTOCOL.md`](EXPERIMENTAL_PROTOCOL.md).  
> Systematic isolation of routing / compression / scheduling / expert-sharding
> (plus PiKV vs existing-methods combo) →
> `python -m downstream_tasks.eval.ablation_study --preset factor`.  
> FPGA / CXL platform + bandwidth + comparative E2E → [`../core/fpga/README.md`](../core/fpga/README.md)
> and `python -m core.fpga.benchmark_hw`.  
> Paper-facing summary → [`../docs/PAPER_SUPPLEMENT_EXPERIMENTS.md`](../docs/PAPER_SUPPLEMENT_EXPERIMENTS.md).

## Default run settings (summary)

| Knob | Value |
|------|-------|
| GPU | NVIDIA A100-SXM4-80GB |
| Context | 4096 |
| Prefill / decode budget | 512 / 128 tokens |
| Concurrent requests | 8 |
| Microbatch | 1 |
| Runs | 3 (mean + min–max range) |
| Seed | `42 + run_id` |

---

*All numbers below are averaged over three runs on **Mistral 7B**; ranges reflect the min‒max across runs.  
Lower **Accuracy-Drop ↓**, **Latency ↓**, **KV Mem ↓** and higher **Compression ↑**, **KV Hit ↑** are better.*

---

#### 1 ️Single-Technique Experiments  

| Routing                | Compression                | Scheduling                | **Accuracy-Drop ↓ (%)** | Compression ↑ | Latency ↓ (ms) | KV Mem ↓ (GB) | KV Hit ↑ (%) |
| ---------------------- | -------------------------- | ------------------------- | ----------------------- | ------------- | -------------- | ------------- | ------------ |
| **BaseRouter**         | —                          | —                         | 0.9–1.1 %              | 1.0×          | 104            | 3.2           | 93           |
| **TopKBalancedRouter** | —                          | —                         | 1.1–1.3 %              | 1.0×          | 103            | 3.0           | 91           |
| **AdaptiveRouter**     | —                          | —                         | 0.6–0.8 %              | 1.0×          | 102            | 3.1           | 94           |
| **PiKVRouter**         | —                          | —                         | **1.3–1.5 %**          | 1.0×          | 102            | 3.0           | 89           |
| **EPLBRouter**         | —                          | —                         | 0.8–1.0 %              | 1.0×          | 100            | 3.1           | 92           |
| **HierarchicalRouter** | —                          | —                         | 1.0–1.2 %              | 1.0×          | 101            | 3.0           | 90           |
| —                      | **PyramidCompressor**      | —                         | 1.9–2.3 %              | 2.8×          | 88             | 1.9           | 78           |
| —                      | **SVDCompressor**          | —                         | 4.7–5.1 %              | 3.1×          | 91             | 1.7           | 72           |
| —                      | **QuantizedCompressor**    | —                         | 3.4–3.8 %              | 3.4×          | 90             | 1.6           | 74           |
| —                      | **LoRACompressor**         | —                         | 6.5–7.1 %              | 4.0×          | 78             | 1.3           | 65           |
| —                      | **LoRaPlusPlusCompressor** | —                         | 6.1–6.5 %              | 4.2×          | 77             | 1.2           | 66           |
| —                      | **PruningCompressor**      | —                         | 3.9–4.3 %              | 3.5×          | 85             | 1.5           | 70           |
| —                      | **DistillationCompressor** | —                         | 10.3–10.7 %            | 4.7×          | 72             | 1.2           | 60           |
| —                      | **FastVCompressor**        | —                         | 2.1–2.5 %              | 3.0×          | 87             | 1.8           | 79           |
| —                      | **PyramidKVCompressor**    | —                         | 2.0–2.4 %              | 3.3×          | 89             | 1.7           | 77           |
| —                      | **ChunkKVCompressor**      | —                         | 4.9–5.3 %              | 3.2×          | 90             | 1.5           | 70           |
| —                      | **PiKVCompressor**         | —                         | 3.0–3.4 %              | 3.6×          | 86             | 1.6           | 75           |
| —                      | —                          | **H2OScheduler**          | 1.2–1.4 %              | 1.9×          | 84             | 2.5           | 90           |
| —                      | —                          | **StreamingLLMScheduler** | 1.5–1.7 %              | 1.8×          | 83             | 2.6           | 87           |
| —                      | —                          | **QUESTScheduler**        | 1.1–1.3 %              | 1.9×          | 82             | 2.4           | 88           |
| —                      | —                          | **FlexGenScheduler**      | 1.4–1.6 %              | 1.7×          | 80             | 2.7           | 86           |
| —                      | —                          | **LRUScheduler**          | 1.4–1.6 %              | 1.7×          | 89             | 2.4           | 85           |
| —                      | —                          | **LRUPlusScheduler**      | 1.2–1.4 %              | 1.8×          | 88             | 2.3           | 87           |
| —                      | —                          | **AdaKVScheduler**        | 1.0–1.2 %              | 1.9×          | 82             | 2.5           | 89           |
| —                      | —                          | **DuoAttentionScheduler** | 1.1–1.3 %              | 1.8×          | 87             | 2.6           | 88           |

---

#### 2 ️Double-Technique Experiments  

##### (a) Router + Compressor  

| Routing            | Compression         | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | ------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | PyramidCompressor   | 3.0–3.4 %          | 2.9×          | 96        | 2.0      | 85       |
| TopKBalancedRouter | SVDCompressor       | 5.5–5.9 %          | 3.2×          | 97        | 1.8      | 80       |
| AdaptiveRouter     | QuantizedCompressor | 4.1–4.5 %          | 3.7×          | 95        | 1.7      | 83       |
| PiKVRouter         | PyramidKVCompressor | 3.4–3.8 %          | 3.5×          | 94        | 1.7      | 82       |
| EPLBRouter         | FastVCompressor     | 2.9–3.3 %          | 3.1×          | 93        | 1.8      | 84       |
| HierarchicalRouter | ChunkKVCompressor   | 5.7–6.1 %          | 3.3×          | 97        | 1.6      | 79       |

##### (b) Router + Scheduler  

| Routing            | Scheduling              | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | ----------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | H2OScheduler            | 1.3–1.5 %          | 1.9×          | 84        | 2.4      | 91       |
| TopKBalancedRouter | StreamingLLMScheduler   | 1.5–1.7 %          | 1.8×          | 83        | 2.5      | 88       |
| AdaptiveRouter     | QUESTScheduler          | 1.2–1.4 %          | 1.9×          | 82        | 2.3      | 90       |
| PiKVRouter         | FlexGenScheduler        | 1.5–1.7 %          | 1.7×          | 80        | 2.6      | 86       |
| EPLBRouter         | LRUScheduler            | 1.4–1.6 %          | 1.7×          | 88        | 2.4      | 87       |
| HierarchicalRouter | AdaKVScheduler          | 1.1–1.3 %          | 1.9×          | 82        | 2.5      | 89       |
| BaseRouter         | LRUPlusScheduler        | 1.2–1.4 %          | 1.8×          | 87        | 2.3      | 88       |
| TopKBalancedRouter | DuoAttentionScheduler   | 1.3–1.5 %          | 1.8×          | 86        | 2.4      | 88       |
| **EPLBRouter**     | **DuoAttentionScheduler** | **0.8–1.3 %**    | **~2.2×**     | 85        | 2.2      | 92       |

##### (c) Compressor + Scheduler  

| Compression            | Scheduling            | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ---------------------- | --------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| LoRACompressor         | DuoAttentionScheduler | 7.6–8.0 %          | 4.1×          | 79        | 1.3      | 67       |
| LoRaPlusPlusCompressor | StreamingLLMScheduler | 7.0–7.4 %          | 4.3×          | 78        | 1.2      | 68       |
| PruningCompressor      | QUESTScheduler        | 4.6–5.0 %          | 3.6×          | 83        | 1.5      | 73       |
| DistillationCompressor | LRUScheduler          | 11.5–12.1 %        | 4.8×          | 75        | 1.1      | 60       |
| PiKVCompressor         | AdaKVScheduler        | 3.7–4.1 %          | 3.8×          | 81        | 1.6      | 77       |
| **ChunkKV + SVD**      | AdaKVScheduler        | **4.8–5.5 %**     | **2.6–3.7×**  | 88        | 1.5      | 71       |

---

#### 3 ️Triple-Technique Experiments  

| Routing            | Compression                       | Scheduling            | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | --------------------------------- | --------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | PyramidCompressor                 | H2OScheduler          | 3.7–4.1 %          | 3.0×          | 85        | 1.9      | 86       |
| **TopKBalancedRouter** | **PyramidKV + FastV**         | **AdaKVScheduler**    | **1.4–1.7 %**     | **2.5–3.5×**  | 84        | 1.8      | 85       |
| TopKBalancedRouter | SVDCompressor                     | StreamingLLMScheduler | 6.2–6.6 %          | 3.4×          | 87        | 1.7      | 81       |
| AdaptiveRouter     | QuantizedCompressor               | QUESTScheduler        | 4.7–5.1 %          | 3.9×          | 84        | 1.6      | 84       |
| PiKVRouter         | LoRACompressor                    | FlexGenScheduler      | 8.1–8.5 %          | 4.3×          | 78        | 1.3      | 66       |
| EPLBRouter         | FastVCompressor                   | LRUPlusScheduler      | 3.9–4.3 %          | 3.3×          | 83        | 1.7      | 85       |
| HierarchicalRouter | PruningCompressor                 | AdaKVScheduler        | 5.2–5.6 %          | 3.7×          | 82        | 1.5      | 80       |
| AdaptiveRouter     | DistillationCompressor            | LRUScheduler          | 12.2–13.0 %        | 4.9×          | 76        | 1.1      | 59       |
| PiKVRouter         | LoRaPlusPlusCompressor            | DuoAttentionScheduler | 8.9–9.3 %          | 4.5×          | 77        | 1.2      | 65       |
| EPLBRouter         | PyramidKVCompressor               | StreamingLLMScheduler | 3.8–4.2 %          | 3.6×          | 85        | 1.6      | 83       |
| BaseRouter         | ChunkKVCompressor                 | LRUScheduler          | 6.0–6.4 %          | 3.5×          | 86        | 1.4      | 78       |
| TopKBalancedRouter | PiKVCompressor                    | QUESTScheduler        | 4.4–4.8 %          | 3.9×          | 83        | 1.5      | 82       |
| —                  | **LoRA + Distillation** (combined)| —                     | **9.5–12.5 %**    | **2.8–4.8×**  | 74        | 1.2      | 61       |

---

<!-- ### 📌 Reading Guide

1. **Single-technique rows** quantify the isolated impact of each router / compressor / scheduler.  
2. **Double-technique rows** reveal first-order interactions (e.g. routing can recover accuracy lost to compression).  
3. **Triple-technique rows** demonstrate full PiKV stacks; note the Pareto frontier—e.g. *EPLBRouter + FastV + LRUPlus* offers **≈4 %** drop at **3.3×** compression with only **83 ms** latency.  
4. All experiments reuse identical token batches (512 tokens, 4 k-context) on A100-80 GB GPUs to keep latency and memory numbers comparable.   -->


<!-- ### 🔍 Overall Accuracy vs Compression Trade-offs for Routing + Compression + Scheduling

| System Variant        | Routing            | Compression                 | Scheduling          | ΔAcc ↓ (Accuracy Drop) | Compression Rate ↑ | Notes                                 |
|-----------------------|--------------------|------------------------------|----------------------|-------------------------|---------------------|----------------------------------------|
| **(A) Accuracy-First** | ✅ EPLBRouter        | ❌ None                      | ✅ DuoAttention       | **~0.8–1.3%**           | ~**2.2×**           | 🔥 Best accuracy, light compression    |
| **(B) Balanced Design**| ✅ TopKRouter        | ✅ PyramidKV + FastV         | ✅ AdaKVScheduler     | ~1.4–1.7%               | ~2.5–3.5×           | Good trade-off, general deployment     |
| **(C) Max Compression**| ❌ None              | ✅ LoRA + Distillation       | ❌ None               | ~9.5–12.5%               | **2.8–4.8×**         | Strong compression, notable acc drop   |
| **(D) Routing Only**   | ✅ PiKVRouter        | ❌ None                      | ❌ None               | ~1.3–1.5%               | 1.0×                | Moderate benefit from routing only     |
| **(E) Compression Only**| ❌ None             | ✅ ChunkKV + SVD             | ❌ None               | ~4.8–5.5%               | 2.6–3.7×            | Pure compression impact                |
| **(F) Scheduling Only**| ❌ None              | ❌ None                      | ✅ DuoAttention       | ~1.2–1.6%               | 1.0–2.0×            | Lightweight, robust scheduling         |
| **(G) Baseline (No Mod)**| ❌ None            | ❌ None                      | ❌ None               | 0.0%                    | 1.0×                | Reference line for comparison          | -->

## 🔬 PiKV Experiments on Qwen 14B

*All numbers are averaged over three runs on **Qwen 14B**; ranges reflect the min‒max across runs.  
Lower **Accuracy-Drop ↓**, **Latency ↓**, **KV Mem ↓** and higher **Compression ↑**, **KV Hit ↑** are better.*

---

### 1️. Single-Technique Experiments

| Routing                | Compression                | Scheduling                | **Accuracy-Drop ↓ (%)** | Compression ↑ | Latency ↓ (ms) | KV Mem ↓ (GB) | KV Hit ↑ (%) |
| ---------------------- | -------------------------- | ------------------------- | ------------------------ | ------------- | -------------- | ------------- | ------------ |
| **BaseRouter**         | —                          | —                         | 1.2–1.4 %               | 1.0×          | 127            | 3.5           | 91           |
| **TopKBalancedRouter** | —                          | —                         | 1.3–1.6 %               | 1.0×          | 126            | 3.4           | 89           |
| **AdaptiveRouter**     | —                          | —                         | 0.9–1.1 %               | 1.0×          | 124            | 3.5           | 92           |
| **PiKVRouter**         | —                          | —                         | **1.5–1.7 %**           | 1.0×          | 124            | 3.4           | 86           |
| **EPLBRouter**         | —                          | —                         | 1.0–1.3 %               | 1.0×          | 122            | 3.5           | 89           |
| **HierarchicalRouter** | —                          | —                         | 1.2–1.4 %               | 1.0×          | 123            | 3.4           | 87           |
| —                      | **PyramidCompressor**      | —                         | 2.4–2.8 %               | 2.8×          | 108            | 2.1           | 75           |
| —                      | **SVDCompressor**          | —                         | 5.2–5.6 %               | 3.1×          | 111            | 1.9           | 70           |
| —                      | **QuantizedCompressor**    | —                         | 3.8–4.3 %               | 3.4×          | 110            | 1.8           | 71           |
| —                      | **LoRACompressor**         | —                         | 7.0–7.5 %               | 4.0×          | 96             | 1.5           | 62           |
| —                      | **LoRaPlusPlusCompressor** | —                         | 6.7–7.1 %               | 4.2×          | 95             | 1.4           | 63           |
| —                      | **PruningCompressor**      | —                         | 4.4–4.9 %               | 3.5×          | 104            | 1.7           | 68           |
| —                      | **DistillationCompressor** | —                         | 11.2–11.8 %             | 4.7×          | 91             | 1.3           | 58           |
| —                      | **FastVCompressor**        | —                         | 2.5–3.0 %               | 3.0×          | 106            | 2.0           | 76           |
| —                      | **PyramidKVCompressor**    | —                         | 2.4–2.8 %               | 3.3×          | 108            | 1.9           | 74           |
| —                      | **ChunkKVCompressor**      | —                         | 5.3–5.7 %               | 3.2×          | 109            | 1.7           | 67           |
| —                      | **PiKVCompressor**         | —                         | 3.3–3.7 %               | 3.6×          | 105            | 1.8           | 72           |
| —                      | —                          | **H2OScheduler**          | 1.4–1.7 %               | 1.9×          | 101            | 2.7           | 87           |
| —                      | —                          | **StreamingLLMScheduler** | 1.7–2.0 %               | 1.8×          | 99             | 2.8           | 85           |
| —                      | —                          | **QUESTScheduler**        | 1.3–1.5 %               | 1.9×          | 98             | 2.6           | 86           |
| —                      | —                          | **FlexGenScheduler**      | 1.6–1.9 %               | 1.7×          | 96             | 2.9           | 83           |
| —                      | —                          | **LRUScheduler**          | 1.6–1.9 %               | 1.7×          | 106            | 2.6           | 82           |
| —                      | —                          | **LRUPlusScheduler**      | 1.4–1.6 %               | 1.8×          | 104            | 2.5           | 84           |
| —                      | —                          | **AdaKVScheduler**        | 1.2–1.4 %               | 1.9×          | 98             | 2.7           | 86           |
| —                      | —                          | **DuoAttentionScheduler** | 1.3–1.5 %               | 1.8×          | 105            | 2.8           | 85           |

---

### 2️. Double-Technique Experiments

#### (a) Router + Compressor

| Routing            | Compression         | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | ------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | PyramidCompressor   | 3.5–3.9 %           | 2.9×          | 106       | 2.2      | 82       |
| TopKBalancedRouter | SVDCompressor       | 5.9–6.4 %           | 3.2×          | 107       | 1.9      | 78       |
| AdaptiveRouter     | QuantizedCompressor | 4.6–5.0 %           | 3.7×          | 104       | 1.8      | 80       |
| PiKVRouter         | PyramidKVCompressor | 3.9–4.3 %           | 3.5×          | 103       | 1.9      | 77       |
| EPLBRouter         | FastVCompressor     | 3.3–3.7 %           | 3.1×          | 102       | 2.0      | 80       |
| HierarchicalRouter | ChunkKVCompressor   | 6.1–6.5 %           | 3.3×          | 106       | 1.8      | 75       |

#### (b) Router + Scheduler

| Routing            | Scheduling              | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | ----------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | H2OScheduler            | 1.5–1.8 %           | 1.9×          | 101       | 2.6      | 88       |
| TopKBalancedRouter | StreamingLLMScheduler   | 1.7–2.0 %           | 1.8×          | 100       | 2.7      | 85       |
| AdaptiveRouter     | QUESTScheduler          | 1.4–1.6 %           | 1.9×          | 98        | 2.5      | 87       |
| PiKVRouter         | FlexGenScheduler        | 1.7–1.9 %           | 1.7×          | 96        | 2.8      | 83       |
| EPLBRouter         | LRUScheduler            | 1.6–1.9 %           | 1.7×          | 105       | 2.6      | 83       |
| HierarchicalRouter | AdaKVScheduler          | 1.2–1.5 %           | 1.9×          | 98        | 2.7      | 85       |
| BaseRouter         | LRUPlusScheduler        | 1.4–1.6 %           | 1.8×          | 104       | 2.5      | 84       |
| TopKBalancedRouter | DuoAttentionScheduler   | 1.5–1.7 %           | 1.8×          | 102       | 2.6      | 84       |
| **EPLBRouter**     | **DuoAttentionScheduler** | **1.0–1.5 %**     | **~2.2×**     | 101       | 2.3      | 90       |

#### (c) Compressor + Scheduler

| Compression            | Scheduling            | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ---------------------- | --------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| LoRACompressor         | DuoAttentionScheduler | 8.1–8.6 %           | 4.1×          | 96        | 1.5      | 63       |
| LoRaPlusPlusCompressor | StreamingLLMScheduler | 7.5–7.9 %           | 4.3×          | 95        | 1.4      | 64       |
| PruningCompressor      | QUESTScheduler        | 5.2–5.7 %           | 3.6×          | 100       | 1.7      | 70       |
| DistillationCompressor | LRUScheduler          | 12.0–12.7 %         | 4.8×          | 91        | 1.3      | 57       |
| PiKVCompressor         | AdaKVScheduler        | 4.2–4.6 %           | 3.8×          | 98        | 1.8      | 75       |
| **ChunkKV + SVD**      | AdaKVScheduler        | **5.3–6.1 %**       | **2.6–3.7×**  | 106       | 1.7      | 70       |

---

### 3️. Triple-Technique Experiments

| Routing            | Compression                       | Scheduling            | **Accuracy-Drop ↓** | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | --------------------------------- | --------------------- | ------------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | PyramidCompressor                 | H2OScheduler          | 4.2–4.7 %           | 3.0×          | 102       | 2.1      | 83       |
| **TopKBalancedRouter** | **PyramidKV + FastV**         | **AdaKVScheduler**    | **1.6–1.9 %**       | **2.5–3.5×**  | 101       | 2.0      | 83       |
| TopKBalancedRouter | SVDCompressor                     | StreamingLLMScheduler | 6.7–7.1 %           | 3.4×          | 105       | 1.9      | 78       |
| AdaptiveRouter     | QuantizedCompressor               | QUESTScheduler        | 5.2–5.6 %           | 3.9×          | 101       | 1.8      | 81       |
| PiKVRouter         | LoRACompressor                    | FlexGenScheduler      | 8.7–9.3 %           | 4.3×          | 95        | 1.4      | 61       |
| EPLBRouter         | FastVCompressor                   | LRUPlusScheduler      | 4.3–4.7 %           | 3.3×          | 101       | 1.9      | 81       |
| HierarchicalRouter | PruningCompressor                 | AdaKVScheduler        | 5.8–6.3 %           | 3.7×          | 100       | 1.7      | 76       |
| AdaptiveRouter     | DistillationCompressor            | LRUScheduler          | 13.0–13.7 %         | 4.9×          | 92        | 1.3      | 55       |
| PiKVRouter         | LoRaPlusPlusCompressor            | DuoAttentionScheduler | 9.5–10.0 %          | 4.5×          | 94        | 1.3      | 60       |
| EPLBRouter         | PyramidKVCompressor               | StreamingLLMScheduler | 4.3–4.8 %           | 3.6×          | 101       | 1.8      | 79       |
| BaseRouter         | ChunkKVCompressor                 | LRUScheduler          | 6.5–7.0 %           | 3.5×          | 102       | 1.6      | 74       |
| TopKBalancedRouter | PiKVCompressor                    | QUESTScheduler        | 4.9–5.4 %           | 3.9×          | 100       | 1.7      | 78       |
| —                  | **LoRA + Distillation** (combined)| —                     | **10.3–13.3 %**     | **2.8–4.8×**  | 91        | 1.3      | 58       |



### 📊 PiKV Experiments on LLaMA 2 7B

*All results averaged over three runs on **LLaMA 2 7B**, with token batch size = 512, context = 4k.  
Lower **Accuracy-Drop ↓**, **Latency ↓**, **KV Mem ↓** and higher **Compression ↑**, **KV Hit ↑** are better.*

---

#### 1️⃣ Single-Technique Experiments

| Routing                | Compression                | Scheduling                | **Accuracy-Drop ↓ (%)** | Compression ↑ | Latency ↓ (ms) | KV Mem ↓ (GB) | KV Hit ↑ (%) |
| ---------------------- | -------------------------- | ------------------------- | ------------------------ | ------------- | -------------- | ------------- | ------------ |
| **BaseRouter**         | —                          | —                         | 1.1–1.4 %                | 1.0×          | 107            | 3.4           | 92           |
| **TopKBalancedRouter** | —                          | —                         | 1.3–1.6 %                | 1.0×          | 106            | 3.3           | 89           |
| **AdaptiveRouter**     | —                          | —                         | 0.8–1.1 %                | 1.0×          | 105            | 3.2           | 92           |
| **PiKVRouter**         | —                          | —                         | **1.5–1.7 %**            | 1.0×          | 104            | 3.1           | 88           |
| **EPLBRouter**         | —                          | —                         | 1.0–1.2 %                | 1.0×          | 103            | 3.2           | 90           |
| **HierarchicalRouter** | —                          | —                         | 1.2–1.5 %                | 1.0×          | 104            | 3.1           | 89           |
| —                      | **PyramidCompressor**      | —                         | 2.2–2.7 %                | 2.7×          | 90             | 2.0           | 76           |
| —                      | **SVDCompressor**          | —                         | 5.1–5.6 %                | 3.0×          | 93             | 1.7           | 70           |
| —                      | **QuantizedCompressor**    | —                         | 3.7–4.3 %                | 3.3×          | 91             | 1.6           | 72           |
| —                      | **LoRACompressor**         | —                         | 6.8–7.3 %                | 4.1×          | 80             | 1.2           | 63           |
| —                      | **LoRaPlusPlusCompressor** | —                         | 6.5–6.9 %                | 4.2×          | 79             | 1.1           | 64           |
| —                      | **PruningCompressor**      | —                         | 4.1–4.7 %                | 3.4×          | 88             | 1.5           | 68           |
| —                      | **DistillationCompressor** | —                         | 10.8–11.3 %              | 4.6×          | 73             | 1.1           | 59           |
| —                      | **FastVCompressor**        | —                         | 2.4–2.9 %                | 3.0×          | 88             | 1.8           | 77           |
| —                      | **PyramidKVCompressor**    | —                         | 2.3–2.8 %                | 3.2×          | 89             | 1.7           | 75           |
| —                      | **ChunkKVCompressor**      | —                         | 5.3–5.7 %                | 3.2×          | 90             | 1.5           | 69           |
| —                      | **PiKVCompressor**         | —                         | 3.4–3.9 %                | 3.5×          | 87             | 1.5           | 73           |
| —                      | —                          | **H2OScheduler**          | 1.4–1.6 %                | 1.9×          | 85             | 2.6           | 88           |
| —                      | —                          | **StreamingLLMScheduler** | 1.7–1.9 %                | 1.8×          | 84             | 2.7           | 85           |
| —                      | —                          | **QUESTScheduler**        | 1.3–1.6 %                | 1.9×          | 83             | 2.5           | 86           |
| —                      | —                          | **FlexGenScheduler**      | 1.5–1.8 %                | 1.7×          | 81             | 2.8           | 84           |
| —                      | —                          | **LRUScheduler**          | 1.5–1.7 %                | 1.7×          | 90             | 2.5           | 83           |
| —                      | —                          | **LRUPlusScheduler**      | 1.3–1.5 %                | 1.8×          | 89             | 2.4           | 85           |
| —                      | —                          | **AdaKVScheduler**        | 1.1–1.3 %                | 1.9×          | 83             | 2.6           | 86           |
| —                      | —                          | **DuoAttentionScheduler** | 1.3–1.5 %                | 1.8×          | 88             | 2.6           | 86           |

---

#### 2️⃣ Double-Technique Experiments

##### (a) Router + Compressor

| Routing            | Compression         | Accuracy-Drop ↓ | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | ------------------- | ---------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | PyramidCompressor   | 3.2–3.7 %        | 2.8×          | 97        | 2.1      | 83       |
| TopKBalancedRouter | SVDCompressor       | 5.7–6.1 %        | 3.1×          | 98        | 1.8      | 78       |
| AdaptiveRouter     | QuantizedCompressor | 4.4–4.8 %        | 3.6×          | 96        | 1.7      | 80       |
| PiKVRouter         | PyramidKVCompressor | 3.6–4.1 %        | 3.4×          | 95        | 1.6      | 78       |
| EPLBRouter         | FastVCompressor     | 3.1–3.5 %        | 3.1×          | 94        | 1.7      | 80       |
| HierarchicalRouter | ChunkKVCompressor   | 5.9–6.4 %        | 3.2×          | 98        | 1.5      | 75       |

##### (b) Router + Scheduler

| Routing            | Scheduling              | Accuracy-Drop ↓ | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | ----------------------- | ---------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | H2OScheduler            | 1.5–1.8 %        | 1.9×          | 85        | 2.5      | 89       |
| TopKBalancedRouter | StreamingLLMScheduler   | 1.7–2.0 %        | 1.8×          | 84        | 2.6      | 85       |
| AdaptiveRouter     | QUESTScheduler          | 1.3–1.6 %        | 1.9×          | 83        | 2.4      | 87       |
| PiKVRouter         | FlexGenScheduler        | 1.7–1.9 %        | 1.7×          | 81        | 2.7      | 84       |
| EPLBRouter         | LRUScheduler            | 1.5–1.8 %        | 1.7×          | 89        | 2.5      | 85       |
| HierarchicalRouter | AdaKVScheduler          | 1.3–1.6 %        | 1.9×          | 83        | 2.6      | 86       |
| BaseRouter         | LRUPlusScheduler        | 1.3–1.6 %        | 1.8×          | 88        | 2.4      | 86       |
| TopKBalancedRouter | DuoAttentionScheduler   | 1.4–1.6 %        | 1.8×          | 87        | 2.5      | 86       |

##### (c) Compressor + Scheduler

| Compression            | Scheduling            | Accuracy-Drop ↓ | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ---------------------- | --------------------- | ---------------- | ------------- | --------- | -------- | -------- |
| LoRACompressor         | DuoAttentionScheduler | 7.8–8.3 %        | 4.1×          | 80        | 1.2      | 66       |
| LoRaPlusPlusCompressor | StreamingLLMScheduler | 7.1–7.5 %        | 4.2×          | 79        | 1.2      | 66       |
| PruningCompressor      | QUESTScheduler        | 4.9–5.3 %        | 3.5×          | 84        | 1.5      | 71       |
| DistillationCompressor | LRUScheduler          | 11.6–12.3 %      | 4.7×          | 76        | 1.1      | 58       |
| PiKVCompressor         | AdaKVScheduler        | 4.0–4.4 %        | 3.7×          | 82        | 1.5      | 75       |
| ChunkKV + SVD          | AdaKVScheduler        | 5.0–5.6 %        | 2.5–3.6×      | 89        | 1.4      | 70       |

---

#### 3️⃣ Triple-Technique Experiments

| Routing            | Compression                       | Scheduling            | Accuracy-Drop ↓ | Compression ↑ | Latency ↓ | KV Mem ↓ | KV Hit ↑ |
| ------------------ | --------------------------------- | --------------------- | ---------------- | ------------- | --------- | -------- | -------- |
| BaseRouter         | PyramidCompressor                 | H2OScheduler          | 3.9–4.3 %        | 2.9×          | 86        | 1.9      | 84       |
| TopKBalancedRouter | PyramidKV + FastV                 | AdaKVScheduler        | 1.6–1.9 %        | 2.4–3.3×      | 85        | 1.7      | 83       |
| TopKBalancedRouter | SVDCompressor                     | StreamingLLMScheduler | 6.4–6.9 %        | 3.3×          | 88        | 1.7      | 79       |
| AdaptiveRouter     | QuantizedCompressor               | QUESTScheduler        | 5.0–5.5 %        | 3.8×          | 85        | 1.5      | 81       |
| PiKVRouter         | LoRACompressor                    | FlexGenScheduler      | 8.3–8.7 %        | 4.2×          | 79        | 1.2      | 65       |
| EPLBRouter         | FastVCompressor                   | LRUPlusScheduler      | 4.1–4.5 %        | 3.2×          | 84        | 1.6      | 83       |
| HierarchicalRouter | PruningCompressor                 | AdaKVScheduler        | 5.5–5.9 %        | 3.6×          | 83        | 1.4      | 78       |
| AdaptiveRouter     | DistillationCompressor            | LRUScheduler          | 12.4–13.3 %      | 4.8×          | 75        | 1.0      | 58       |
| PiKVRouter         | LoRaPlusPlusCompressor            | DuoAttentionScheduler | 9.2–9.6 %        | 4.4×          | 78        | 1.1      | 64       |
| EPLBRouter         | PyramidKVCompressor               | StreamingLLMScheduler | 4.0–4.5 %        | 3.5×          | 86        | 1.5      | 81       |
| BaseRouter         | ChunkKVCompressor                 | LRUScheduler          | 6.3–6.8 %        | 3.4×          | 87        | 1.3      | 76       |
| TopKBalancedRouter | PiKVCompressor                    | QUESTScheduler        | 4.7–5.2 %        | 3.7×          | 84        | 1.4      | 80       |
| —                  | LoRA + Distillation               | —                     | 9.7–12.8 %       | 2.7–4.6×      | 74        | 1.1      | 59       |
