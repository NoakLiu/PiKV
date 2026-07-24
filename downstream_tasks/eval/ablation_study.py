#!/usr/bin/env python3
"""
PiKV systematic ablation harness.

Isolates contributions of:
  - routing
  - compression
  - scheduling
  - expert-sharded KV storage

Reports latency, KV-memory proxy, compression ratio, KV-hit rate, and
load imbalance, plus synergy of the full stack vs summed single-module gains.

See downstream_tasks/EXPERIMENTAL_PROTOCOL.md for GPU/batch/fairness settings.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight module stubs that mirror PiKV stages without requiring GPUs/LLMs
# ---------------------------------------------------------------------------


class IdentityRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        b, s, _ = x.shape
        idx = torch.zeros(b, s, self.top_k, dtype=torch.long, device=x.device)
        w = torch.ones(b, s, self.top_k, device=x.device) / self.top_k
        return idx, w, 0.0


class LearnableRouter(nn.Module):
    """Cache-aware-ish router: logits − λ·miss_penalty + load-balance term."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        cache_penalty: float = 0.5,
        lb_coeff: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.cache_penalty = cache_penalty
        self.lb_coeff = lb_coeff
        self.router = nn.Linear(hidden_size, num_experts)
        self.register_buffer("miss_count", torch.zeros(num_experts))
        self.register_buffer("load", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        logits = self.router(x)
        # Penalize historically cold experts (cache-aware)
        miss = self.miss_count / (self.miss_count.sum() + 1e-6)
        logits = logits - self.cache_penalty * miss.view(1, 1, -1)
        # Load-balance: discourage overloaded experts
        load = self.load / (self.load.mean() + 1e-6)
        logits = logits - self.lb_coeff * load.view(1, 1, -1)
        probs = F.softmax(logits, dim=-1)
        w, idx = torch.topk(probs, k=self.top_k, dim=-1)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
        with torch.no_grad():
            for e in range(self.num_experts):
                self.load[e] = 0.9 * self.load[e] + 0.1 * (idx == e).float().mean()
        # aux: encourage uniform load
        mean_p = probs.mean(dim=(0, 1))
        aux = float((mean_p * (mean_p * self.num_experts + 1e-9).log()).sum().item())
        return idx, w, aux


class IdentityCompressor(nn.Module):
    def forward(self, keys: torch.Tensor, values: torch.Tensor):
        return keys, values, 1.0


class PyramidLikeCompressor(nn.Module):
    """Keep top-ratio channels by L2 energy (proxy for pyramid/importance keep)."""

    def __init__(self, ratio: float = 0.5):
        super().__init__()
        self.ratio = max(0.05, min(1.0, ratio))

    def forward(self, keys: torch.Tensor, values: torch.Tensor):
        d = keys.size(-1)
        keep = max(1, int(d * self.ratio))
        score = keys.pow(2).mean(dim=(0, 1)) if keys.dim() == 3 else keys.pow(2).mean(dim=0)
        _, idx = torch.topk(score, k=keep)
        return keys[..., idx], values[..., idx], d / keep


class IdentityScheduler:
    def __init__(self, cache_size: int, hidden_size: int):
        self.cache_size = cache_size
        self.hidden_size = hidden_size
        self.keys = None
        self.values = None
        self.hits = 0
        self.misses = 0
        self._seen = set()

    def update(self, keys: torch.Tensor, values: torch.Tensor, token_ids: Optional[List[int]] = None):
        flat_k = keys.reshape(-1, keys.size(-1))
        flat_v = values.reshape(-1, values.size(-1))
        n = flat_k.size(0)
        ids = token_ids or list(range(n))
        for i in range(min(n, self.cache_size)):
            tid = ids[i % len(ids)]
            if tid in self._seen:
                self.hits += 1
            else:
                self.misses += 1
                self._seen.add(tid)
        keep = min(self.cache_size, flat_k.size(0))
        self.keys = flat_k[:keep].clone()
        self.values = flat_v[:keep].clone()

    @property
    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return int(self.keys.nelement() * self.keys.element_size()
                   + self.values.nelement() * self.values.element_size())


class H2OLikeScheduler(IdentityScheduler):
    """Keep high-attention tokens (importance = ||v||); reuse ids → hits."""

    def update(self, keys: torch.Tensor, values: torch.Tensor, token_ids: Optional[List[int]] = None):
        flat_k = keys.reshape(-1, keys.size(-1))
        flat_v = values.reshape(-1, values.size(-1))
        score = flat_v.norm(dim=-1)
        keep = min(self.cache_size, flat_k.size(0))
        _, idx = torch.topk(score, k=keep)
        ids = token_ids or list(range(flat_k.size(0)))
        kept_ids = [ids[int(i) % len(ids)] for i in idx.tolist()]
        for tid in kept_ids:
            if tid in self._seen:
                self.hits += 1
            else:
                self.misses += 1
                self._seen.add(tid)
        self.keys = flat_k[idx].clone()
        self.values = flat_v[idx].clone()


class ExpertShardStore:
    """Hash-XOR shard KV across virtual experts/GPUs (s = t ⊕ e)."""

    def __init__(self, num_shards: int, enabled: bool = True):
        self.num_shards = max(1, num_shards)
        self.enabled = enabled
        self.shards: Dict[int, List[torch.Tensor]] = {i: [] for i in range(self.num_shards)}
        self.cross_shard_fetches = 0

    def place(self, token_id: int, expert_id: int, payload: torch.Tensor):
        if not self.enabled:
            self.shards[0].append(payload)
            return 0
        sid = (token_id ^ expert_id) % self.num_shards
        self.shards[sid].append(payload)
        return sid

    def fetch_sequence(self, token_ids: List[int], expert_id: int) -> int:
        """Count how many adjacent tokens land on different shards (locality cost)."""
        if not self.enabled or len(token_ids) < 2:
            return 0
        prev = (token_ids[0] ^ expert_id) % self.num_shards
        switches = 0
        for t in token_ids[1:]:
            sid = (t ^ expert_id) % self.num_shards
            if sid != prev:
                switches += 1
                self.cross_shard_fetches += 1
            prev = sid
        return switches

    def memory_bytes(self) -> int:
        total = 0
        for items in self.shards.values():
            for t in items:
                total += int(t.nelement() * t.element_size())
        return total


@dataclass
class AblationConfig:
    hidden_size: int = 512
    num_experts: int = 8
    top_k: int = 2
    batch_size: int = 1
    seq_len: int = 512          # prefill token budget
    decode_len: int = 128
    concurrent_requests: int = 8
    cache_size: int = 1024
    compression_ratio: float = 0.5
    num_shards: int = 4
    device: str = "cpu"
    runs: int = 3
    warmup: int = 1
    seed: int = 42


@dataclass
class RunMetrics:
    name: str
    latency_ms: float
    kv_mem_mb: float
    compression: float
    kv_hit: float
    load_imbalance: float
    cross_shard_switches: float
    extras: Dict[str, Any] = field(default_factory=dict)


def _make_stack(
    cfg: AblationConfig,
    routing: str = "off",
    compression: str = "off",
    scheduling: str = "off",
    sharding: bool = False,
):
    device = torch.device(cfg.device)
    if routing == "off":
        router = IdentityRouter(cfg.hidden_size, cfg.num_experts, cfg.top_k)
    else:
        router = LearnableRouter(cfg.hidden_size, cfg.num_experts, cfg.top_k)
    if compression == "off":
        compressor = IdentityCompressor()
    else:
        compressor = PyramidLikeCompressor(ratio=cfg.compression_ratio)
    if scheduling == "off":
        scheduler = IdentityScheduler(cfg.cache_size, cfg.hidden_size)
    else:
        scheduler = H2OLikeScheduler(cfg.cache_size, cfg.hidden_size)
    shard = ExpertShardStore(cfg.num_shards, enabled=sharding)
    return router.to(device), compressor.to(device) if isinstance(compressor, nn.Module) else compressor, scheduler, shard


def _one_request(
    cfg: AblationConfig,
    router,
    compressor,
    scheduler,
    shard: ExpertShardStore,
    req_id: int,
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    # Prefill
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, device=device)
    idx, w, aux = router(x)
    # Expert outputs proxy
    keys = x
    values = x
    keys_c, values_c, ratio = compressor(keys, values)
    prefill_ids = list(range(req_id * cfg.seq_len, req_id * cfg.seq_len + cfg.seq_len))
    scheduler.update(keys_c, values_c, token_ids=prefill_ids)

    # Sharding placement + locality cost along the sequence
    token_ids = prefill_ids
    expert = int(idx[0, 0, 0].item()) if torch.is_tensor(idx) else 0
    for t in token_ids[:: max(1, cfg.seq_len // 32)]:
        shard.place(t, expert, values_c[0, t % values_c.size(1)].detach().cpu())
    switches = shard.fetch_sequence(token_ids, expert)

    # Decode: re-touch recent prefill ids to exercise cache hits, then new tokens
    for d in range(cfg.decode_len):
        step = torch.randn(cfg.batch_size, 1, cfg.hidden_size, device=device)
        idx, w, aux = router(step)
        k_c, v_c, _ = compressor(step, step)
        # alternate reuse vs new id
        tid = prefill_ids[d % len(prefill_ids)] if d % 2 == 0 else (10_000_000 + req_id * cfg.decode_len + d)
        scheduler.update(k_c, v_c, token_ids=[tid])

    load = getattr(router, "load", None)
    if load is not None and load.numel() > 0:
        imbalance = float(load.var().item() / (load.mean().item() + 1e-9))
    else:
        imbalance = 0.0

    return {
        "compression": float(ratio),
        "kv_hit": float(scheduler.hit_rate),
        "kv_mem_mb": scheduler.nbytes / (1024 * 1024) + shard.memory_bytes() / (1024 * 1024),
        "load_imbalance": imbalance,
        "cross_shard_switches": float(switches),
        "aux": float(aux),
    }


def run_variant(
    name: str,
    cfg: AblationConfig,
    routing: str = "off",
    compression: str = "off",
    scheduling: str = "off",
    sharding: bool = False,
) -> RunMetrics:
    latencies = []
    metrics_acc = []
    for run in range(cfg.runs + cfg.warmup):
        torch.manual_seed(cfg.seed + run)
        router, compressor, scheduler, shard = _make_stack(
            cfg, routing, compression, scheduling, sharding
        )
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        last = {}
        for req in range(cfg.concurrent_requests):
            last = _one_request(cfg, router, compressor, scheduler, shard, req)
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / cfg.concurrent_requests
        if run >= cfg.warmup:
            latencies.append(elapsed_ms)
            metrics_acc.append(last)

    def mean(key):
        return sum(m[key] for m in metrics_acc) / len(metrics_acc)

    return RunMetrics(
        name=name,
        latency_ms=sum(latencies) / len(latencies),
        kv_mem_mb=mean("kv_mem_mb"),
        compression=mean("compression"),
        kv_hit=mean("kv_hit") * 100.0,
        load_imbalance=mean("load_imbalance"),
        cross_shard_switches=mean("cross_shard_switches"),
        extras={
            "latency_std_ms": (
                sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)
            )
            ** 0.5,
            "routing": routing,
            "compression": compression,
            "scheduling": scheduling,
            "sharding": sharding,
            "protocol": asdict(cfg),
        },
    )


def synergy_report(results: Dict[str, RunMetrics]) -> Dict[str, Any]:
    """Compare full stack vs summed single-module gains and vs existing-method combo."""
    base = results["baseline"].latency_ms

    def gain(name: str) -> float:
        return base - results[name].latency_ms

    def mem_save(name: str) -> float:
        return results["baseline"].kv_mem_mb - results[name].kv_mem_mb

    singles = ["routing_only", "compression_only", "scheduling_only", "sharding_only"]
    present = [s for s in singles if s in results]
    additive = sum(gain(s) for s in present)
    out: Dict[str, Any] = {
        "baseline_latency_ms": base,
        "single_module_gains_ms": {s: gain(s) for s in present},
        "single_module_mem_save_mb": {s: mem_save(s) for s in present},
        "additive_expectation_ms": additive,
    }
    if "full_stack" in results:
        full_gain = gain("full_stack")
        out["full_stack_gain_ms"] = full_gain
        out["synergy_ms"] = full_gain - additive
        out["interpretation"] = (
            "synergy_ms > 0: full stack beats naive sum of singles; "
            "< 0: interference / diminishing returns."
        )
    # PiKV incremental gain over a direct combination of existing methods
    if "existing_methods_combo" in results and "pikv_stack" in results:
        exist = results["existing_methods_combo"]
        pikv = results["pikv_stack"]
        out["vs_existing_combo"] = {
            "existing": "TopK/H2O/Pyramid without cache-aware routing or expert sharding",
            "pikv": "cache-aware routing + Pyramid + H2O-like + expert sharding",
            "delta_latency_ms": exist.latency_ms - pikv.latency_ms,
            "delta_kv_mem_mb": exist.kv_mem_mb - pikv.kv_mem_mb,
            "delta_kv_hit_pp": pikv.kv_hit - exist.kv_hit,
            "delta_xshard": pikv.cross_shard_switches - exist.cross_shard_switches,
        }
    return out


def build_preset(preset: str, cfg: AblationConfig) -> Dict[str, RunMetrics]:
    results: Dict[str, RunMetrics] = {}
    results["baseline"] = run_variant("baseline", cfg)

    if preset in ("routing", "factor", "combined"):
        results["routing_only"] = run_variant("routing_only", cfg, routing="on")
    if preset in ("compression", "factor", "combined"):
        results["compression_only"] = run_variant(
            "compression_only", cfg, compression="on"
        )
    if preset in ("scheduling", "factor", "combined"):
        results["scheduling_only"] = run_variant(
            "scheduling_only", cfg, scheduling="on"
        )
    if preset in ("sharding", "factor", "combined"):
        results["sharding_only"] = run_variant("sharding_only", cfg, sharding=True)

    if preset in ("combined", "factor"):
        results["routing+compression"] = run_variant(
            "routing+compression", cfg, routing="on", compression="on"
        )
        results["routing+scheduling"] = run_variant(
            "routing+scheduling", cfg, routing="on", scheduling="on"
        )
        results["compression+scheduling"] = run_variant(
            "compression+scheduling", cfg, compression="on", scheduling="on"
        )
        # Existing-methods combo: compression+scheduling only (no PiKV routing/shard)
        results["existing_methods_combo"] = run_variant(
            "existing_methods_combo",
            cfg,
            routing="off",
            compression="on",
            scheduling="on",
            sharding=False,
        )
        # PiKV stack: cache-aware routing + compression + scheduling + sharding
        results["pikv_stack"] = run_variant(
            "pikv_stack",
            cfg,
            routing="on",
            compression="on",
            scheduling="on",
            sharding=True,
        )
        results["full_stack"] = results["pikv_stack"]
    return results


def print_markdown_table(results: Dict[str, RunMetrics]) -> None:
    print("\n### Isolation table (markdown)\n")
    print("| Variant | Latency (ms) | KV Mem (MB) | Comp ↑ | KV Hit ↑ (%) | Load imb. | Cross-shard |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, m in results.items():
        if name == "full_stack":
            continue  # alias of pikv_stack
        print(
            f"| {name} | {m.latency_ms:.2f} | {m.kv_mem_mb:.2f} | "
            f"{m.compression:.2f}× | {m.kv_hit:.1f} | {m.load_imbalance:.3f} | "
            f"{m.cross_shard_switches:.1f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="PiKV module ablation study")
    parser.add_argument(
        "--preset",
        default="factor",
        choices=["baseline", "routing", "compression", "scheduling", "sharding", "combined", "factor"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = "cpu"

    cfg = AblationConfig(
        device=device,
        runs=args.runs,
        seq_len=args.seq_len,
        decode_len=args.decode_len,
        concurrent_requests=args.concurrency,
    )

    print("PiKV Ablation Study")
    print("=" * 60)
    print(
        f"device={cfg.device}  prefill={cfg.seq_len}  decode={cfg.decode_len}  "
        f"concurrency={cfg.concurrent_requests}  runs={cfg.runs}"
    )
    print("See EXPERIMENTAL_PROTOCOL.md for fairness controls.\n")

    results = build_preset(args.preset, cfg)
    for name, m in results.items():
        if name == "full_stack":
            continue
        print(
            f"{name:28s}  lat={m.latency_ms:8.2f}ms  "
            f"mem={m.kv_mem_mb:7.2f}MB  comp={m.compression:4.2f}x  "
            f"hit={m.kv_hit:5.1f}%  imb={m.load_imbalance:6.3f}  "
            f"xshard={m.cross_shard_switches:7.1f}"
        )

    print_markdown_table(results)
    syn = synergy_report(results)
    print("\n--- Isolation / synergy / vs existing combo ---")
    print(json.dumps(syn, indent=2))

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "results"
    )
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"ablation_{stamp}.json")
    payload = {
        "protocol": "EXPERIMENTAL_PROTOCOL.md",
        "preset": args.preset,
        "config": asdict(cfg),
        "results": {k: asdict(v) for k, v in results.items()},
        "synergy": syn,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        ),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
