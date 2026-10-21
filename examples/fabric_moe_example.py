#!/usr/bin/env python3
"""Demo: FabricMoE -- fabric-aware expert routing for multi-FPGA clusters.

Runs three scenarios:
  1. Standalone FabricMoE training on a multi-FPGA mesh (HCR + DFM + duals).
  2. A fabric-graph swap: train on two different G_H and report the
     Jensen-Shannon divergence between the two learned routing distributions.
  3. FabricMoE fused with LAER + MoEBlaze + FSMoE via ``create_fused_moe_training``.
"""

import torch

from core.distributed import (
    FabricGraph,
    create_fabric_moe,
    create_fused_moe_training,
)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> float:
    """Jensen-Shannon divergence between two routing distributions."""
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).clamp_min(eps).log()).sum()
    kl_qm = (q * (q / m).clamp_min(eps).log()).sum()
    return float(0.5 * (kl_pm + kl_qm).item())


def run_standalone(steps: int = 40):
    print("=== FabricMoE standalone (HCR + DFM + adaptive duals) ===")
    fabric = FabricGraph.multi_fpga_mesh(num_experts=16, num_regions=4, num_devices=4)
    model = create_fabric_moe(
        hidden_size=64, num_experts=16, top_k=2,
        num_regions=4, num_devices=4, num_layers=2,
        fabric=fabric, relayout_interval=8,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps):
        x = torch.randn(4, 16, 64)
        # Bias some dims to stress routing skew across the fabric
        x[..., :8] = x[..., :8] + (1.5 if step % 3 == 0 else -0.5)
        m = model.train_step(x, optimizer=opt)
        if step % 10 == 0 or step == steps - 1:
            l0 = m["layers"][0]
            mu = l0["duals"]["mu"]
            print(
                f"step={step:03d} loss={m['loss']:.4f} aux={m['aux_loss']:.4f} "
                f"mu_lat={mu['latency']:.2f} mu_bw={mu['traffic']:.2f} "
                f"mu_cong={mu['congestion']:.2f} "
                f"dfm_mape_lat={l0['dfm']['latency_mape']:.3f}"
            )
    print("standalone OK\n")


def run_fabric_swap(steps: int = 30):
    print("=== Fabric-graph swap ===")
    torch.manual_seed(7)
    fab_a = FabricGraph.multi_fpga_mesh(num_experts=16, num_regions=4, num_devices=4)
    fab_b = FabricGraph.hierarchical_ring(num_experts=16, num_devices=8)

    def train_and_route(fab):
        torch.manual_seed(7)
        model = create_fabric_moe(
            hidden_size=64, num_experts=16, top_k=2,
            num_regions=fab.num_regions, num_devices=fab.num_devices,
            num_layers=1, fabric=fab,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(steps):
            x = torch.randn(4, 16, 64)
            model.train_step(x, optimizer=opt)
        model.eval()
        layer = model.layers[0]
        layer._ensure_fabric_embeddings()
        c_t = layer.telemetry_gru(layer._telemetry_vector())
        out = layer.router(torch.randn(32, 64), layer._z_experts,
                           layer._z_regions, c_t, top_k=2, training=False)
        return out["routing_probs"].mean(dim=0)

    p_a = train_and_route(fab_a)
    p_b = train_and_route(fab_b)
    js = _js_divergence(p_a, p_b)
    print(f"JS(routing_A, routing_B) = {js:.3f}")
    print("fabric-swap OK\n")


def run_fused(steps: int = 20):
    print("=== FabricMoE fused with LAER + MoEBlaze + FSMoE ===")
    model = create_fused_moe_training(
        hidden_size=64, num_experts=8, top_k=2, num_layers=2,
        enable_laer=True, enable_moeblaze=True, enable_fsmoe=True,
        enable_fabric=True, fabric_num_devices=2, relayout_interval=8,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps):
        x = torch.randn(4, 16, 64)
        m = model.train_step(x, optimizer=opt)
        if step % 5 == 0 or step == steps - 1:
            l0 = m["layers"][0]
            fab = l0.get("fabric", {})
            print(
                f"step={step:03d} loss={m['loss']:.4f} lm={m['lm_loss']:.4f} "
                f"fab_aux={m['fabric_aux_loss']:.4f} "
                f"relayouts={l0.get('laer_relayout', {}).get('relayout_events', 0)} "
                f"fsmoe_adapt={l0.get('fsmoe', {}).get('adaptations', 0)} "
                f"fab_cuts={fab.get('num_cuts', 0)}"
            )
    print("fused OK")


if __name__ == "__main__":
    run_standalone()
    run_fabric_swap()
    run_fused()
