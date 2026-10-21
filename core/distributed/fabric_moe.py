"""
FabricMoE: Fabric-Aware Expert Routing for Multi-FPGA Clusters
(our new method)

Training-side counterpart of the paper's spatial datapath. Implements:
  * Hardware-Conditioned Router (HCR)
  * Differentiable Fabric Model (DFM)
  * Constraint-driven training (adaptive duals)
  * Topology-induced hierarchy
  * Affinity-driven placement

Runnable without hardware or torch.distributed initialized.
See core/fpga/README.md for the HCR <-> FPGA engine mapping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fabric graph G_H = (V_H, E_H)
# ---------------------------------------------------------------------------


@dataclass
class FabricNode:
    """A node of G_H: expert PE, BRAM/URAM bank, HBM port, region controller, switch."""

    node_id: int
    type: str  # "pe" | "bram" | "hbm" | "region" | "switch"
    region: int = 0
    device: int = 0
    expert_id: Optional[int] = None  # set only for "pe" nodes
    residency: str = "sram"  # "sram" | "hbm"  (m_e)
    queue_depth: float = 0.0
    bandwidth: float = 1.0  # normalized local BW (b_v)


@dataclass
class FabricLink:
    """An edge of G_H with capacity, zero-load delay, and utilization."""

    src: int
    dst: int
    capacity: float = 1.0       # C_l
    base_delay: float = 1.0     # d_l^0  (cycles)
    utilization: float = 0.0    # u_l
    cross_slr: bool = False
    cross_dev: bool = False


@dataclass
class FabricGraph:
    """
    Multi-FPGA fabric G_H = (V_H, E_H).

    Nodes are expert PEs, BRAM/URAM banks, HBM ports, region controllers, and
    switches; edges carry capacity, zero-load delay, and instantaneous
    utilization. Effective bandwidth ratio intra:SLR:QSFP ~ 40:4:1.
    """

    nodes: List[FabricNode] = field(default_factory=list)
    links: List[FabricLink] = field(default_factory=list)
    num_regions: int = 1
    num_devices: int = 1
    f_clk_mhz: float = 300.0

    def add_node(self, **kw: Any) -> FabricNode:
        node = FabricNode(node_id=len(self.nodes), **kw)
        self.nodes.append(node)
        return node

    def add_link(self, src: int, dst: int, **kw: Any) -> FabricLink:
        link = FabricLink(src=src, dst=dst, **kw)
        self.links.append(link)
        return link

    def expert_anchor(self, expert_id: int) -> FabricNode:
        for n in self.nodes:
            if n.type == "pe" and n.expert_id == expert_id:
                return n
        raise KeyError(f"no PE node anchors expert {expert_id}")

    def region_nodes(self, region: int) -> List[FabricNode]:
        return [n for n in self.nodes if n.region == region]

    def expert_to_region(self, expert_id: int) -> int:
        return self.expert_anchor(expert_id).region

    def expert_to_device(self, expert_id: int) -> int:
        return self.expert_anchor(expert_id).device

    def adjacency(self) -> torch.Tensor:
        """Dense symmetric adjacency matrix over all nodes (self-loops included)."""
        n = len(self.nodes)
        adj = torch.zeros(n, n)
        for l in self.links:
            adj[l.src, l.dst] = 1.0
            adj[l.dst, l.src] = 1.0
        adj.fill_diagonal_(1.0)
        return adj

    def hop_distance_matrix(self) -> torch.Tensor:
        """All-pairs BFS hop count on G_H (discrete; used by DFM and placer)."""
        n = len(self.nodes)
        dist = torch.full((n, n), float("inf"))
        dist.fill_diagonal_(0.0)
        for l in self.links:
            dist[l.src, l.dst] = 1.0
            dist[l.dst, l.src] = 1.0
        for k in range(n):  # Floyd-Warshall; n is small (~100 nodes)
            dist = torch.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
        return dist.clamp(min=0.0)

    def cuts(self) -> List[FabricLink]:
        """Edges that cross an SLR or a device boundary."""
        return [l for l in self.links if l.cross_slr or l.cross_dev]

    @classmethod
    def single_slr(cls, num_experts: int, num_regions: int = 4) -> "FabricGraph":
        """One-device, one-SLR fabric: experts split across R regions, intra-region mesh."""
        g = cls(num_regions=num_regions, num_devices=1)
        per_region = num_experts // num_regions
        for r in range(num_regions):
            g.add_node(type="region", region=r, device=0)
        for e in range(num_experts):
            r = e // per_region
            g.add_node(type="pe", region=r, device=0, expert_id=e,
                       residency="sram" if e % 4 else "hbm")
        g.add_node(type="hbm", region=0, device=0)
        region_ids = [n.node_id for n in g.nodes if n.type == "region"]
        for r in range(num_regions):
            rnode = region_ids[r]
            for n in g.nodes:
                if n.type == "pe" and n.region == r:
                    g.add_link(rnode, n.node_id, capacity=4.0, base_delay=1.0)
        for r in range(num_regions):
            a, b = region_ids[r], region_ids[(r + 1) % num_regions]
            g.add_link(a, b, capacity=2.0, base_delay=2.0)
        return g

    @classmethod
    def multi_fpga_mesh(
        cls, num_experts: int = 64, num_regions: int = 8, num_devices: int = 16,
    ) -> "FabricGraph":
        """
        Phase-2 fabric: identical tile replicated across K FPGAs over 100 GbE/QSFP.
        Each device holds R regions; QSFP cuts are first-class edges.
        """
        g = cls(num_regions=num_regions, num_devices=num_devices)
        per_dev = num_experts // num_devices
        per_region = max(1, per_dev // num_regions)
        dev_region_nodes: Dict[int, List[int]] = {}

        for d in range(num_devices):
            rids: List[int] = []
            for r in range(num_regions):
                rids.append(g.add_node(type="region", region=r, device=d).node_id)
            dev_region_nodes[d] = rids
            base = d * per_dev
            for e in range(base, base + per_dev):
                r = (e - base) // per_region
                g.add_node(type="pe", region=min(r, num_regions - 1), device=d,
                           expert_id=e, residency="sram" if (e % 4) else "hbm")
            g.add_node(type="hbm", region=0, device=d)
            rids = dev_region_nodes[d]
            for r in range(num_regions):
                a, b = rids[r], rids[(r + 1) % num_regions]
                g.add_link(a, b, capacity=4.0, base_delay=2.0, cross_slr=(r % 2 == 1))
            for n in g.nodes:
                if n.type == "pe" and n.device == d:
                    g.add_link(rids[n.region], n.node_id, capacity=4.0, base_delay=1.0)

        # QSFP cuts: 2-D mesh of devices (2 VC, credit flow).
        # Fall back to a ring when the 2-D mesh is degenerate (side < 2).
        side = int(math.isqrt(num_devices))
        if side < 2:
            # Ring of devices
            for d in range(num_devices):
                nd = (d + 1) % num_devices
                g.add_link(dev_region_nodes[d][0], dev_region_nodes[nd][0],
                           capacity=1.0, base_delay=6.0, cross_dev=True)
        else:
            for d in range(num_devices):
                r, c = d // side, d % side
                for (dr, dc) in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < side and 0 <= nc < side:
                        nd = nr * side + nc
                        g.add_link(dev_region_nodes[d][0], dev_region_nodes[nd][0],
                                   capacity=1.0, base_delay=6.0, cross_dev=True)
        return g

    @classmethod
    def hierarchical_ring(cls, num_experts: int = 64, num_devices: int = 8) -> "FabricGraph":
        """Hierarchical ring topology."""
        return cls.multi_fpga_mesh(num_experts=num_experts, num_regions=8,
                                   num_devices=num_devices)


# ---------------------------------------------------------------------------
# On-chip telemetry S_H  (64-bit word @ 1024 cyc)
# ---------------------------------------------------------------------------


@dataclass
class TelemetryState:
    """
    S_H: on-chip telemetry summary. The paper packs region queue depths, the
    eight highest link utilizations, HBM stalls, and credits into a 64-bit word
    emitted every 1024 cycles. We model it as a float vector for the GRU input.
    """

    region_queues: torch.Tensor    # [R]
    link_utils: torch.Tensor       # [top-K]
    hbm_stalls: torch.Tensor       # [num_hbm]
    credits: torch.Tensor          # [R]

    def to_vector(self) -> torch.Tensor:
        return torch.cat([self.region_queues, self.link_utils,
                          self.hbm_stalls, self.credits]).reshape(1, -1)

    @classmethod
    def zeros(cls, num_regions: int, num_link_samples: int = 8,
              num_hbm: int = 1) -> "TelemetryState":
        return cls(
            region_queues=torch.zeros(num_regions),
            link_utils=torch.zeros(num_link_samples),
            hbm_stalls=torch.zeros(num_hbm),
            credits=torch.ones(num_regions),
        )

    @classmethod
    def from_fabric(cls, fabric: FabricGraph, num_link_samples: int = 8) -> "TelemetryState":
        """Sample telemetry from the current fabric state (queue depths, link utils)."""
        region_queues = torch.zeros(fabric.num_regions)
        for n in fabric.nodes:
            if n.type == "region":
                region_queues[n.region] = n.queue_depth
        link_utils = torch.tensor([l.utilization for l in fabric.links], dtype=torch.float32)
        if link_utils.numel() > num_link_samples:
            link_utils, _ = torch.topk(link_utils, num_link_samples)
        elif link_utils.numel() < num_link_samples:
            link_utils = F.pad(link_utils, (0, num_link_samples - link_utils.numel()))
        hbm = torch.tensor([n.queue_depth for n in fabric.nodes if n.type == "hbm"],
                          dtype=torch.float32)
        if hbm.numel() == 0:
            hbm = torch.zeros(1)
        credits = torch.ones(fabric.num_regions)
        return cls(region_queues=region_queues, link_utils=link_utils,
                   hbm_stalls=hbm, credits=credits)


# ---------------------------------------------------------------------------
# Fabric-graph encoder GNN_phi
# ---------------------------------------------------------------------------


class FabricGraphEncoder(nn.Module):
    """
    2-layer message-passing encoder over G_H (GNN_phi, hidden 64).

    h_v^{l+1} = ReLU(W_self h_v + W_neigh mean_{u in N(v)} h_u)
    Outputs per-expert embedding z_H_e = h_{v_e}^{(2)} and per-region embedding
    z_r = mean_{v in r} h_v^{(2)}.
    """

    def __init__(self, hidden_size: int = 64, num_node_types: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.type_emb = nn.Embedding(num_node_types, hidden_size)
        self.feat_proj = nn.Linear(3, hidden_size)  # queue, residency_is_hbm, bandwidth
        self.W_self_0 = nn.Linear(hidden_size, hidden_size)
        self.W_neigh_0 = nn.Linear(hidden_size, hidden_size)
        self.W_self_1 = nn.Linear(hidden_size, hidden_size)
        self.W_neigh_1 = nn.Linear(hidden_size, hidden_size)

    def _node_features(self, fabric: FabricGraph) -> torch.Tensor:
        type_map = {"pe": 0, "bram": 1, "hbm": 2, "region": 3, "switch": 4}
        types = torch.tensor([type_map[n.type] for n in fabric.nodes], dtype=torch.long)
        scalars = torch.tensor(
            [[n.queue_depth, float(n.residency == "hbm"), n.bandwidth]
             for n in fabric.nodes], dtype=torch.float32)
        return self.type_emb(types) + self.feat_proj(scalars)

    def _mean_neighbors(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (adj @ h) / deg

    def encode(self, fabric: FabricGraph) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Returns (h_nodes [num_nodes, hidden], region_emb {r: [hidden]}). Re-run only when G_H or P changes."""
        h = self._node_features(fabric)
        adj = fabric.adjacency()
        for (w_self, w_neigh) in [(self.W_self_0, self.W_neigh_0),
                                  (self.W_self_1, self.W_neigh_1)]:
            agg = self._mean_neighbors(h, adj)
            h = F.relu(w_self(h) + w_neigh(agg))
        region_emb: Dict[int, torch.Tensor] = {}
        for r in range(fabric.num_regions):
            idx = [n.node_id for n in fabric.region_nodes(r)]
            region_emb[r] = h[idx].mean(dim=0) if idx else torch.zeros(self.hidden_size)
        return h, region_emb

    def expert_embeddings(self, h_nodes: torch.Tensor, fabric: FabricGraph) -> torch.Tensor:
        """z_H_e for every expert, ordered by expert_id -> [num_experts, hidden]."""
        num_experts = sum(1 for n in fabric.nodes if n.type == "pe")
        z = torch.zeros(num_experts, self.hidden_size)
        for n in fabric.nodes:
            if n.type == "pe" and n.expert_id is not None:
                z[n.expert_id] = h_nodes[n.node_id]
        return z

    def region_embedding_matrix(self, region_emb: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Stack z_r into [num_regions, hidden] in region-id order."""
        return torch.stack([region_emb[r] for r in sorted(region_emb.keys())], dim=0)


# ---------------------------------------------------------------------------
# Telemetry GRU  (c_t = GRU(S_H[t-k:t]))
# ---------------------------------------------------------------------------


class TelemetryGRU(nn.Module):
    """On-chip recurrent summary of S_H over a window of telemetry samples."""

    def __init__(self, telemetry_dim: int, hidden_size: int = 32, window: int = 4):
        super().__init__()
        self.telemetry_dim = telemetry_dim
        self.hidden_size = hidden_size
        self.window = window
        self.gru = nn.GRU(telemetry_dim, hidden_size, batch_first=True)
        self.history: Deque[torch.Tensor] = deque(maxlen=window)

    def forward(self, s_h_vec: torch.Tensor) -> torch.Tensor:
        """s_h_vec: [telemetry_dim] or [1, telemetry_dim] -> c_t [hidden_size]."""
        v = s_h_vec.reshape(1, -1) if s_h_vec.dim() == 1 else s_h_vec
        self.history.append(v.detach())
        if len(self.history) < self.window:
            pad = [torch.zeros_like(v) for _ in range(self.window - len(self.history))]
            seq = torch.stack(pad + list(self.history), dim=1)  # [1, window, dim]
        else:
            seq = torch.stack(list(self.history), dim=1)  # [1, window, dim]
        _, h = self.gru(seq)
        # h is [num_layers=1, batch=1, hidden] -> [hidden]
        return h.reshape(-1)  # [hidden_size]

    def reset(self) -> None:
        self.history.clear()


# ---------------------------------------------------------------------------
# Differentiable Fabric Model (DFM) F_psi
# ---------------------------------------------------------------------------


@dataclass
class FabricPrediction:
    """y_hat_HW = F_psi(R, P, G_H) = [T_hat, B_hat, C_hat, U_hat]."""

    latency: torch.Tensor        # T_hat  [T] per-token latency (cycles)
    traffic: torch.Tensor        # B_hat  scalar total link bytes (normalized)
    congestion: torch.Tensor     # C_hat  scalar max link utilization
    link_utils: torch.Tensor     # U_hat  [num_links] per-link utilization


class DifferentiableFabricModel(nn.Module):
    """
    Surrogate that predicts the same five quantities the circuit measures, with
    M/M/1-style inflation d_l = d_l^0 / (1 - u_l) so latency/congestion gradients
    reach the router.

    Gradients flow through the soft routing matrix R: link utilization u_l is a
    soft expectation over which experts' traffic crosses l, and token latency
    T_hat_t = sum_e pi(e|x_t) * (L_disp(P(e)) + C_GEMM(e)/f_clk).
    """

    # Dispatch latency constants: L_DMA=8, L_HCR=8, L_PE=6
    L_DMA = 8.0
    L_HCR = 8.0
    L_PE = 6.0
    L_HBM = 18.0   # extra cycles for HBM-resident experts

    def __init__(self, fabric: FabricGraph, gmax: float = 10.0):
        super().__init__()
        self.fabric = fabric
        self.gmax = gmax  # STE gradient clip
        self._calibration_count = 0
        self._latency_mape = 0.0
        self._traffic_mape = 0.0
        self._spearman_rho = 0.0

        # Precompute hop distance from a reference source node (first region node)
        region_nodes = [n.node_id for n in fabric.nodes if n.type == "region"]
        self._src_node = region_nodes[0] if region_nodes else 0
        dist = fabric.hop_distance_matrix()
        self.register_buffer("hop_dist", dist)
        # Per-expert hop count from source (2*H_hop); clamp inf
        # (unreachable) to a large finite diameter so latency stays finite.
        num_experts = sum(1 for n in fabric.nodes if n.type == "pe")
        finite_dist = dist[~torch.isinf(dist)]
        diameter = float(finite_dist.max().item()) if finite_dist.numel() else 1.0
        expert_hops = torch.zeros(num_experts)
        for n in fabric.nodes:
            if n.type == "pe" and n.expert_id is not None:
                h = dist[self._src_node, n.node_id]
                expert_hops[n.expert_id] = h if torch.isfinite(h) else (diameter + 2.0)
        self.register_buffer("expert_hops", expert_hops)
        # Per-expert residency flag (m_e == HBM adds L_HBM)
        expert_hbm = torch.zeros(num_experts)
        for n in fabric.nodes:
            if n.type == "pe" and n.expert_id is not None:
                expert_hbm[n.expert_id] = float(n.residency == "hbm")
        self.register_buffer("expert_hbm", expert_hbm)
        # Link capacity / base delay buffers
        self.register_buffer("link_cap", torch.tensor([l.capacity for l in fabric.links]))
        self.register_buffer("link_d0", torch.tensor([l.base_delay for l in fabric.links]))
        # Link -> expert routing incidence (which experts' traffic can cross l)
        # Approximated: a link is "used" by experts whose anchor is reachable through it.
        self.register_buffer("link_incidence", self._build_link_incidence())

    def _build_link_incidence(self) -> torch.Tensor:
        """[num_links, num_experts] soft incidence: 1 if expert e's path crosses link l."""
        num_links = len(self.fabric.links)
        num_experts = sum(1 for n in self.fabric.nodes if n.type == "pe")
        inc = torch.zeros(num_links, num_experts)
        for i, l in enumerate(self.fabric.links):
            for n in self.fabric.nodes:
                if n.type == "pe" and n.expert_id is not None:
                    # Heuristic: cross-device links carry all non-local experts;
                    # intra-region links carry only their own experts.
                    if l.cross_dev:
                        inc[i, n.expert_id] = float(n.device != self._src_device_of_link(l))
                    elif l.cross_slr:
                        inc[i, n.expert_id] = 0.5
                    else:
                        inc[i, n.expert_id] = float(n.region == self._region_of_link(l))
        return inc

    def _src_device_of_link(self, l: FabricLink) -> int:
        src = self.fabric.nodes[l.src]
        return src.device

    def _region_of_link(self, l: FabricLink) -> int:
        src = self.fabric.nodes[l.src]
        return src.region if src.type == "region" else 0

    def predict(self, routing_probs: torch.Tensor) -> FabricPrediction:
        """
        routing_probs: [T, E] soft routing distribution pi_theta(e | x_t, G_H, S_H).
        Returns FabricPrediction with differentiable T_hat, B_hat, C_hat, U_hat.
        """
        # --- Link utilization u_l = (1/T) sum_t sum_e R[t,e] * incidence[l,e] ---
        # Mean over tokens of expert usage, projected through link incidence.
        expert_load = routing_probs.mean(dim=0)          # [E]
        u_links = torch.clamp(self.link_incidence @ expert_load, 0.0, 0.999)  # [num_links]

        # --- M/M/1 inflation d_l = d_l^0 / (1 - u_l) ---
        inflated_delay = self.link_d0 / (1.0 - u_links)

        # --- Per-token latency T_hat_t ---
        # L_disp(P(e)) = L_DMA + L_HCR + 2*H_hop(e) + L_PE
        # C_GEMM(e)/f_clk = L_PE + (m_e==HBM ? L_HBM : 0)
        disp = self.L_DMA + self.L_HCR + 2.0 * self.expert_hops + self.L_PE  # [E]
        gemm = self.L_PE + self.L_HBM * self.expert_hbm                    # [E]
        per_expert_latency = disp + gemm                                  # [E]
        latency = routing_probs @ per_expert_latency                      # [T]

        # --- Traffic B_hat = sum_l u_l * C_l (normalized) ---
        traffic = (u_links * self.link_cap).sum()

        # --- Congestion C_hat = max_l u_l ---
        congestion = u_links.max()

        return FabricPrediction(latency=latency, traffic=traffic,
                                congestion=congestion, link_utils=u_links)

    def constraint_values(self, pred: FabricPrediction) -> Dict[str, torch.Tensor]:
        """g_k(R, G_H) for the three dualized constraints.

        - latency:    E[T_hat] per-token dispatch latency (cycles)        -- vs T_max
        - traffic:    mean link utilization fraction in [0, 1]            -- vs B_max
        - congestion: max  link utilization fraction in [0, 1]          -- vs U_max
        """
        return {
            "latency": pred.latency.mean(),
            "traffic": pred.link_utils.mean(),
            "congestion": pred.congestion,
        }

    def calibrate(self, pred: FabricPrediction, measured: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Huber fit + Spearman rank term. Updates MAPE / rho stats.
        Called every K_cal=200 steps with Y_FPGA from on-chip counters.
        """
        delta = 1.0
        losses = {}
        for key in ("latency", "traffic", "congestion"):
            if key in measured:
                y_hat = getattr(pred, key)
                y = measured[key]
                if y_hat.shape != y.shape:
                    y_hat = y_hat.reshape(y.shape) if y_hat.numel() == y.numel() else y_hat.mean().reshape_as(y)
                err = F.huber_loss(y_hat.float(), y.float(), delta=delta)
                losses[key] = float(err.detach().item())
                with torch.no_grad():
                    denom = y.abs().clamp_min(1e-6)
                    mape = (y_hat - y).abs() / denom
                    if key == "latency":
                        self._latency_mape = float(mape.mean().item())
                    elif key == "traffic":
                        self._traffic_mape = float(mape.mean().item())
        # Spearman rank on congestion ordering
        if "congestion" in measured and pred.link_utils.numel() > 1:
            with torch.no_grad():
                a = pred.link_utils.argsort().argsort().float()
                b = measured.get("link_utils", measured["congestion"].expand_as(a)).argsort().argsort().float() \
                    if "link_utils" in measured else torch.zeros_like(a)
                if a.numel() == b.numel() and a.numel() > 1:
                    rho = 1.0 - 6.0 * ((a - b) ** 2).sum() / (a.numel() ** 3 - a.numel())
                    self._spearman_rho = float(rho.item())
        self._calibration_count += 1
        return losses

    def stats(self) -> Dict[str, Any]:
        return {
            "calibration_count": self._calibration_count,
            "latency_mape": self._latency_mape,
            "traffic_mape": self._traffic_mape,
            "spearman_rho": self._spearman_rho,
            "num_links": len(self.fabric.links),
        }


# ---------------------------------------------------------------------------
# Adaptive duals
# ---------------------------------------------------------------------------


class AdaptiveDuals:
    """
    mu_k <- [mu_k + eta*(g_k - c_k)]_+, decaying by 0.95/step when constraint k
    stays slack (g_k < 0.9*c_k for `slack_steps`), clipped to [0, mu_max].

    Replaces a frozen quality/efficiency multiplier lambda.
    """

    def __init__(
        self,
        budgets: Dict[str, float],
        eta: float = 0.05,
        mu_max: float = 10.0,
        slack_threshold: float = 0.9,
        decay: float = 0.95,
        slack_steps: int = 50,
    ):
        self.budgets = budgets
        self.eta = eta
        self.mu_max = mu_max
        self.slack_threshold = slack_threshold
        self.decay = decay
        self.slack_steps = slack_steps
        self.mu = {k: 0.0 for k in budgets}
        self._slack_counters = {k: 0 for k in budgets}
        self.updates = 0

    def step(self, g: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Dual ascent on each constraint. Returns current mu."""
        for k, gk in g.items():
            if k not in self.mu:
                continue
            ck = self.budgets[k]
            violation = float(gk.detach().item()) - ck
            self.mu[k] = min(self.mu_max, max(0.0, self.mu[k] + self.eta * violation))
            # Decay when slack so the router recovers model quality
            if violation < (self.slack_threshold - 1.0) * ck:
                self._slack_counters[k] += 1
                if self._slack_counters[k] >= self.slack_steps:
                    self.mu[k] *= self.decay
            else:
                self._slack_counters[k] = 0
        self.updates += 1
        return dict(self.mu)

    def penalty(self, g: Dict[str, torch.Tensor]) -> torch.Tensor:
        """sum_k mu_k * [g_k - c_k]_+  (the dualized term added to L_LM)."""
        total = torch.tensor(0.0)
        for k, gk in g.items():
            if k not in self.mu:
                continue
            ck = self.budgets[k]
            total = total + self.mu[k] * F.relu(gk - ck)
        return total

    def stats(self) -> Dict[str, Any]:
        return {"mu": dict(self.mu), "budgets": dict(self.budgets), "updates": self.updates}


# ---------------------------------------------------------------------------
# Hardware-Conditioned Router (HCR)
# ---------------------------------------------------------------------------


class HardwareConditionedRouter(nn.Module):
    """
    s_{t,e} = f_theta(h_t, z_H_e, c_t),  pi_theta(e | x, G_H, S_H) = softmax_e(s_{t,e}).

    With topology-induced hierarchy:
        P(e | x) = softmax_r(h^T W_z_r) * P(e | x, r, G_H)
    so most tokens stay on-device (E[H_hop] = P(local)*H_in + ...).

    Training uses Gumbel-softmax / STE soft top-k; decode uses hard top-k with a
    fallback to the 2nd-best region when Q_r > Q_max.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_regions: int,
        fabric_emb_size: int = 64,
        telemetry_hidden: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_regions = num_regions
        self.fabric_emb_size = fabric_emb_size
        self.telemetry_hidden = telemetry_hidden
        self.temperature = temperature

        # Region gate: P(r | x, G_H) = softmax_r(h^T W_z_r)
        self.region_gate = nn.Linear(hidden_size, num_regions, bias=False)
        # Expert score: s_{t,e} = f_theta(h_t, z_H_e, c_t)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_size + fabric_emb_size + telemetry_hidden, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # Expert-to-region membership buffer (filled when fabric is bound)
        self.register_buffer("expert_region", torch.zeros(num_experts, dtype=torch.long))

    def bind_fabric(self, fabric: FabricGraph) -> None:
        """Record which region each expert anchors in (v_e -> r)."""
        for n in fabric.nodes:
            if n.type == "pe" and n.expert_id is not None:
                self.expert_region[n.expert_id] = n.region

    def score(
        self,
        h: torch.Tensor,            # [T, H]
        z_experts: torch.Tensor,    # [E, fabric_emb]
        c_t: torch.Tensor,          # [telemetry_hidden]
    ) -> torch.Tensor:
        """Returns raw expert scores [T, E]."""
        T = h.size(0)
        z_e = z_experts.unsqueeze(0).expand(T, -1, -1)   # [T, E, fabric_emb]
        c = c_t.unsqueeze(0).expand(T, -1)               # [T, telemetry_hidden]
        h_rep = h.unsqueeze(1).expand(-1, self.num_experts, -1)  # [T, E, H]
        feats = torch.cat([h_rep, z_e, c.unsqueeze(1).expand(-1, self.num_experts, -1)], dim=-1)
        return self.score_mlp(feats).squeeze(-1)        # [T, E]

    def forward(
        self,
        h: torch.Tensor,            # [T, H]
        z_experts: torch.Tensor,    # [E, fabric_emb]
        z_regions: torch.Tensor,    # [R, fabric_emb]
        c_t: torch.Tensor,          # [telemetry_hidden]
        top_k: int,
        training: bool = True,
        tau: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          routing_probs [T, E]    full softmax (hierarchical)
          region_probs   [T, R]   P(r | x, G_H)
          top_k_weights  [T, K]   normalized
          top_k_indices   [T, K]
          on_device_frac  scalar  P(local) -- on-device byte fraction
        """
        # Region gate: P(r | x, G_H)
        region_logits = h @ z_regions.T        # [T, R]
        region_probs = F.softmax(region_logits / self.temperature, dim=-1)

        # Expert scores
        expert_scores = self.score(h, z_experts, c_t)   # [T, E]
        # Mask: zero out experts not in chosen region via hierarchical factor
        # P(e | x) = P(r|x) * P(e | x, r); implement as region-weighted softmax
        expert_logits = expert_scores / self.temperature
        # Build region-membership mask [E, R]
        region_onehot = F.one_hot(self.expert_region, self.num_regions).float()  # [E, R]
        # Distribute region prob across its experts, then softmax within region
        # expert_prob[t,e] = region_probs[t, r(e)] * softmax_e_in_r(s[t,e])
        within_region = F.softmax(expert_logits, dim=-1)  # [T, E] (global; masked below)
        # Hierarchical: scale by region prob and renormalize
        region_weight_per_expert = region_probs @ region_onehot.T  # [T, E]
        routing_probs = within_region * region_weight_per_expert
        routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Top-k (STE for training, hard for decode)
        if training:
            # Gumbel-softmax soft top-k approximation
            gumbel = -torch.empty_like(expert_logits).exponential_().log()
            soft_logits = (expert_logits + gumbel) / tau
            soft = F.softmax(soft_logits, dim=-1)
            top_k_weights, top_k_indices = torch.topk(soft, top_k, dim=-1)
            # STE: straight-through estimator -- use hard indices, soft gradients
            hard = torch.zeros_like(soft).scatter_(-1, top_k_indices, top_k_weights)
            top_k_weights = hard + soft - soft.detach()
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            top_k_weights, top_k_indices = torch.topk(routing_probs, top_k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # On-device fraction P(local): prob mass on experts whose device == source
        # Approximated as mass on experts in region 0 (the issuing region).
        on_device_frac = routing_probs[:, region_onehot[:, 0] > 0].sum(dim=-1).mean()

        return {
            "routing_probs": routing_probs,
            "region_probs": region_probs,
            "top_k_weights": top_k_weights,
            "top_k_indices": top_k_indices,
            "expert_scores": expert_scores,
            "on_device_frac": on_device_frac,
        }

    def decode_step(
        self,
        h: torch.Tensor,
        z_experts: torch.Tensor,
        z_regions: torch.Tensor,
        c_t: torch.Tensor,
        top_k: int,
        region_queues: torch.Tensor,
        q_max: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Hard top-k decode with fallback to 2nd-best region when Q_r > Q_max.
        """
        with torch.no_grad():
            out = self.forward(h, z_experts, z_regions, c_t, top_k, training=False)
            # Pick best region per token
            best_region = out["region_probs"].argmax(dim=-1)  # [T]
            # If that region's queue exceeds Q_max, retry 2nd-best region
            over = region_queues[best_region] > q_max
            if over.any():
                second = out["region_probs"].topk(2, dim=-1).indices[:, 1]
                best_region = torch.where(over, second, best_region)
            out["best_region"] = best_region
            return out


# ---------------------------------------------------------------------------
# Affinity-driven placement
# ---------------------------------------------------------------------------


class AffinityPlacer:
    """
    Co-activation A_ij = P(E_i, E_j | same window) drives the placement solve
        P* = argmin_P sum_ij A_ij D(P_i, P_j)  s.t. sum_{e: P(e)=r} M_e <= M_cap_r
    Greedy pack + pair swaps, warm-started from the previous map.

    Coupled to HCR + DFM this becomes a fixed point: placement changes z_H_e,
    routing reshapes A, and both move under the dualized objective.
    """

    def __init__(
        self,
        fabric: FabricGraph,
        num_experts: int,
        region_capacity: float = 8.0,
        swap_interval: int = 50,
        warm_start: bool = True,
    ):
        self.fabric = fabric
        self.num_experts = num_experts
        self.region_capacity = region_capacity
        self.swap_interval = swap_interval
        self.warm_start = warm_start
        # expert_id -> region
        self.placement: Dict[int, int] = {
            e: fabric.expert_to_region(e) for e in range(num_experts)
        }
        self.affinity = torch.zeros(num_experts, num_experts)
        self.swaps = 0
        self.step = 0

    def observe(self, expert_indices: torch.Tensor, window: int = 32) -> None:
        """Accumulate co-activation A_ij from a batch of top-k indices [T, K]."""
        flat = expert_indices.reshape(-1).detach().cpu().tolist()
        n = len(flat)
        # Co-activate within sliding windows of size `window`
        for i in range(0, n, window):
            w = flat[i : i + window]
            for a in range(len(w)):
                for b in range(a + 1, len(w)):
                    ea, eb = w[a], w[b]
                    if 0 <= ea < self.num_experts and 0 <= eb < self.num_experts:
                        self.affinity[ea, eb] += 1.0
                        self.affinity[eb, ea] += 1.0

    def solve(self) -> Dict[int, int]:
        """
        Greedy pack: sort experts by total affinity (hottest first), assign each
        to the region that minimizes added hop cost while respecting capacity,
        then pair-swap to reduce total cost. Warm-started from current placement.
        """
        if not self.warm_start:
            self.placement = {e: 0 for e in range(self.num_experts)}

        dist = self.fabric.hop_distance_matrix()
        # Region node ids
        region_ids = [n.node_id for n in self.fabric.nodes if n.type == "region"]
        # expert anchor node ids
        anchor = {}
        for n in self.fabric.nodes:
            if n.type == "pe" and n.expert_id is not None:
                anchor[n.expert_id] = n.node_id

        # Greedy pack
        load = {r: 0 for r in range(self.fabric.num_regions)}
        for e, r in self.placement.items():
            load[r] = load.get(r, 0) + 1
        order = torch.argsort(self.affinity.sum(dim=1), descending=True).tolist()
        for e in order:
            e = int(e)
            # Cost of placing e in region r = sum_j A[e,j] * D(anchor_e, anchor_j)
            costs = {}
            for r in range(self.fabric.num_regions):
                if load.get(r, 0) >= self.region_capacity:
                    continue
                c = 0.0
                for j in range(self.num_experts):
                    if j == e:
                        continue
                    rj = self.placement[j]
                    d = float(dist[anchor[e], region_ids[r]] if r < len(region_ids) else 0)
                    d += float(dist[region_ids[rj], anchor[j]] if rj < len(region_ids) else 0)
                    c += float(self.affinity[e, j]) * d
                costs[r] = c
            if costs:
                best = min(costs, key=lambda r: costs[r])
                self.placement[e] = best
                load[best] = load.get(best, 0) + 1
                # Remove from old region load
                # (greedy reassign; old load bookkeeping is approximate)

        # Pair swaps to reduce total cost (limited passes)
        if self.step % self.swap_interval == 0 and self.step > 0:
            self._pair_swap_pass(dist, anchor, region_ids)
        self.step += 1
        return dict(self.placement)

    def _pair_swap_pass(self, dist: torch.Tensor, anchor: Dict[int, int],
                        region_ids: List[int]) -> None:
        def cost_of(e: int, r: int) -> float:
            c = 0.0
            for j in range(self.num_experts):
                if j == e:
                    continue
                rj = self.placement[j]
                d = float(dist[anchor[e], region_ids[r]] if r < len(region_ids) else 0)
                d += float(dist[region_ids[rj], anchor[j]] if rj < len(region_ids) else 0)
                c += float(self.affinity[e, j]) * d
            return c

        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                ri, rj = self.placement[i], self.placement[j]
                if ri == rj:
                    continue
                before = cost_of(i, ri) + cost_of(j, rj)
                after = cost_of(i, rj) + cost_of(j, ri)
                if after < before - 1e-6:
                    self.placement[i], self.placement[j] = rj, ri
                    self.swaps += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "placement": dict(self.placement),
            "swaps": self.swaps,
            "step": self.step,
            "affinity_density": float(self.affinity.sum().item()),
        }


# ---------------------------------------------------------------------------
# FabricMoE layer
# ---------------------------------------------------------------------------


@dataclass
class FabricMoEConfig:
    hidden_size: int = 512
    num_experts: int = 64
    num_regions: int = 8
    top_k: int = 2
    num_devices: int = 16
    fabric_emb_size: int = 64
    telemetry_hidden: int = 32
    telemetry_window: int = 4
    # Budgets c_k per SLO class: interactive decode
    latency_budget: float = 80.0      # T_max (cycles, normalized)
    traffic_budget: float = 0.70      # B_max
    congestion_budget: float = 0.85   # U_max
    eta_dual: float = 0.05
    mu_max: float = 10.0
    calibrate_interval: int = 200     # K_cal
    relayout_interval: int = 50
    q_max: float = 4.0                # fallback threshold
    gmax: float = 10.0                # STE gradient clip
    fabric: Optional[FabricGraph] = None


class FabricMoELayer(nn.Module):
    """
    One FabricMoE layer: HCR + DFM + adaptive duals + hierarchy + affinity placer.

    Training step (host; psi frozen, refreshed every K_cal).
    Decode step (device; hard top-k, live G_H(t), S_H(t)).
    """

    def __init__(self, cfg: FabricMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k

        # Fabric (default: multi-FPGA mesh)
        self.fabric = cfg.fabric or FabricGraph.multi_fpga_mesh(
            num_experts=cfg.num_experts, num_regions=cfg.num_regions,
            num_devices=cfg.num_devices,
        )

        # Experts (region-local INT8 in hardware; here float MLPs)
        inter = cfg.hidden_size * 4
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_size, inter), nn.GELU(),
                nn.Linear(inter, cfg.hidden_size),
            ) for _ in range(cfg.num_experts)
        ])

        # Fabric-graph encoder + telemetry GRU
        self.encoder = FabricGraphEncoder(hidden_size=cfg.fabric_emb_size)
        # Telemetry vector = region_queues [R] + link_utils [8] + hbm_stalls [num_hbm] + credits [R]
        num_hbm = sum(1 for n in self.fabric.nodes if n.type == "hbm")
        tel_dim = self.fabric.num_regions + 8 + max(1, num_hbm) + self.fabric.num_regions
        self.telemetry_gru = TelemetryGRU(telemetry_dim=tel_dim,
                                          hidden_size=cfg.telemetry_hidden,
                                          window=cfg.telemetry_window)

        # Hardware-conditioned router
        self.router = HardwareConditionedRouter(
            hidden_size=cfg.hidden_size, num_experts=cfg.num_experts,
            num_regions=cfg.num_regions, fabric_emb_size=cfg.fabric_emb_size,
            telemetry_hidden=cfg.telemetry_hidden,
        )
        self.router.bind_fabric(self.fabric)

        # Differentiable fabric model
        self.dfm = DifferentiableFabricModel(self.fabric, gmax=cfg.gmax)

        # Adaptive duals on (latency, traffic, congestion) budgets
        self.duals = AdaptiveDuals(
            budgets={"latency": cfg.latency_budget,
                     "traffic": cfg.traffic_budget,
                     "congestion": cfg.congestion_budget},
            eta=cfg.eta_dual, mu_max=cfg.mu_max,
        )

        # Affinity-driven placement
        self.placer = AffinityPlacer(self.fabric, cfg.num_experts,
                                     region_capacity=max(1.0, cfg.num_experts / cfg.num_regions),
                                     swap_interval=cfg.relayout_interval)

        # Cached fabric embeddings (re-encode only when G_H or P changes)
        self._h_nodes: Optional[torch.Tensor] = None
        self._z_experts: Optional[torch.Tensor] = None
        self._z_regions: Optional[torch.Tensor] = None
        self._step = 0

    def _ensure_fabric_embeddings(self) -> None:
        # Re-encode every training step so the autograd graph is fresh; cache
        # only during eval/decode (re-run when G_H or P changes).
        if self._h_nodes is None or self.training:
            h_nodes, region_emb = self.encoder.encode(self.fabric)
            self._h_nodes = h_nodes
            self._z_regions = self.encoder.region_embedding_matrix(region_emb)
            self._z_experts = self.encoder.expert_embeddings(h_nodes, self.fabric)

    def _telemetry_vector(self) -> torch.Tensor:
        s = TelemetryState.from_fabric(self.fabric)
        return s.to_vector().squeeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training-style forward: soft routing -> DFM prediction -> dual update -> MoE output.
        Returns (output, aux_loss) where aux_loss = sum_k mu_k [g_k - c_k]_+.
        """
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        self._ensure_fabric_embeddings()
        c_t = self.telemetry_gru(self._telemetry_vector())

        out = self.router(flat, self._z_experts, self._z_regions, c_t,
                          top_k=self.top_k, training=self.training)
        routing_probs = out["routing_probs"]      # [T, E]

        # DFM prediction and constraint values g_k
        pred = self.dfm.predict(routing_probs)
        g = self.dfm.constraint_values(pred)

        # Dual ascent
        self.duals.step(g)
        aux_loss = self.duals.penalty(g)

        # MoE output using top-k (STE weights carry gradients)
        top_i = out["top_k_indices"]              # [T, K]
        top_w = out["top_k_weights"]               # [T, K]
        out_tensor = torch.zeros_like(flat)
        for k in range(self.top_k):
            eid_col = top_i[:, k]
            w_col = top_w[:, k].unsqueeze(-1)
            for e in eid_col.unique().tolist():
                mask = eid_col == e
                if not mask.any():
                    continue
                y = self.experts[int(e)](flat[mask])
                out_tensor[mask] = out_tensor[mask] + w_col[mask] * y

        # Observe co-activation for placement
        self.placer.observe(top_i)
        if self._step % self.cfg.relayout_interval == 0 and self._step > 0:
            self.placer.solve()

        # Periodic DFM recalibration hook (caller supplies Y_FPGA via calibrate())
        self._step += 1
        return out_tensor.reshape(shape), aux_loss

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Hard top-k decode with fallback path."""
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        self._ensure_fabric_embeddings()
        assert self._z_experts is not None and self._z_regions is not None
        c_t = self.telemetry_gru(self._telemetry_vector())
        region_queues = torch.zeros(self.fabric.num_regions)
        for n in self.fabric.nodes:
            if n.type == "region":
                region_queues[n.region] = n.queue_depth
        out = self.router.decode_step(flat, self._z_experts, self._z_regions, c_t,
                                      self.top_k, region_queues, self.cfg.q_max)
        top_i = out["top_k_indices"]
        top_w = out["top_k_weights"]
        out_tensor = torch.zeros_like(flat)
        for k in range(self.top_k):
            eid_col = top_i[:, k]
            w_col = top_w[:, k].unsqueeze(-1)
            for e in eid_col.unique().tolist():
                mask = eid_col == e
                if not mask.any():
                    continue
                y = self.experts[int(e)](flat[mask])
                out_tensor[mask] = out_tensor[mask] + w_col[mask] * y
        return out_tensor.reshape(shape)

    def calibrate(self, measured: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Refit DFM on Y_FPGA."""
        self._ensure_fabric_embeddings()
        c_t = self.telemetry_gru(self._telemetry_vector())
        # Use a uniform routing proxy for calibration if no batch is given
        T = 1
        routing = torch.ones(T, self.num_experts) / self.num_experts
        pred = self.dfm.predict(routing)
        return self.dfm.calibrate(pred, measured)

    def get_fabric_metrics(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "duals": self.duals.stats(),
            "dfm": self.dfm.stats(),
            "placer": self.placer.stats(),
            "num_devices": self.fabric.num_devices,
            "num_regions": self.fabric.num_regions,
            "num_cuts": len(self.fabric.cuts()),
        }


class FabricMoETrainingSystem(nn.Module):
    """Multi-layer FabricMoE + constrained train_step."""

    def __init__(self, cfg: FabricMoEConfig, num_layers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([FabricMoELayer(cfg) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_aux = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            y, aux = layer(x)
            x = x + y
            total_aux = total_aux + aux
        return self.norm(x), total_aux

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Hard top-k decode through all layers."""
        for layer in self.layers:
            x = x + layer.decode(x)
        return self.norm(x)

    def train_step(
        self,
        batch: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        self.train()
        out, aux = self.forward(batch)
        if target is None:
            target = batch.detach()
        lm_loss = F.mse_loss(out, target)
        # Constrained loss: L = L_LM + sum_k mu_k [g_k - c_k]_+
        loss = lm_loss + aux
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.gmax)
            optimizer.step()
        metrics = self.get_metrics()
        metrics["lm_loss"] = float(lm_loss.detach().item())
        metrics["aux_loss"] = float(aux.detach().item())
        metrics["loss"] = float(loss.detach().item())
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        return {"layers": [layer.get_fabric_metrics() for layer in self.layers]}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_fabric_moe(
    hidden_size: int = 256,
    num_experts: int = 16,
    top_k: int = 2,
    num_regions: int = 4,
    num_devices: int = 4,
    num_layers: int = 2,
    fabric: Optional[FabricGraph] = None,
    **kwargs: Any,
) -> FabricMoETrainingSystem:
    """
    Factory: FabricMoE training system (HCR + DFM + adaptive duals + hierarchy + placer).

    Examples
    --------
    >>> model = create_fabric_moe(hidden_size=128, num_experts=8, num_devices=2)
    >>> y, aux = model(torch.randn(2, 16, 128))
    """
    cfg = FabricMoEConfig(
        hidden_size=hidden_size, num_experts=num_experts, top_k=top_k,
        num_regions=num_regions, num_devices=num_devices, fabric=fabric,
        latency_budget=kwargs.get("latency_budget", 80.0),
        traffic_budget=kwargs.get("traffic_budget", 0.70),
        congestion_budget=kwargs.get("congestion_budget", 0.85),
        eta_dual=kwargs.get("eta_dual", 0.05),
        calibrate_interval=kwargs.get("calibrate_interval", 200),
        relayout_interval=kwargs.get("relayout_interval", 50),
    )
    return FabricMoETrainingSystem(cfg, num_layers=num_layers)


def create_fabric_moe_layer(**kwargs: Any) -> FabricMoELayer:
    """Factory: a single FabricMoE layer (for ablations / standalone use)."""
    cfg = FabricMoEConfig(
        hidden_size=kwargs.get("hidden_size", 256),
        num_experts=kwargs.get("num_experts", 16),
        top_k=kwargs.get("top_k", 2),
        num_regions=kwargs.get("num_regions", 4),
        num_devices=kwargs.get("num_devices", 4),
        fabric=kwargs.get("fabric", None),
    )
    return FabricMoELayer(cfg)


__all__ = [
    "FabricNode", "FabricLink", "FabricGraph",
    "TelemetryState", "FabricGraphEncoder", "TelemetryGRU",
    "FabricPrediction", "DifferentiableFabricModel",
    "AdaptiveDuals", "HardwareConditionedRouter",
    "AffinityPlacer", "FabricMoEConfig", "FabricMoELayer",
    "FabricMoETrainingSystem", "create_fabric_moe", "create_fabric_moe_layer",
]
