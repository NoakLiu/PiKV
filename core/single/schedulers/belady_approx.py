import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..module.pikv_scheduling import BaseScheduler


class BeladyApproxScheduler(BaseScheduler):
    """Belady-Approx eviction by predicting next-use distance.
    Uses last attention distribution with positional decay to estimate next reuse step.
    Lightweight EMA predictor: E[next] = alpha * last_gap + (1-alpha) * E[next].
    Evict items with largest predicted next-use (furthest reuse).
    """

    def __init__(self, cache_size: int, hidden_size: int,
                 alpha: float = 0.6, pos_decay: float = 0.97):
        super().__init__(cache_size, hidden_size)
        self.alpha = float(alpha)
        self.pos_decay = float(pos_decay)

        self.register_buffer('ema_next_gap', torch.zeros(cache_size))
        self.register_buffer('last_access_step', torch.zeros(cache_size))
        self.register_buffer('global_step', torch.tensor(0))

        # track last attention mass observed per slot
        self.register_buffer('last_attention', torch.zeros(cache_size))

    @torch.no_grad()
    def update_attention(self, indices: torch.Tensor, attn_weights: torch.Tensor):
        # attn_weights shape [N] or [N, ...] -> reduce
        if attn_weights.dim() > 1:
            attn_weights = attn_weights.mean(dim=list(range(1, attn_weights.dim())))
        self.last_attention[indices] = attn_weights
        self.last_access_step[indices] = self.global_step

    @torch.no_grad()
    def step(self):
        self.global_step += 1
        # positional decay of last-attention as recency proxy
        self.last_attention.mul_(self.pos_decay)

    @torch.no_grad()
    def _predict_next_use(self, cache_len: int) -> torch.Tensor:
        # Use EMA over observed gaps (current_step - last_access)
        current = self.global_step
        gaps = (current - self.last_access_step[:cache_len]).clamp(min=0)
        self.ema_next_gap[:cache_len] = (
            self.alpha * gaps + (1.0 - self.alpha) * self.ema_next_gap[:cache_len]
        )
        # Smaller attention implies further reuse; invert by 1/attn
        attn = self.last_attention[:cache_len].clamp(min=1e-6)
        prediction = self.ema_next_gap[:cache_len] * (1.0 / attn)
        return prediction

    def select_eviction_candidates(self, keys: torch.Tensor, values: torch.Tensor,
                                   metadata: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        cache_len = keys.size(0)
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)

        pred_next = self._predict_next_use(cache_len)
        num_to_evict = cache_len - self.cache_size
        _, evict_indices = torch.topk(pred_next, num_to_evict, largest=True)

        evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[evict_indices] = True

        self.update_stats(eviction=bool(evict_mask.any().item()))
        return evict_mask

