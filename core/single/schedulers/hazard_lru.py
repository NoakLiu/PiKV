import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..module.pikv_scheduling import BaseScheduler


class HazardLRUScheduler(BaseScheduler):
    """Hazard-LRU: fuse aging, dissimilarity, and uncertainty into a risk score.
    score = alpha * age + beta * (1 - cosine_similarity) + gamma * entropy
    Evict highest-risk first.
    """

    def __init__(self, cache_size: int, hidden_size: int,
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        super().__init__(cache_size, hidden_size)
        total = max(1e-6, alpha + beta + gamma)
        self.alpha = float(alpha) / total
        self.beta = float(beta) / total
        self.gamma = float(gamma) / total

        self.register_buffer('last_update_step', torch.zeros(cache_size))
        self.register_buffer('global_step', torch.tensor(0))
        self.register_buffer('last_key', torch.zeros(cache_size, hidden_size))

    @torch.no_grad()
    def update_access(self, indices: torch.Tensor, keys: torch.Tensor):
        self.last_update_step[indices] = self.global_step
        self.last_key[indices] = keys

    @torch.no_grad()
    def step(self):
        self.global_step += 1

    def _entropy_from_metadata(self, metadata: Optional[Dict[str, torch.Tensor]], cache_len: int) -> torch.Tensor:
        if metadata is None:
            return torch.zeros(cache_len, device=self.last_key.device)
        ent = metadata.get('uncertainty_entropy', None)
        if ent is None:
            return torch.zeros(cache_len, device=self.last_key.device)
        if ent.dim() > 1:
            ent = ent.mean(dim=list(range(1, ent.dim())))
        return ent[:cache_len].to(self.last_key.device)

    def select_eviction_candidates(self, keys: torch.Tensor, values: torch.Tensor,
                                   metadata: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        cache_len = keys.size(0)
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)

        # age component
        age = (self.global_step - self.last_update_step[:cache_len]).clamp(min=0)
        age = age / (age.max() + 1e-6)

        # dissimilarity component (1 - cosine similarity to last key)
        last_k = self.last_key[:cache_len]
        sim = F.cosine_similarity(keys[:cache_len], last_k, dim=-1)
        dissim = (1.0 - sim) * 0.5 + 0.5  # map to [0,1]

        # uncertainty component
        entropy = self._entropy_from_metadata(metadata, cache_len)
        entropy = entropy / (entropy.max() + 1e-6)

        risk = self.alpha * age + self.beta * dissim + self.gamma * entropy

        num_to_evict = cache_len - self.cache_size
        _, evict_indices = torch.topk(risk, num_to_evict, largest=True)

        evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[evict_indices] = True

        self.update_stats(eviction=bool(evict_mask.any().item()))
        return evict_mask

