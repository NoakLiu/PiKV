import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from ..module.pikv_scheduling import BaseScheduler


class TwoQueueScheduler(BaseScheduler):
    """Hierarchical cache with admission control (HBM↔CPU↔NVMe like tiers).
    - Admission: only tokens with predicted cumulative attention quality > tau enter HBM
    - Cold data demoted after cooldown into lower tiers.
    """

    def __init__(self, cache_size: int, hidden_size: int,
                 admission_tau: float = 0.5, demote_cooldown: int = 16,
                 cpu_ratio: float = 0.5):
        super().__init__(cache_size, hidden_size)
        self.admission_tau = float(admission_tau)
        self.demote_cooldown = int(demote_cooldown)
        self.cpu_size = int(cache_size * cpu_ratio)

        # HBM tier
        self.register_buffer('hbm_keys', torch.zeros(cache_size, hidden_size))
        self.register_buffer('hbm_values', torch.zeros(cache_size, hidden_size))
        self.register_buffer('hbm_valid', torch.zeros(cache_size, dtype=torch.bool))
        self.register_buffer('hbm_size', torch.tensor(0))
        self.register_buffer('hbm_quality', torch.zeros(cache_size))
        self.register_buffer('hbm_last_access', torch.zeros(cache_size))

        # CPU tier
        self.register_buffer('cpu_keys', torch.zeros(self.cpu_size, hidden_size))
        self.register_buffer('cpu_values', torch.zeros(self.cpu_size, hidden_size))
        self.register_buffer('cpu_valid', torch.zeros(self.cpu_size, dtype=torch.bool))
        self.register_buffer('cpu_size_cur', torch.tensor(0))
        self.register_buffer('cpu_last_access', torch.zeros(self.cpu_size))

        self.register_buffer('global_step', torch.tensor(0))

    @torch.no_grad()
    def _predict_quality(self, key: torch.Tensor, value: torch.Tensor) -> float:
        # Simple proxy: norm-based quality
        quality = 0.5 * key.norm().item() + 0.5 * value.norm().item()
        return float(quality)

    @torch.no_grad()
    def _admit_to_hbm(self, key: torch.Tensor, value: torch.Tensor, quality: float):
        if self.hbm_size < self.cache_size:
            idx = int(self.hbm_size.item())
            self.hbm_keys[idx] = key
            self.hbm_values[idx] = value
            self.hbm_valid[idx] = True
            self.hbm_quality[idx] = quality
            self.hbm_last_access[idx] = self.global_step
            self.hbm_size += 1
        else:
            # Evict lowest quality from HBM
            valid_quality = self.hbm_quality[:self.hbm_size]
            _, evict_idx = torch.topk(valid_quality, 1, largest=False)
            eidx = int(evict_idx.item())
            self.hbm_keys[eidx] = key
            self.hbm_values[eidx] = value
            self.hbm_valid[eidx] = True
            self.hbm_quality[eidx] = quality
            self.hbm_last_access[eidx] = self.global_step

    @torch.no_grad()
    def _place_in_cpu(self, key: torch.Tensor, value: torch.Tensor):
        if self.cpu_size_cur < self.cpu_size:
            idx = int(self.cpu_size_cur.item())
            self.cpu_keys[idx] = key
            self.cpu_values[idx] = value
            self.cpu_valid[idx] = True
            self.cpu_last_access[idx] = self.global_step
            self.cpu_size_cur += 1
        else:
            # Evict oldest from CPU tier
            _, evict_idx = torch.topk(self.cpu_last_access[:self.cpu_size_cur], 1, largest=False)
            eidx = int(evict_idx.item())
            self.cpu_keys[eidx] = key
            self.cpu_values[eidx] = value
            self.cpu_valid[eidx] = True
            self.cpu_last_access[eidx] = self.global_step

    @torch.no_grad()
    def _demote_cold_from_hbm(self):
        # Demote items not accessed within cooldown to CPU
        if self.hbm_size.item() == 0:
            return
        last_acc = self.hbm_last_access[:self.hbm_size]
        cold_mask = (self.global_step - last_acc) >= self.demote_cooldown
        if cold_mask.any():
            idxs = torch.where(cold_mask)[0]
            for i in idxs:
                i = int(i.item())
                if self.hbm_valid[i]:
                    self._place_in_cpu(self.hbm_keys[i], self.hbm_values[i])
                    self.hbm_valid[i] = False
            # compact HBM
            keep_mask = self.hbm_valid[:self.hbm_size]
            num_keep = int(keep_mask.sum().item())
            if num_keep > 0:
                self.hbm_keys[:num_keep] = self.hbm_keys[:self.hbm_size][keep_mask]
                self.hbm_values[:num_keep] = self.hbm_values[:self.hbm_size][keep_mask]
                self.hbm_quality[:num_keep] = self.hbm_quality[:self.hbm_size][keep_mask]
                self.hbm_last_access[:num_keep] = self.hbm_last_access[:self.hbm_size][keep_mask]
            self.hbm_valid[:num_keep] = True
            self.hbm_valid[num_keep:self.hbm_size] = False
            self.hbm_size = torch.tensor(num_keep, device=self.hbm_size.device)

    @torch.no_grad()
    def select_eviction_candidates(self, keys: torch.Tensor, values: torch.Tensor,
                                   metadata: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # For unified interface: evict from HBM to meet cache_size
        cache_len = keys.size(0)
        if cache_len <= self.cache_size:
            return torch.zeros(cache_len, dtype=torch.bool, device=keys.device)

        # Evict lowest-quality items first
        qual = self.hbm_quality[:cache_len]
        num_to_evict = cache_len - self.cache_size
        _, evict_idx = torch.topk(qual, num_to_evict, largest=False)
        evict_mask = torch.zeros(cache_len, dtype=torch.bool, device=keys.device)
        evict_mask[evict_idx] = True
        self.update_stats(eviction=bool(evict_mask.any().item()))
        return evict_mask

    @torch.no_grad()
    def admit(self, key: torch.Tensor, value: torch.Tensor):
        self.global_step += 1
        quality = self._predict_quality(key, value)
        if quality >= self.admission_tau:
            self._admit_to_hbm(key, value, quality)
        else:
            self._place_in_cpu(key, value)
        self._demote_cold_from_hbm()

