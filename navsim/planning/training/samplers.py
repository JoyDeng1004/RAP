"""Distributed samplers used by controlled alignment experiments."""

from __future__ import annotations

import math
from typing import Iterator, Optional, Sequence

import torch
from torch.utils.data import Sampler


class ExactDistributedSampler(Sampler[int]):
    """Shard one global order by rank without padding or dropping samples."""

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        shuffle: bool = True,
        tokens: Optional[Sequence[str]] = None,
    ) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError(f"invalid distributed topology: replicas={num_replicas}, rank={rank}")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.tokens = list(tokens) if tokens is not None else [str(index) for index in range(len(dataset))]
        if len(self.tokens) != len(dataset) or len(self.tokens) != len(set(self.tokens)):
            raise ValueError("ExactDistributedSampler requires one unique token per dataset item")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def global_indices(self, epoch: Optional[int] = None) -> list[int]:
        epoch = self.epoch if epoch is None else int(epoch)
        if not self.shuffle:
            return list(range(len(self.dataset)))
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def global_token_order(self, epoch: Optional[int] = None) -> list[str]:
        return [self.tokens[index] for index in self.global_indices(epoch)]

    def __iter__(self) -> Iterator[int]:
        return iter(self.global_indices()[self.rank :: self.num_replicas])

    def __len__(self) -> int:
        size = len(self.dataset)
        return max(math.ceil((size - self.rank) / self.num_replicas), 0)
