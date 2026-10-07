# SPDX-License-Identifier: Apache-2.0
"""Deterministic token-level placement for expert-parallel prefill.

Counts refer to token/expert assignments: a token routed to k experts contributes
k assignments. Occurrences of an expert are ordered by source rank, then by the
flattened [token, top-k slot] position on that rank.
"""

from bisect import bisect_right
from dataclasses import dataclass
from operator import index
from typing import Sequence


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def contiguous_owners(num_experts: int, world_size: int) -> tuple[int, ...]:
    """Assign contiguous expert ranges, including a non-divisible remainder."""
    num_experts = _integer(num_experts, "num_experts", 1)
    world_size = _integer(world_size, "world_size", 1)
    quotient, remainder = divmod(num_experts, world_size)
    return tuple(rank for rank in range(world_size)
                 for _ in range(quotient + (rank < remainder)))


@dataclass(frozen=True)
class RoutingPlan:
    """Per-rank counts and half-open occurrence intervals for every expert."""

    counts: tuple[int, ...]
    owners: tuple[int, ...]
    quotas: tuple[int, ...]
    allocation: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        counts = tuple(_integer(n, "count") for n in self.counts)
        owners = tuple(_integer(n, "owner") for n in self.owners)
        quotas = tuple(_integer(n, "quota") for n in self.quotas)
        rows = tuple(tuple(_integer(n, "allocation") for n in row)
                     for row in self.allocation)
        if not counts or not rows or len(quotas) != len(rows):
            raise ValueError("a plan requires experts and one quota per rank")
        if len(owners) != len(counts) or any(rank >= len(rows) for rank in owners):
            raise ValueError("invalid expert ownership")
        if any(len(row) != len(counts) for row in rows):
            raise ValueError("allocation rows must cover every expert")
        if tuple(map(sum, zip(*rows))) != counts:
            raise ValueError("allocation must conserve every expert's assignments")
        if tuple(map(sum, rows)) != quotas:
            raise ValueError("allocation must match the per-rank quotas")
        for name, value in (("counts", counts), ("owners", owners),
                            ("quotas", quotas), ("allocation", rows)):
            object.__setattr__(self, name, value)

    @property
    def world_size(self) -> int:
        return len(self.allocation)

    @property
    def loads(self) -> tuple[int, ...]:
        return tuple(sum(row) for row in self.allocation)

    def intervals(self, rank: int) -> tuple[tuple[int, int], ...]:
        rank = _integer(rank, "rank")
        if rank >= self.world_size:
            raise ValueError("rank is outside the process group")
        starts = [sum(row[e] for row in self.allocation[:rank])
                  for e in range(len(self.counts))]
        return tuple((start, start + count)
                     for start, count in zip(starts, self.allocation[rank]))

    def missing_experts(self, rank: int) -> tuple[int, ...]:
        self.intervals(rank)  # Validate the rank even for an empty workload.
        return tuple(e for e, count in enumerate(self.allocation[rank])
                     if count and self.owners[e] != rank)

    def destinations(self, expert_ids: Sequence[int], *, rank: int,
                     rank_counts: Sequence[Sequence[int]]) -> tuple[int, ...]:
        """Map each local assignment exactly once, without block rounding.

        All participants must supply the same rank-major count matrix. The
        local sequence must contain the entire rank's routing batch, even if
        expert computation is subsequently chunked.
        """
        self.intervals(rank)
        if len(rank_counts) != self.world_size:
            raise ValueError("rank_counts must contain one row per rank")
        rows = tuple(tuple(_integer(n, "count") for n in row)
                     for row in rank_counts)
        if any(len(row) != len(self.counts) for row in rows):
            raise ValueError("rank_counts has the wrong expert dimension")
        if tuple(map(sum, zip(*rows))) != self.counts:
            raise ValueError("rank_counts does not match the routing plan")
        ends = []
        for expert in range(len(self.counts)):
            cumulative = 0
            expert_ends = []
            for row in self.allocation:
                cumulative += row[expert]
                expert_ends.append(cumulative)
            ends.append(expert_ends)
        offsets = [sum(row[e] for row in rows[:rank])
                   for e in range(len(self.counts))]
        seen = [0] * len(self.counts)
        result = []
        for raw_expert in expert_ids:
            expert = _integer(raw_expert, "expert_id")
            if expert >= len(self.counts):
                raise ValueError("expert_id is outside the expert pool")
            if seen[expert] >= rows[rank][expert]:
                raise ValueError("routing sequence exceeds its declared counts")
            result.append(bisect_right(ends[expert], offsets[expert] + seen[expert]))
            seen[expert] += 1
        if tuple(seen) != rows[rank]:
            raise ValueError("routing sequence is incomplete")
        return tuple(result)


def balanced_plan(counts: Sequence[int], world_size: int,
                  owners: Sequence[int] | None = None) -> RoutingPlan:
    """Keep cold experts local, then place hot overflow in free capacity.

    Integer quotas sum to the exact number of assignments. Stable tie breaking
    gives every rank the same plan, including empty and heavily skewed batches.
    """
    world_size = _integer(world_size, "world_size", 1)
    counts = tuple(_integer(n, "count") for n in counts)
    if not counts:
        raise ValueError("at least one expert is required")
    owners = (contiguous_owners(len(counts), world_size) if owners is None
              else tuple(_integer(r, "owner") for r in owners))
    if len(owners) != len(counts) or any(r >= world_size for r in owners):
        raise ValueError("owners must name one valid rank per expert")
    quotient, remainder = divmod(sum(counts), world_size)
    quotas = tuple(quotient + (rank < remainder) for rank in range(world_size))
    capacity = list(quotas)
    remaining = list(counts)
    allocation = [[0] * len(counts) for _ in range(world_size)]

    def assign(rank: int, expert: int) -> None:
        take = min(capacity[rank], remaining[expert])
        allocation[rank][expert] += take
        capacity[rank] -= take
        remaining[expert] -= take

    for rank in range(world_size):
        local = (e for e, owner in enumerate(owners) if owner == rank)
        for expert in sorted(local, key=lambda e: (counts[e], e)):
            assign(rank, expert)
    for expert in sorted(range(len(counts)), key=lambda e: (-remaining[e], e)):
        while remaining[expert]:
            rank = max(range(world_size), key=lambda r: (capacity[r], -r))
            if capacity[rank] == 0:
                raise RuntimeError("routing capacity was exhausted")
            assign(rank, expert)
    return RoutingPlan(counts, owners, quotas,
                       tuple(tuple(row) for row in allocation))
