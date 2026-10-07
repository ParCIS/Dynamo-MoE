# SPDX-License-Identifier: Apache-2.0
"""Exact token dispatch and inverse combine, modeled without a network.

Every top-k slot is an independent assignment identified by
(source_rank, token_index, topk_slot). Packed destination batches model the
payload of All-to-All-v; combine performs its inverse permutation and reduction.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .prefill import RoutingPlan, _integer


@dataclass(frozen=True)
class RoutingBatch:
    """Tokens and already-selected routes from one source rank.

    Arrays are borrowed for the duration of forward. The model's selected
    expert IDs and scores are preserved, including grouped or biased routing.
    Zero-token batches are valid; negative expert IDs and nonfinite inputs are
    rejected. Padding must be removed by the caller before forming a batch.
    """

    hidden: np.ndarray
    expert_ids: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        for name in ("hidden", "expert_ids", "scores"):
            object.__setattr__(self, name, np.asarray(getattr(self, name)))
        if self.hidden.ndim != 2 or self.hidden.shape[1] == 0:
            raise ValueError("hidden must have shape [tokens, positive hidden_size]")
        if (self.expert_ids.ndim != 2 or self.expert_ids.shape[1] == 0
                or self.expert_ids.shape[0] != self.hidden.shape[0]):
            raise ValueError("expert_ids must have shape [tokens, positive top_k]")
        if self.expert_ids.dtype.kind not in "iu" or np.any(self.expert_ids < 0):
            raise ValueError("expert_ids must contain nonnegative integers")
        if self.scores.shape != self.expert_ids.shape:
            raise ValueError("scores must have the same shape as expert_ids")
        for tensor in (self.hidden, self.scores):
            if tensor.dtype.kind != "f" or not np.isfinite(tensor).all():
                raise ValueError("hidden and scores must contain finite floating values")

    @classmethod
    def from_logits(cls, hidden: np.ndarray, logits: np.ndarray, top_k: int,
                    *, renormalize: bool = True) -> "RoutingBatch":
        """Optional softmax/top-k router with stable expert-ID tie breaking."""
        logits = np.asarray(logits)
        top_k = _integer(top_k, "top_k", 1)
        if logits.ndim != 2 or logits.shape[1] < top_k:
            raise ValueError("logits must have at least top_k expert columns")
        if logits.dtype.kind != "f" or not np.isfinite(logits).all():
            raise ValueError("logits must be finite floating values")
        if not isinstance(renormalize, bool):
            raise ValueError("renormalize must be a bool")
        logits = logits.astype(np.result_type(logits.dtype, np.float32), copy=False)
        ids = np.argsort(-logits, axis=1, kind="stable")[:, :top_k]
        # Extremely separated finite logits may subtract to -inf; their
        # mathematically negligible softmax probability is exactly zero here.
        with np.errstate(over="ignore", under="ignore"):
            scores = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores /= scores.sum(axis=1, keepdims=True)
        selected = np.take_along_axis(scores, ids, axis=1)
        if renormalize:
            selected /= selected.sum(axis=1, keepdims=True)
        return cls(hidden, ids, selected)


@dataclass(frozen=True)
class PackedTokens:
    """One destination's tokens with all metadata needed by inverse combine."""

    hidden: np.ndarray
    source_ranks: np.ndarray
    token_indices: np.ndarray
    topk_slots: np.ndarray
    expert_ids: np.ndarray
    scores: np.ndarray


def routing_counts(batches: Sequence[RoutingBatch],
                   num_experts: int) -> tuple[tuple[int, ...], ...]:
    """The rank-major metadata that an All-Gather makes globally available."""
    num_experts = _integer(num_experts, "num_experts", 1)
    rows = []
    for batch in batches:
        if batch.expert_ids.size and int(batch.expert_ids.max()) >= num_experts:
            raise ValueError("a routed expert is outside the weight pool")
        ids = batch.expert_ids.astype(np.int64, copy=False).ravel()
        rows.append(tuple(np.bincount(ids, minlength=num_experts).tolist()))
    return tuple(rows)


def dispatch(batches: Sequence[RoutingBatch], plan: RoutingPlan) -> tuple[PackedTokens, ...]:
    """Pack rank-major assignments into variable-sized destination payloads."""
    if len(batches) != plan.world_size:
        raise ValueError("one routing batch is required per rank")
    if len({batch.hidden.shape[1] for batch in batches}) != 1:
        raise ValueError("all ranks must use the same hidden dimension")
    rank_counts = routing_counts(batches, len(plan.counts))
    buckets: list[list[tuple[np.ndarray, ...]]] = [[] for _ in batches]
    for source, batch in enumerate(batches):
        flat_ids = batch.expert_ids.astype(np.int64, copy=False).ravel()
        destinations = np.asarray(plan.destinations(flat_ids, rank=source,
                                                    rank_counts=rank_counts))
        for destination in range(plan.world_size):
            positions = np.flatnonzero(destinations == destination)
            tokens, slots = np.divmod(positions, batch.expert_ids.shape[1])
            buckets[destination].append((
                batch.hidden[tokens], np.full(len(tokens), source, dtype=np.intp),
                tokens, slots, flat_ids[positions], batch.scores.ravel()[positions],
            ))
    return tuple(PackedTokens(*(np.concatenate(parts, axis=0)
                                for parts in zip(*bucket)))
                 for bucket in buckets)


def combine(batches: Sequence[RoutingBatch], packed: Sequence[PackedTokens],
            expert_outputs: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Undo dispatch and sum weighted expert results into original token order.

    Each route is required exactly once. Explicit slot metadata distinguishes
    top-k contributions even when one token selects the same expert twice.
    """
    if len(packed) != len(batches) or len(expert_outputs) != len(packed):
        raise ValueError("combine requires one payload and output per rank")
    dtype = np.result_type(np.float32, *(batch.scores.dtype for batch in batches),
                           *(np.asarray(output).dtype for output in expert_outputs))
    outputs = [np.zeros(batch.hidden.shape, dtype=dtype) for batch in batches]
    seen = [np.zeros(batch.expert_ids.shape, dtype=np.int64) for batch in batches]
    for payload, raw_values in zip(packed, expert_outputs):
        values = np.asarray(raw_values)
        if values.shape != payload.hidden.shape or values.dtype.kind != "f":
            raise ValueError("expert output shape and dtype must match packed tokens")
        count = len(values)
        fields = (payload.source_ranks, payload.token_indices, payload.topk_slots,
                  payload.expert_ids, payload.scores)
        if any(field.shape != (count,) for field in fields):
            raise ValueError("packed metadata must have one element per assignment")
        for indices in fields[:4]:
            if indices.dtype.kind not in "iu" or np.any(indices < 0):
                raise ValueError("packed routing indices must be nonnegative integers")
        if np.any(payload.source_ranks >= len(batches)):
            raise ValueError("invalid source rank in packed metadata")
        for source, batch in enumerate(batches):
            mask = payload.source_ranks == source
            tokens, slots = payload.token_indices[mask], payload.topk_slots[mask]
            if np.any(tokens >= len(batch.hidden)) or np.any(slots >= batch.expert_ids.shape[1]):
                raise ValueError("packed route is outside its source batch")
            if (not np.array_equal(payload.expert_ids[mask], batch.expert_ids[tokens, slots])
                    or not np.array_equal(payload.scores[mask], batch.scores[tokens, slots])):
                raise ValueError("dispatch changed a selected expert or routing score")
            np.add.at(seen[source], (tokens, slots), 1)
            weighted = values[mask].astype(dtype, copy=False) * payload.scores[mask, None]
            np.add.at(outputs[source], tokens, weighted)
    if any(not np.all(counts == 1) for counts in seen):
        raise ValueError("every top-k assignment must be combined exactly once")
    return tuple(outputs)
