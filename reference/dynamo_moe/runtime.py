# SPDX-License-Identifier: Apache-2.0
"""Executable, single-layer prefill semantics, with no serving framework.

All ranks are modeled in one process so scheduling, dispatch, expert migration,
and combine can be inspected numerically. NumPy GEMMs implement SwiGLU experts;
futures model transfer completion. This is not a CUDA or network benchmark.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Sequence

import numpy as np

from .prefill import RoutingPlan, _integer, balanced_plan, contiguous_owners
from .routing import PackedTokens, RoutingBatch, combine, dispatch, routing_counts
from .weights import ExpertWeights, SingleLayerExpertStore


@dataclass(frozen=True)
class PrefillResult:
    """Numerical outputs plus an inspectable record of the complete routing step."""

    outputs: tuple[np.ndarray, ...]
    plan: RoutingPlan
    dispatched: tuple[PackedTokens, ...]
    missing_experts: tuple[tuple[int, ...], ...]
    transferred_bytes: tuple[int, ...]


def _silu(values: np.ndarray) -> np.ndarray:
    # Split the sigmoid to avoid overflowing exp for very negative values.
    sigmoid = np.empty_like(values)
    positive = values >= 0
    sigmoid[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_negative = np.exp(values[~positive])
    sigmoid[~positive] = exp_negative / (1 + exp_negative)
    return values * sigmoid


def _compute(payload: PackedTokens, weights: ExpertWeights,
             output: np.ndarray) -> None:
    """Fill only the assignments whose experts exist in this weight batch."""
    for slot, expert in enumerate(weights.expert_ids):
        positions = np.flatnonzero(payload.expert_ids == expert)
        if not positions.size:
            continue
        hidden = payload.hidden[positions].astype(output.dtype, copy=False)
        w13 = weights.w13[slot].astype(output.dtype, copy=False)
        w2 = weights.w2[slot].astype(output.dtype, copy=False)
        gate, up = np.split(hidden @ w13.T, 2, axis=1)
        output[positions] = (_silu(gate) * up) @ w2.T


class PrefillLayer:
    """One layer's resident weights and complete balanced-EP forward path.

    ``store`` contains only this layer. Owners are fixed across requests; missed
    experts are loaded into temporary per-call storage and never replace the
    resident mapping. There is no cross-layer cache or global singleton.

    All ranks are represented by a sequence of RoutingBatch objects. Hidden
    states and weights must remain unchanged during forward. A shared-expert
    callback, if supplied, computes the entire shared branch on local tokens.
    Its output is added once, after scaling only the routed branch. This keeps
    model-specific routing and shared-expert gating outside the scheduler.
    """

    def __init__(self, store: SingleLayerExpertStore, world_size: int, *,
                 owners: Sequence[int] | None = None):
        self.store = store
        self.world_size = _integer(world_size, "world_size", 1)
        self.owners = (contiguous_owners(store.num_experts, self.world_size)
                       if owners is None else tuple(owners))
        # Validate a custom topology even before any tokens are provided.
        self.owners = balanced_plan([0] * store.num_experts, self.world_size,
                                    self.owners).owners
        self._resident = tuple(self._load(tuple(e for e, owner in enumerate(self.owners)
                                               if owner == rank))
                               for rank in range(self.world_size))
        self._lock = Lock()

    def _load(self, ids: tuple[int, ...]) -> ExpertWeights:
        weights = self.store.load(ids)
        expected13 = (len(ids), 2 * self.store.intermediate_size, self.store.hidden_size)
        expected2 = (len(ids), self.store.hidden_size, self.store.intermediate_size)
        if (weights.expert_ids != ids or weights.w13.shape != expected13
                or weights.w2.shape != expected2
                or weights.w13.dtype != self.store.dtype
                or weights.w2.dtype != self.store.dtype):
            raise ValueError("loaded weights do not match the requested expert layout")
        return weights

    def forward(self, batches: Sequence[RoutingBatch], *,
                shared_expert: Callable[[np.ndarray], np.ndarray] | None = None,
                routed_scale: float = 1.0) -> PrefillResult:
        """Return outputs, balanced counts, assignment metadata, and transfer sizes.

        The sequence includes empty ranks. All top-k slots count toward the
        load, and computation never rounds the assignment intervals to blocks.
        Missing weights are awaited explicitly; a failed load aborts the step.
        Accumulation uses at least float32. No performance logging is emitted.
        """
        with self._lock:
            return self._forward(tuple(batches), shared_expert, routed_scale)

    def _forward(self, batches: tuple[RoutingBatch, ...],
                 shared_expert: Callable[[np.ndarray], np.ndarray] | None,
                 routed_scale: float) -> PrefillResult:
        if len(batches) != self.world_size:
            raise ValueError("one batch is required for every source rank")
        if any(batch.hidden.shape[1] != self.store.hidden_size for batch in batches):
            raise ValueError("hidden states must match the layer's hidden dimension")
        if not math.isfinite(routed_scale):
            raise ValueError("routed_scale must be finite")
        rank_counts = routing_counts(batches, self.store.num_experts)
        counts = tuple(map(sum, zip(*rank_counts)))
        plan = balanced_plan(counts, self.world_size, self.owners)
        missing_ids = tuple(plan.missing_experts(rank) for rank in range(self.world_size))
        dtype = np.result_type(np.float32, self.store.dtype,
                               *(batch.hidden.dtype for batch in batches))
        transferred_bytes = [0] * self.world_size

        # Separate storage and completion handles for each call prevent stale
        # mappings, bounded-slot overflow, and cross-request event reuse.
        workers = max(1, sum(bool(ids) for ids in missing_ids))
        with ThreadPoolExecutor(max_workers=workers) as transfers:
            pending = {rank: transfers.submit(self._load, ids)
                       for rank, ids in enumerate(missing_ids) if ids}
            packed = dispatch(batches, plan)
            shared = []
            if shared_expert is not None:
                for batch in batches:
                    value = np.asarray(shared_expert(batch.hidden))
                    if (value.shape != batch.hidden.shape or value.dtype.kind != "f"
                            or not np.isfinite(value).all()):
                        raise ValueError("shared expert must return finite [tokens, hidden] values")
                    shared.append(value)

            results = [np.zeros(payload.hidden.shape, dtype=dtype) for payload in packed]
            for rank, payload in enumerate(packed):
                _compute(payload, self._resident[rank], results[rank])
            for rank, future in pending.items():
                weights = future.result()  # Completion is required before use.
                transferred_bytes[rank] = weights.nbytes
                _compute(packed[rank], weights, results[rank])

        routed = combine(batches, packed, results)
        outputs = tuple(routed_scale * result + (shared[rank] if shared else 0)
                        for rank, result in enumerate(routed))
        return PrefillResult(outputs, plan, packed, missing_ids,
                             tuple(transferred_bytes))
