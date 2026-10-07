# SPDX-License-Identifier: Apache-2.0
"""Single-layer EP/TP switching with inspectable weight layouts and outputs.

All ranks run in one process. A future models one in-flight layer transfer;
its result is published as a complete layout before the next forward. No GPU
timing, serving performance, or network behavior is inferred from this model.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .policy import Mode, RatePolicy
from .prefill import RoutingPlan
from .routing import PackedTokens, combine, dispatch, routing_counts
from .runtime import PrefillLayer, _compute


class DecodeLayer(PrefillLayer):
    def __init__(self, store, world_size, *, threshold, counter_bits=3):
        super().__init__(store, world_size)
        if (store.num_experts % world_size or
                store.intermediate_size % world_size):
            raise ValueError("EP/TP switching requires divisible dimensions")
        self.policy = RatePolicy(threshold, counter_bits)
        self.mode = Mode.EP
        self._active = self._resident
        del self._resident
        self._pending = None
        self._transfers = ThreadPoolExecutor(max_workers=1)
        self._closed = False

    def _prepare(self, mode):
        if mode == Mode.TP:
            return tuple(self.store.load_tp(rank, self.world_size)
                         for rank in range(self.world_size))
        return tuple(self._load(tuple(e for e, owner in enumerate(self.owners)
                                     if owner == rank))
                     for rank in range(self.world_size))

    def observe(self, request_rate):
        """Consume one global request-rate sample per observation window."""
        with self._lock:
            self._check_open()
            target = self.policy.sample(request_rate)
            self._advance(False)
            return target

    def _check_open(self):
        if self._closed:
            raise RuntimeError("decode layer has been closed")

    def _advance(self, wait):
        if self._pending is not None:
            target, future = self._pending
            if not wait and not future.done():
                return
            self._pending = None
            weights = future.result()  # Failure leaves the active layout intact.
            if target == self.policy.target:
                self._active, self.mode = weights, target
        if self.mode != self.policy.target:
            target = self.policy.target
            self._pending = (target, self._transfers.submit(self._prepare, target))

    def synchronize(self):
        """Finish the requested switch; useful for deterministic analysis."""
        with self._lock:
            self._check_open()
            self._advance(False)
            while self._pending is not None:
                self._advance(True)

    def forward(self, batches, *, shared_expert=None, routed_scale=1.0):
        with self._lock:
            self._check_open()
            self._advance(False)
            batches = tuple(batches)
            if len(batches) != self.world_size or any(
                    b.hidden.shape[1] != self.store.hidden_size for b in batches):
                raise ValueError("one compatible batch is required per rank")
            if not math.isfinite(routed_scale):
                raise ValueError("routed_scale must be finite")
            rows = routing_counts(batches, self.store.num_experts)
            dtype = np.result_type(np.float32, self.store.dtype,
                                   *(b.hidden.dtype for b in batches))
            if self.mode == Mode.EP:
                counts = tuple(map(sum, zip(*rows)))
                allocation = tuple(tuple(count if self.owners[e] == r else 0
                                         for e, count in enumerate(counts))
                                   for r in range(self.world_size))
                plan = RoutingPlan(counts, self.owners,
                                   tuple(map(sum, allocation)), allocation)
                packed = dispatch(batches, plan)
                results = [np.zeros(p.hidden.shape, dtype=dtype) for p in packed]
                for payload, weights, output in zip(packed, self._active, results):
                    _compute(payload, weights, output)
                routed = combine(batches, packed, results)
            else:
                routed = []
                for batch in batches:
                    tokens, top_k = batch.expert_ids.shape
                    positions = np.repeat(np.arange(tokens), top_k)
                    payload = PackedTokens(batch.hidden[positions],
                                           np.zeros(len(positions), dtype=np.int64),
                                           positions, np.tile(np.arange(top_k), tokens),
                                           batch.expert_ids.ravel(), batch.scores.ravel())
                    total = np.zeros(payload.hidden.shape, dtype=dtype)
                    for weights in self._active:
                        partial = np.zeros_like(total)
                        _compute(payload, weights, partial)
                        total += partial
                    output = np.zeros(batch.hidden.shape, dtype=dtype)
                    np.add.at(output, positions, total * payload.scores[:, None])
                    routed.append(output)
            outputs = []
            for batch, output in zip(batches, routed):
                output = output * routed_scale
                if shared_expert is not None:
                    shared = np.asarray(shared_expert(batch.hidden))
                    if (shared.shape != output.shape or shared.dtype.kind != "f"
                            or not np.isfinite(shared).all()):
                        raise ValueError("shared expert must return finite compatible values")
                    output = output + shared
                outputs.append(output)
            return tuple(outputs)

    def close(self):
        with self._lock:
            self._transfers.shutdown(wait=True)
            self._pending = None
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
