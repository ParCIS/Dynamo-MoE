# SPDX-License-Identifier: Apache-2.0
"""Single-layer expert storage and explicit host-to-worker copy semantics.

NumPy arrays model tensor contents. ``load`` is the transfer boundary: a device
backend can replace it while preserving expert IDs, shapes, and completion
ordering. This module never stores other layers or creates global state.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .prefill import _integer


@dataclass(frozen=True)
class ExpertWeights:
    """One worker's packed experts, indexed by their global expert IDs."""

    expert_ids: tuple[int, ...]
    w13: np.ndarray
    w2: np.ndarray

    @property
    def nbytes(self) -> int:
        return self.w13.nbytes + self.w2.nbytes


class SingleLayerExpertStore:
    """Read one fully initialized layer without duplicating the full host pool.

    w13 has shape [experts, 2 * intermediate, hidden], with gate then up;
    w2 has shape [experts, hidden, intermediate]. The store borrows the input
    arrays. Their owner must not modify them while this store is in use.
    ``load`` returns independent read-only copies of only the requested experts.
    It is thread-safe while the borrowed host tensors remain immutable.
    """

    def __init__(self, w13: np.ndarray, w2: np.ndarray):
        w13, w2 = np.asarray(w13), np.asarray(w2)
        if w13.ndim != 3 or w2.ndim != 3:
            raise ValueError("expert weights must be three-dimensional")
        experts, doubled_intermediate, hidden = w13.shape
        if min(w13.shape) <= 0 or doubled_intermediate % 2:
            raise ValueError("w13 requires positive dimensions and paired gate/up weights")
        if w2.shape != (experts, hidden, doubled_intermediate // 2):
            raise ValueError("w13 and w2 shapes are incompatible")
        if w13.dtype.kind != "f" or w13.dtype != w2.dtype:
            raise ValueError("weights must have the same floating-point dtype")
        # A view protects this API without changing the caller's writeable flag.
        self._w13, self._w2 = w13.view(), w2.view()
        self._w13.flags.writeable = self._w2.flags.writeable = False
        self.num_experts = experts
        self.hidden_size = hidden
        self.intermediate_size = doubled_intermediate // 2
        self.dtype = w13.dtype

    def load(self, expert_ids: Sequence[int]) -> ExpertWeights:
        ids = tuple(_integer(e, "expert_id") for e in expert_ids)
        if len(ids) != len(set(ids)) or any(e >= self.num_experts for e in ids):
            raise ValueError("requested expert IDs must be unique and in range")
        indices = np.asarray(ids, dtype=np.intp)
        w13 = np.ascontiguousarray(self._w13[indices])
        w2 = np.ascontiguousarray(self._w2[indices])
        w13.flags.writeable = w2.flags.writeable = False
        return ExpertWeights(ids, w13, w2)

    def load_tp(self, rank: int, world_size: int) -> ExpertWeights:
        """Shard each projection along the same intermediate coordinates."""
        rank = _integer(rank, "rank")
        world_size = _integer(world_size, "world_size", 1)
        if rank >= world_size or self.intermediate_size % world_size:
            raise ValueError("TP requires a valid rank and divisible intermediate size")
        width = self.intermediate_size // world_size
        start, stop = rank * width, (rank + 1) * width
        full = self.intermediate_size
        w13 = np.concatenate((self._w13[:, start:stop],
                              self._w13[:, full + start:full + stop]), axis=1)
        w2 = self._w2[:, :, start:stop].copy()
        w13.flags.writeable = w2.flags.writeable = False
        return ExpertWeights(tuple(range(self.num_experts)), w13, w2)
