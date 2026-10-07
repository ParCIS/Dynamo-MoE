# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in settings for the existing vLLM worker.

VLLM_DYNAMO_MOE_PHASE: off (default), prefill, or decode.
VLLM_DYNAMO_MOE_LAYERS: comma-separated FusedMoE prefixes; empty selects all.
VLLM_DYNAMO_MOE_RATE_THRESHOLD: admitted requests/second, required for decode.
VLLM_DYNAMO_MOE_INTERVAL: observation window in seconds (default 1).
VLLM_DYNAMO_MOE_COUNTER_BITS: hysteresis width (default 3).

Choosing one layer keeps the same runtime interfaces and scheduling semantics.
Thresholds are supplied explicitly because they depend on the serving workload.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    phase: str = "off"
    layers: tuple[str, ...] = ()
    rate_threshold: Optional[float] = None
    interval: float = 1.0
    counter_bits: int = 3

    def __post_init__(self):
        if self.phase not in ("off", "prefill", "decode"):
            raise ValueError("Dynamo-MoE phase must be off, prefill, or decode")
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("observation interval must be finite and positive")
        if (isinstance(self.counter_bits, bool)
                or not isinstance(self.counter_bits, int) or self.counter_bits < 1):
            raise ValueError("counter bits must be a positive integer")
        if self.phase == "decode" and self.rate_threshold is None:
            raise ValueError("decode requires VLLM_DYNAMO_MOE_RATE_THRESHOLD")
        if self.rate_threshold is not None and (
                not math.isfinite(self.rate_threshold) or self.rate_threshold < 0):
            raise ValueError("request-rate threshold must be finite and nonnegative")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("layer prefixes must be unique")

    @classmethod
    def from_env(cls):
        rate = os.getenv("VLLM_DYNAMO_MOE_RATE_THRESHOLD")
        return cls(
            phase=os.getenv("VLLM_DYNAMO_MOE_PHASE", "off"),
            layers=tuple(x.strip() for x in
                         os.getenv("VLLM_DYNAMO_MOE_LAYERS", "").split(",") if x.strip()),
            rate_threshold=None if rate is None else float(rate),
            interval=float(os.getenv("VLLM_DYNAMO_MOE_INTERVAL", "1")),
            counter_bits=int(os.getenv("VLLM_DYNAMO_MOE_COUNTER_BITS", "3")),
        )
