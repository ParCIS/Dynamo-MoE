# SPDX-License-Identifier: Apache-2.0
"""Request-rate hysteresis, independent of tensors and process groups."""

import math
from dataclasses import dataclass, field
from enum import Enum


class Mode(str, Enum):
    EP = "ep"
    TP = "tp"


@dataclass
class RatePolicy:
    """One update per completed observation window; equality holds the counter."""

    threshold: float
    bits: int = 3
    target: Mode = Mode.EP
    counter: int = field(init=False)

    def __post_init__(self):
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("request-rate threshold must be finite and nonnegative")
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or self.bits < 1:
            raise ValueError("counter bits must be a positive integer")
        self.counter = 1 << (self.bits - 1)
        self.target = Mode(self.target)

    @property
    def maximum(self) -> int:
        return (1 << self.bits) - 1

    def sample(self, request_rate: float) -> Mode:
        if not math.isfinite(request_rate) or request_rate < 0:
            raise ValueError("request rate must be finite and nonnegative")
        if request_rate > self.threshold:
            self.counter = min(self.maximum, self.counter + 1)
        elif request_rate < self.threshold:
            self.counter = max(0, self.counter - 1)
        if self.counter == 0:
            self.target = Mode.TP
        elif self.counter == self.maximum:
            self.target = Mode.EP
        return self.target
