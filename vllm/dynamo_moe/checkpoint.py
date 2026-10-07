# SPDX-License-Identifier: Apache-2.0
"""Lazy expert loading from vLLM's already-resolved checkpoint files.

This replaces the original /dev/shm dump and global tensor cache. Only the
requested layer/layout is materialized in pinned host memory. A missing or
incompatible tensor is an error, never an uninitialized-weight fallback.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from safetensors import safe_open


@dataclass
class HostWeights:
    w13: torch.Tensor
    w2: torch.Tensor


class ExpertCheckpoint:
    def __init__(self, files: list[str]):
        self.tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
        for filename in sorted(set(files)):
            with safe_open(filename, framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    if ".experts." not in name or not name.endswith("_proj.weight"):
                        continue
                    if name in self.tensors:
                        raise ValueError(f"Duplicate expert tensor: {name}")
                    self.tensors[name] = (filename, tuple(reader.get_slice(name).get_shape()))

    def validate_layer(self, layer) -> None:
        width, hidden = layer.intermediate_size_per_partition, layer.hidden_size
        for expert in range(layer.global_num_experts):
            for projection, shape in (("gate", (width, hidden)),
                                      ("up", (width, hidden)),
                                      ("down", (hidden, width))):
                name = f"{layer.layer_name}.{expert}.{projection}_proj.weight"
                if name not in self.tensors:
                    raise ValueError(f"Missing expert checkpoint tensor: {name}")
                if self.tensors[name][1] != shape:
                    raise ValueError(f"Unexpected expert checkpoint shape: {name}")

    def _tensor(self, prefix: str, expert: int, projection: str) -> torch.Tensor:
        name = f"{prefix}.{expert}.{projection}_proj.weight"
        filename, _ = self.tensors[name]
        with safe_open(filename, framework="pt", device="cpu") as reader:
            value = reader.get_tensor(name)
        if not value.is_floating_point():
            raise ValueError(f"Quantized expert tensors are unsupported: {name}")
        return value

    def load(self, layer, expert_ids: tuple[int, ...], *,
             tp_rank: Optional[int] = None, tp_size: int = 1) -> HostWeights:
        """Copy whole EP experts or matched gate/up/down TP slices of one layer."""
        if len(set(expert_ids)) != len(expert_ids) or any(
                e < 0 or e >= layer.global_num_experts for e in expert_ids):
            raise ValueError("Invalid expert selection")
        width, hidden = layer.intermediate_size_per_partition, layer.hidden_size
        if tp_size < 1 or width % tp_size:
            raise ValueError("Intermediate size must be divisible by TP size")
        if tp_rank is not None and not 0 <= tp_rank < tp_size:
            raise ValueError("Invalid TP rank")
        local_width = width if tp_rank is None else width // tp_size
        start = 0 if tp_rank is None else tp_rank * local_width
        end = start + local_width
        w13 = torch.empty((len(expert_ids), 2 * local_width, hidden),
                          dtype=layer.params_dtype, device="cpu", pin_memory=True)
        w2 = torch.empty((len(expert_ids), hidden, local_width),
                         dtype=layer.params_dtype, device="cpu", pin_memory=True)
        for slot, expert in enumerate(expert_ids):
            # Gate and up are sliced separately; reshaping a full gate/up tensor
            # into a TP shape does not produce the correct partition.
            w13[slot, :local_width].copy_(
                self._tensor(layer.layer_name, expert, "gate")[start:end])
            w13[slot, local_width:].copy_(
                self._tensor(layer.layer_name, expert, "up")[start:end])
            w2[slot].copy_(self._tensor(layer.layer_name, expert, "down")[:, start:end])
        return HostWeights(w13, w2)


@dataclass
class Transfer:
    w13: torch.Tensor
    w2: torch.Tensor
    ready: torch.cuda.Event
    host: Optional[HostWeights]

    def wait(self) -> None:
        # Called only after resident work is enqueued. The host wait also makes
        # releasing pinned source memory safe without a process-global event list.
        self.ready.synchronize()
        stream = torch.cuda.current_stream(self.w13.device)
        self.w13.record_stream(stream)
        self.w2.record_stream(stream)
        self.host = None


def stage(host: HostWeights, device: torch.device, stream: torch.cuda.Stream,
          reuse: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
          last_use: Optional[torch.cuda.Event] = None) -> Transfer:
    """Copy into a private buffer, optionally reusing retired layer storage."""
    with torch.cuda.stream(stream):
        if last_use is not None:
            stream.wait_event(last_use)
        tensors = []
        for index, source in enumerate((host.w13, host.w2)):
            old = None if reuse is None else reuse[index]
            if (old is not None and old.numel() == source.numel()
                    and old.dtype == source.dtype and old.device == device):
                target = old.view(source.shape)
                target.record_stream(stream)
            else:
                target = torch.empty_like(source, device=device)
            target.copy_(source, non_blocking=True)
            tensors.append(target)
        ready = torch.cuda.Event()
        ready.record(stream)
    return Transfer(tensors[0], tensors[1], ready, host)
