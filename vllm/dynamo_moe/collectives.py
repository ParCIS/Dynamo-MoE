# SPDX-License-Identifier: Apache-2.0
"""Explicit variable-sized collectives on the existing vLLM EP process group."""

import torch
import torch.distributed as dist


class Collectives:
    def __init__(self, group):
        self.group = group
        self.rank = group.rank_in_group
        self.size = group.world_size
        self.device = group.device

    def gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        if self.size == 1:
            return [tensor]
        outputs = [torch.empty_like(tensor) for _ in range(self.size)]
        dist.all_gather(outputs, tensor.contiguous(), group=self.group.device_group)
        return outputs

    def sizes(self, count: int) -> list[int]:
        tensor = torch.tensor([count], dtype=torch.int64, device=self.device)
        return [int(t.item()) for t in self.gather(tensor)]

    def gather_variable(self, tensor: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        if self.size == 1 or not sum(sizes):
            return tensor
        padded = tensor.new_zeros((max(sizes), *tensor.shape[1:]))
        padded[:len(tensor)].copy_(tensor)
        return torch.cat([part[:size] for part, size in zip(self.gather(padded), sizes)])

    def exchange_sizes(self, counts: list[int]) -> list[int]:
        if self.size == 1:
            return counts
        send = torch.tensor(counts, dtype=torch.int64, device=self.device)
        receive = torch.empty_like(send)
        dist.all_to_all_single(receive, send, group=self.group.device_group)
        return receive.cpu().tolist()

    def exchange(self, tensor: torch.Tensor, sends: list[int], receives: list[int]):
        if self.size == 1:
            return tensor
        output = tensor.new_empty((sum(receives), *tensor.shape[1:]))
        dist.all_to_all_single(output, tensor.contiguous(),
                              output_split_sizes=receives, input_split_sizes=sends,
                              group=self.group.device_group)
        return output

    def sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.size > 1:
            dist.all_reduce(tensor, group=self.group.device_group)
        return tensor

    def control(self, values: list[float]) -> list[float]:
        """Broadcast one decision from the group's leader, including its clock."""
        value = torch.tensor(values, dtype=torch.float64, device="cpu")
        if self.size > 1:
            dist.broadcast(value, src=self.group.ranks[0], group=self.group.cpu_group)
        return value.tolist()

    def all_ready(self, ready: bool) -> bool:
        value = torch.tensor([int(ready)], dtype=torch.int32, device="cpu")
        if self.size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MIN, group=self.group.cpu_group)
        return bool(value.item())

    def agree(self, signature) -> None:
        """Check layer order and settings before entering per-layer collectives."""
        if self.size == 1:
            return
        signatures = [None] * self.size
        dist.all_gather_object(signatures, signature, group=self.group.cpu_group)
        if any(value != signatures[0] for value in signatures[1:]):
            raise ValueError("Dynamo-MoE settings and selected layers must agree across ranks")

    def check(self, error: str = "") -> None:
        """Make host preparation failures visible to every participating rank."""
        if self.all_ready(not error):
            return
        errors = [error]
        if self.size > 1:
            errors = [None] * self.size
            dist.all_gather_object(errors, error, group=self.group.cpu_group)
        raise RuntimeError("Dynamo-MoE preparation failed: " +
                           "; ".join(str(message) for message in errors if message))
