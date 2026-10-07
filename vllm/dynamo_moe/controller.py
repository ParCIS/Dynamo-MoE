# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refactored layer-wise EP/TP migration from the original GPUModelRunner.

The worker records admissions without collectives. At the first selected MoE
layer, all ranks (including DP dummy forwards) enter the same control step.
One transfer is in flight, and weight/mode publication happens only after a
group-wide readiness decision. Retired weights form the next transfer buffer.
"""

import time
from typing import Optional

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed import get_ep_group, get_tp_group

from .checkpoint import ExpertCheckpoint, Transfer, stage
from .collectives import Collectives
from .execution import LayerExecution
from .policy import Mode, RatePolicy
from .settings import Settings


class DynamoMoEController:
    def __init__(self, settings: Settings, checkpoint: ExpertCheckpoint, modules: list):
        self.settings = settings
        self.checkpoint = checkpoint
        self.comm = Collectives(get_ep_group())
        self.comm.agree((settings, tuple(
            (layer.layer_name, layer.global_num_experts, layer.hidden_size,
             layer.intermediate_size_per_partition, str(layer.params_dtype))
            for layer in modules)))
        self.compute_stream = torch.cuda.current_stream(self.comm.device)
        self.transfer_stream = torch.cuda.Stream(device=self.comm.device)
        self.layers = [LayerExecution(module, self) for module in modules]
        self.target = Mode.EP
        self.policy = (RatePolicy(settings.rate_threshold, settings.counter_bits)
                       if settings.phase == "decode" else None)
        self._arrivals = 0
        self._started = False
        self._last_sample: Optional[float] = None
        self._pending: Optional[tuple[LayerExecution, Mode, Transfer]] = None
        self._spare = None
        self._last_use = None
        for state in self.layers:
            state.layer.dynamo_moe = state

    def check_stream(self) -> None:
        if torch.cuda.current_stream(self.comm.device) != self.compute_stream:
            raise RuntimeError("Dynamo-MoE requires the vLLM worker's single inference stream")

    def record_requests(self, new_requests: list) -> None:
        """Count first scheduler admissions once per attention-TP replica set."""
        self._started = True
        if get_tp_group().rank_in_group == 0:
            self._arrivals += len(new_requests)

    def _sample_request_rate(self) -> None:
        now = time.monotonic()
        if self._last_sample is None:
            self._last_sample = now
        elapsed = now - self._last_sample
        due, elapsed = self.comm.control([float(elapsed >= self.settings.interval), elapsed])
        if not due:
            return
        arrivals = torch.tensor([self._arrivals], dtype=torch.int64, device="cpu")
        if self.comm.size > 1:
            dist.all_reduce(arrivals, group=self.comm.group.cpu_group)
        self._arrivals = 0
        self._last_sample = now
        if self.comm.rank == 0:
            self.target = self.policy.sample(arrivals.item() / elapsed)
        choice = self.comm.control([float(self.target == Mode.TP)])[0]
        self.target = Mode.TP if choice else Mode.EP

    def begin_step(self) -> None:
        self.check_stream()
        if self.settings.phase != "decode":
            return
        started = torch.tensor([int(self._started)], dtype=torch.int32, device="cpu")
        if self.comm.size > 1:
            dist.all_reduce(started, op=dist.ReduceOp.MAX, group=self.comm.group.cpu_group)
        if not started.item():
            return  # Profiling before the first admitted request keeps the initial layout.
        self._started = True
        self._sample_request_rate()
        if not self._maybe_sync_weight_transfer():
            return
        self._maybe_start_weight_transfer()

    def _maybe_sync_weight_transfer(self) -> bool:
        if self._pending is None:
            return True
        state, target, ticket = self._pending
        if not self.comm.all_ready(ticket.ready.query()):
            return False
        ticket.host = None
        self._pending = None
        if target != self.target:
            self._spare = (ticket.w13, ticket.w2)
            self._last_use = None
            return True
        layer = state.layer
        retired = (layer.w13_weight.data, layer.w2_weight.data)
        last_use = torch.cuda.Event()
        last_use.record(self.compute_stream)
        ticket.w13.record_stream(self.compute_stream)
        ticket.w2.record_stream(self.compute_stream)
        # The worker runs this before model execution on a single host thread.
        # Publishing both tensors and the flag here makes the transition atomic
        # with respect to every forward of this layer.
        layer.w13_weight.data = ticket.w13
        layer.w2_weight.data = ticket.w2
        state.mode = target
        self._spare, self._last_use = retired, last_use
        return True

    def _maybe_start_weight_transfer(self) -> None:
        state = next((item for item in self.layers if item.mode != self.target), None)
        if state is None:
            return
        layer, target = state.layer, self.target
        ticket, error = None, ""
        try:
            if target == Mode.TP:
                ids = tuple(range(layer.global_num_experts))
                host = self.checkpoint.load(layer, ids, tp_rank=self.comm.rank,
                                            tp_size=self.comm.size)
            else:
                host = self.checkpoint.load(layer, state.resident_ids)
            ticket = stage(host, self.comm.device, self.transfer_stream,
                           self._spare, self._last_use)
        except Exception as exc:
            error = f"{layer.layer_name}: {exc}"
        self.comm.check(error)
        self._pending = (state, target, ticket)
        self._spare = self._last_use = None


def attach(model, settings: Settings, files: list[str], config: VllmConfig) -> None:
    """Connect reviewed vLLM layers to the existing loader and worker hooks."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    if not envs.VLLM_USE_V1:
        raise ValueError("Dynamo-MoE requires the vLLM V1 worker")
    if config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("Dynamo-MoE supports a single pipeline stage")
    if (not config.model_config.enforce_eager or
            config.compilation_config.level != CompilationLevel.NO_COMPILATION):
        raise ValueError("Dynamo-MoE requires enforce_eager and compilation level 0")
    if config.lora_config is not None or config.speculative_config is not None:
        raise ValueError("Dynamo-MoE does not support LoRA or speculative decoding")
    architectures = config.model_config.hf_config.architectures or []
    supported = {"DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "Qwen3MoeForCausalLM"}
    if not any(name in supported for name in architectures):
        raise ValueError("Dynamo-MoE supports the DeepSeek-V2/V3 and Qwen3-MoE model adapters")
    modules = [module for module in model.modules() if isinstance(module, FusedMoE)
               and (not settings.layers or module.layer_name in settings.layers)]
    if not modules:
        raise ValueError("No requested FusedMoE layer exists on this worker")
    missing = set(settings.layers) - {layer.layer_name for layer in modules}
    if missing:
        raise ValueError(f"Unknown requested FusedMoE layers: {sorted(missing)}")
    checkpoint = ExpertCheckpoint(files)
    group = get_ep_group()
    for layer in modules:
        if (layer.quant_config is not None or layer.enable_eplb
                or layer.apply_router_weight_on_input):
            raise ValueError("Dynamo-MoE requires unquantized experts, no EPLB, and output routing weights")
        if layer.tp_size != 1 or layer.ep_size != group.world_size:
            raise ValueError("Dynamo-MoE must start with vLLM expert parallelism enabled")
        if layer.ep_rank != group.rank_in_group:
            raise ValueError("FusedMoE and the EP group disagree on rank ordering")
        parallel = layer.moe_parallel_config
        if (parallel.use_pplx_kernels or parallel.use_deepep_ht_kernels
                or parallel.use_deepep_ll_kernels or parallel.use_flashinfer_cutlass_kernels):
            raise ValueError("Dynamo-MoE uses the standard vLLM fused expert kernel")
        if layer.w13_weight.device.type != "cuda":
            raise ValueError("Dynamo-MoE requires CUDA expert weights")
        if settings.phase == "decode" and (
                layer.global_num_experts % group.world_size or
                layer.intermediate_size_per_partition % group.world_size):
            raise ValueError("EP/TP switching requires divisible expert and intermediate dimensions")
        checkpoint.validate_layer(layer)
    model.dynamo_moe_controller = DynamoMoEController(settings, checkpoint, modules)
