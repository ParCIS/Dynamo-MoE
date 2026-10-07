# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamo-MoE execution using vLLM routing and fused expert kernels.

The normal FusedMoE path remains unchanged when this extension is disabled.
Attention-TP replicas contribute tokens only from their TP leader. Nonleaders
return zeros when the model owns the final TP reduction, preserving vLLM's
shared-expert and routed-scaling contracts without duplicate contributions.
"""

import torch

from vllm.distributed import get_tp_group

from .checkpoint import stage
from .planning import balanced_plan
from .policy import Mode


class LayerExecution:
    def __init__(self, layer, controller):
        self.layer = layer
        self.controller = controller
        self.comm = controller.comm
        self.mode = Mode.EP
        quotient = layer.global_num_experts // self.comm.size
        # Keep the existing vLLM map: all remainder experts belong to the last rank.
        self.owners = tuple(min(e // quotient, self.comm.size - 1) if quotient else
                            self.comm.size - 1 for e in range(layer.global_num_experts))
        self.resident_ids = tuple(e for e, owner in enumerate(self.owners)
                                  if owner == self.comm.rank)
        self.resident_map = self._map(self.resident_ids)

    def _map(self, ids: tuple[int, ...]) -> torch.Tensor:
        mapping = torch.full((self.layer.global_num_experts,), -1,
                             dtype=torch.int32, device=self.comm.device)
        if ids:
            mapping[list(ids)] = torch.arange(len(ids), dtype=torch.int32,
                                              device=self.comm.device)
        return mapping

    def _select_experts(self, hidden, logits):
        layer = self.layer
        return layer.select_experts(
            hidden_states=hidden, router_logits=logits, top_k=layer.top_k,
            use_grouped_topk=layer.use_grouped_topk, renormalize=layer.renormalize,
            topk_group=layer.topk_group, num_expert_group=layer.num_expert_group,
            custom_routing_function=layer.custom_routing_function,
            scoring_func=layer.scoring_func,
            e_score_correction_bias=layer.e_score_correction_bias,
            indices_type=torch.int32, enable_eplb=False)

    def _fused(self, hidden, scores, ids, w13, w2, expert_map=None):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        if hidden.shape[0] == 0:
            return torch.zeros_like(hidden)
        return fused_experts(
            hidden_states=hidden.contiguous(), w1=w13, w2=w2,
            topk_weights=scores.contiguous(), topk_ids=ids.contiguous(),
            inplace=False, activation=self.layer.activation,
            global_num_experts=self.layer.global_num_experts, expert_map=expert_map)

    def _expert_subset(self, hidden, ids, w13, w2, mapping):
        output = torch.zeros_like(hidden)
        selected = torch.nonzero(mapping[ids.long()] >= 0, as_tuple=True)[0]
        if selected.numel():
            scores = torch.ones((selected.numel(), 1), dtype=torch.float32,
                                 device=hidden.device)
            values = self._fused(hidden[selected], scores, ids[selected, None],
                                  w13, w2, mapping)
            output.index_copy_(0, selected, values)
        return output

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if self is self.controller.layers[0]:
            self.controller.begin_step()
        self.controller.check_stream()
        tp = get_tp_group()
        is_source = tp.rank_in_group == 0
        hidden = hidden_states if is_source else hidden_states[:0]
        logits = router_logits if is_source else router_logits[:0]
        if len(hidden):
            scores, ids = self._select_experts(hidden, logits)
        else:
            ids = torch.empty((0, self.layer.top_k), dtype=torch.int32, device=hidden.device)
            scores = torch.empty_like(ids, dtype=torch.float32)
        if self.mode == Mode.TP:
            local_output = self._forward_tp(hidden, scores, ids)
        else:
            local_output = self._forward_ep(hidden, scores, ids)
        output = local_output if is_source else torch.zeros_like(hidden_states)
        if self.layer.reduce_results and tp.world_size > 1:
            output = tp.broadcast(output, src=0)
        return output

    def _forward_ep(self, hidden, scores, ids):
        layer = self.layer
        flat_ids = ids.flatten().long()
        local_counts = torch.bincount(flat_ids, minlength=layer.global_num_experts)
        rank_counts = [row.cpu().tolist() for row in self.comm.gather(local_counts)]
        counts = tuple(map(sum, zip(*rank_counts)))
        if not sum(counts):
            return torch.zeros_like(hidden)
        missing = ()
        if self.controller.settings.phase == "prefill":
            plan = balanced_plan(counts, self.comm.size, self.owners)
            destinations = plan.destinations(flat_ids.cpu().tolist(), rank=self.comm.rank,
                                             rank_counts=rank_counts)
            missing = plan.missing_experts(self.comm.rank)
        else:
            destinations = tuple(self.owners[e] for e in flat_ids.cpu().tolist())

        ticket = None
        error = ""
        try:
            if missing:
                host = self.controller.checkpoint.load(layer, missing)
                ticket = stage(host, hidden.device, self.controller.transfer_stream)
        except Exception as exc:
            error = f"{layer.layer_name}: {exc}"
        self.comm.check(error)

        destination = torch.tensor(destinations, dtype=torch.int64, device=hidden.device)
        order = torch.argsort(destination, stable=True)
        send_sizes = torch.bincount(destination, minlength=self.comm.size).cpu().tolist()
        receive_sizes = self.comm.exchange_sizes(send_sizes)
        token_indices = torch.div(order, layer.top_k, rounding_mode="floor")
        received = self.comm.exchange(hidden[token_indices], send_sizes, receive_sizes)
        received_ids = self.comm.exchange(flat_ids[order].int(), send_sizes, receive_sizes)

        result = self._expert_subset(received, received_ids, layer.w13_weight,
                                     layer.w2_weight, self.resident_map)
        if ticket is not None:
            ticket.wait()
            result.add_(self._expert_subset(received, received_ids, ticket.w13,
                                            ticket.w2, self._map(missing)))
        returned = self.comm.exchange(result, receive_sizes, send_sizes)
        weighted = returned.float() * scores.flatten()[order].float().unsqueeze(1)
        output = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
        output.index_add_(0, token_indices, weighted)
        return output.to(hidden.dtype)

    def _forward_tp(self, hidden, scores, ids):
        sizes = self.comm.sizes(len(hidden))
        if not sum(sizes):
            return torch.zeros_like(hidden)
        all_hidden = self.comm.gather_variable(hidden, sizes)
        all_scores = self.comm.gather_variable(scores.float(), sizes)
        all_ids = self.comm.gather_variable(ids.int(), sizes)
        # Each rank holds matched intermediate-dimension shards of every expert.
        partial = self._fused(all_hidden, all_scores, all_ids, self.layer.w13_weight,
                              self.layer.w2_weight).float()
        self.comm.sum(partial)
        start = sum(sizes[:self.comm.rank])
        return partial[start:start + sizes[self.comm.rank]].to(hidden.dtype)
