# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of production planning, loading and transition code.

Production modules are evaluated without their hardware imports. Small tensor
and event doubles exercise actual checkpoint slicing and publication methods;
these tests do not claim to validate CUDA kernels or distributed collectives.
"""
import __future__
import ast
import dataclasses
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vllm" / "dynamo_moe"


def source_module(name, namespace):
    tree = ast.parse((SOURCE / (name + ".py")).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(SOURCE / name), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return SimpleNamespace(**namespace)


policy = source_module("policy", dict(math=math, dataclass=dataclasses.dataclass,
                                    field=dataclasses.field,
                                    Enum=__import__("enum").Enum))
settings = source_module("settings", dict(math=math, os=os, dataclass=dataclasses.dataclass))
planning = source_module("planning", dict(dataclass=dataclasses.dataclass,
                                          index=__import__("operator").index,
                                          bisect_right=__import__("bisect").bisect_right))


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.streams = []

    def __getitem__(self, key):
        return Tensor(self.values[key])

    def copy_(self, other):
        self.values[:] = other.values

    def record_stream(self, stream):
        self.streams.append(stream)


class Event:
    def __init__(self, done=True):
        self.done = done
        self.stream = None

    def query(self):
        return self.done

    def record(self, stream):
        self.stream = stream


fake_torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event),
                             empty=lambda shape, dtype, **kw: Tensor(np.empty(shape, dtype)))
checkpoint = source_module("checkpoint", dict(torch=fake_torch,
                                               dataclass=dataclasses.dataclass))
controller_globals = dict(torch=fake_torch, Mode=policy.Mode)
controller = source_module("controller", controller_globals)


class ProductionTests(unittest.TestCase):
    def test_checkpoint_load_ep_and_tp_numerically(self):
        source = object.__new__(checkpoint.ExpertCheckpoint)
        arrays = {}
        rng = np.random.default_rng(3)
        for e in range(4):
            for projection, shape in (("gate", (6, 3)), ("up", (6, 3)), ("down", (3, 6))):
                arrays[e, projection] = Tensor(rng.normal(size=shape))
        source._tensor = lambda prefix, expert, projection: arrays[expert, projection]
        layer = SimpleNamespace(global_num_experts=4, intermediate_size_per_partition=6,
                                hidden_size=3, params_dtype=np.float64, layer_name="layer")
        full = source.load(layer, (0, 1, 2, 3))
        shards = [source.load(layer, (0, 1, 2, 3), tp_rank=r, tp_size=2) for r in range(2)]
        x = rng.normal(size=(5, 3))
        for expert in range(4):
            def mlp(weights):
                gate, up = np.split(x @ weights.w13.values[expert].T, 2, axis=1)
                return (gate * (0.5 + 0.5 * np.tanh(gate / 2)) * up) @ weights.w2.values[expert].T
            np.testing.assert_allclose(sum(mlp(w) for w in shards), mlp(full), atol=1e-12)
        subset = source.load(layer, (3, 1))
        np.testing.assert_array_equal(subset.w13.values, full.w13.values[[3, 1]])
        with self.assertRaises(ValueError):
            source.load(layer, (0,), tp_rank=0, tp_size=4)

    def make_controller(self):
        obj = object.__new__(controller.DynamoMoEController)
        obj.comm = SimpleNamespace(all_ready=lambda ready: ready, rank=1, size=2, device="cuda")
        obj.compute_stream = object()
        obj.transfer_stream = object()
        obj.target = policy.Mode.TP
        obj._spare = obj._last_use = None
        layer = SimpleNamespace(w13_weight=SimpleNamespace(data=Tensor([1])),
                                w2_weight=SimpleNamespace(data=Tensor([2])),
                                global_num_experts=4, layer_name="layer")
        state = SimpleNamespace(layer=layer, mode=policy.Mode.EP, resident_ids=(2, 3))
        ticket = SimpleNamespace(w13=Tensor([3]), w2=Tensor([4]), ready=Event(), host=object())
        obj.layers = [state]
        obj._pending = (state, policy.Mode.TP, ticket)
        return obj, state, ticket

    def test_switch_waits_for_every_rank(self):
        obj, state, ticket = self.make_controller()
        obj.comm.all_ready = lambda ready: False
        old = state.layer.w13_weight.data
        self.assertFalse(obj._maybe_sync_weight_transfer())
        self.assertIs(state.layer.w13_weight.data, old)
        self.assertEqual(state.mode, policy.Mode.EP)
        self.assertIsNotNone(ticket.host)
        obj.comm.all_ready = lambda ready: ready
        self.assertTrue(obj._maybe_sync_weight_transfer())
        self.assertIs(state.layer.w13_weight.data, ticket.w13)
        self.assertIs(state.layer.w2_weight.data, ticket.w2)
        self.assertIs(obj._spare[0], old)
        self.assertIs(obj._last_use.stream, obj.compute_stream)
        self.assertEqual(state.mode, policy.Mode.TP)

    def test_cancelled_transition_does_not_publish_weights(self):
        obj, state, ticket = self.make_controller()
        obj.target = policy.Mode.EP
        old = state.layer.w13_weight.data
        obj._maybe_sync_weight_transfer()
        self.assertIs(state.layer.w13_weight.data, old)
        self.assertEqual(state.mode, policy.Mode.EP)
        self.assertIs(obj._spare[0], ticket.w13)

    def test_stage_failure_leaves_active_layer_intact(self):
        obj, state, _ = self.make_controller()
        obj._pending = None
        old = state.layer.w13_weight.data
        obj.checkpoint = SimpleNamespace(load=lambda *a, **kw: object())
        def check(error):
            if error:
                raise RuntimeError(error)
        obj.comm.check = check
        def fail(*args):
            raise OSError("missing weights")
        with patch.dict(controller_globals, stage=fail):
            with self.assertRaisesRegex(RuntimeError, "missing weights"):
                obj._maybe_start_weight_transfer()
        self.assertIsNone(obj._pending)
        self.assertIs(state.layer.w13_weight.data, old)
        self.assertEqual(state.mode, policy.Mode.EP)

    def test_production_planner_conserves_routes_with_empty_ranks(self):
        rng = random.Random(112)
        for _ in range(100):
            ranks, experts = rng.randint(1, 8), rng.randint(1, 20)
            sequences = [[rng.randrange(experts) for _ in range(rng.randrange(50))]
                         for _ in range(ranks)]
            rows = [[seq.count(e) for e in range(experts)] for seq in sequences]
            counts = tuple(map(sum, zip(*rows)))
            plan = planning.balanced_plan(counts, ranks)
            actual = [[0] * experts for _ in range(ranks)]
            for rank, sequence in enumerate(sequences):
                targets = plan.destinations(sequence, rank=rank, rank_counts=rows)
                for expert, target in zip(sequence, targets):
                    actual[target][expert] += 1
            self.assertEqual(tuple(map(tuple, actual)), plan.allocation)
            self.assertLessEqual(max(plan.loads) - min(plan.loads), 1)

    def test_explicit_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.Settings.from_env().phase, "off")
        with self.assertRaises(ValueError):
            settings.Settings(phase="decode")
        with self.assertRaises(ValueError):
            settings.Settings(layers=("layer", "layer"))

    def test_vllm_router_signature_compatibility(self):
        tree = ast.parse((ROOT / "vllm/model_executor/layers/fused_moe/layer.py").read_text())
        select = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "select_experts")
        call = next(n for n in ast.walk(ast.parse((SOURCE / "execution.py").read_text()))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "select_experts")
        self.assertLessEqual({k.arg for k in call.keywords}, {a.arg for a in select.args.args})


if __name__ == "__main__":
    unittest.main()
