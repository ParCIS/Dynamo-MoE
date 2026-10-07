# SPDX-License-Identifier: Apache-2.0
"""Device-free invariants and numerical checks against a direct MoE equation."""

import dataclasses
import random
import unittest
from collections import Counter
from threading import Event
from unittest.mock import patch

import numpy as np

from dynamo_moe import (
    PrefillLayer, RoutingBatch, RoutingPlan, SingleLayerExpertStore,
    balanced_plan, contiguous_owners,
)
from dynamo_moe.routing import combine, dispatch, routing_counts
import dynamo_moe.runtime as runtime_module


def direct_moe(batch, w13, w2):
    """Independent scalar-route oracle: no placement, dispatch, or migration."""
    output = np.zeros(batch.hidden.shape, dtype=np.float64)
    width = w2.shape[-1]
    for token, hidden in enumerate(batch.hidden):
        for slot, expert in enumerate(batch.expert_ids[token]):
            gate = w13[expert, :width] @ hidden
            up = w13[expert, width:] @ hidden
            activated = gate * (0.5 + 0.5 * np.tanh(gate / 2)) * up
            output[token] += batch.scores[token, slot] * (w2[expert] @ activated)
    return output


class PlanTests(unittest.TestCase):
    def test_conservation_balance_and_determinism(self):
        rng = random.Random(41)
        for _ in range(300):
            ranks, experts = rng.randint(1, 12), rng.randint(1, 35)
            counts = [rng.randrange(200) if rng.random() < 0.6 else 0
                      for _ in range(experts)]
            owners = [rng.randrange(ranks) for _ in counts]
            plan = balanced_plan(counts, ranks, owners)
            self.assertEqual(tuple(map(sum, zip(*plan.allocation))), tuple(counts))
            self.assertEqual(sum(plan.loads), sum(counts))
            self.assertLessEqual(max(plan.loads) - min(plan.loads), 1)
            self.assertEqual(plan, balanced_plan(counts, ranks, owners))
            for expert, count in enumerate(counts):
                end = 0
                for rank in range(ranks):
                    start, next_end = plan.intervals(rank)[expert]
                    self.assertEqual(start, end)
                    end = next_end
                self.assertEqual(end, count)

    def test_cold_experts_stay_local_before_hot_overflow(self):
        plan = balanced_plan([1, 9, 0, 0], 2)
        self.assertEqual(plan.allocation, ((1, 4, 0, 0), (0, 5, 0, 0)))
        self.assertEqual(plan.missing_experts(0), ())
        self.assertEqual(plan.missing_experts(1), (1,))

    def test_integer_counts_beyond_float_precision(self):
        counts = [2**55 + 3, 2**54 + 5, 0]
        plan = balanced_plan(counts, 3)
        self.assertEqual(sum(plan.loads), sum(counts))
        self.assertEqual(tuple(map(sum, zip(*plan.allocation))), tuple(counts))

    def test_empty_and_nondivisible_topology(self):
        self.assertEqual(contiguous_owners(5, 3), (0, 0, 1, 1, 2))
        self.assertEqual(contiguous_owners(2, 4), (0, 1))
        self.assertEqual(balanced_plan([0, 0], 4).loads, (0, 0, 0, 0))
        self.assertEqual(balanced_plan([1], 4).loads, (1, 0, 0, 0))

    def test_occurrence_boundaries_across_sources(self):
        sequences = [[0] * 31 + [1] * 2, [0] * 6, [1] * 5]
        rows = [[sequence.count(e) for e in range(2)] for sequence in sequences]
        plan = balanced_plan([37, 7], 3)
        observed = [[0, 0] for _ in sequences]
        for rank, sequence in enumerate(sequences):
            destinations = plan.destinations(sequence, rank=rank, rank_counts=rows)
            for expert, destination in zip(sequence, destinations):
                observed[destination][expert] += 1
        self.assertEqual(tuple(map(tuple, observed)), plan.allocation)

    def test_invalid_inputs_are_rejected(self):
        for counts, ranks in [([], 2), ([-1], 2), ([1.5], 1), ([True], 1), ([1], 0)]:
            with self.subTest(counts=counts, ranks=ranks), self.assertRaises(ValueError):
                balanced_plan(counts, ranks)
        with self.assertRaises(ValueError):
            balanced_plan([1], 2, [2])
        with self.assertRaises(ValueError):
            RoutingPlan((2,), (0,), (1,), ((1,),))
        plan = balanced_plan([2], 2)
        with self.assertRaises(ValueError):
            plan.destinations([], rank=0, rank_counts=[[1], [1]])
        with self.assertRaises(ValueError):
            plan.destinations([0, 0], rank=0, rank_counts=[[1], [1]])
        with self.assertRaises(ValueError):
            plan.destinations([0], rank=-1, rank_counts=[[1], [1]])


class RoutingTests(unittest.TestCase):
    def test_dispatch_combine_preserve_duplicate_topk_slots(self):
        batches = (
            RoutingBatch(np.arange(12.0).reshape(4, 3),
                         np.array([[0, 0], [2, 0], [1, 1], [0, 1]]),
                         np.array([[0.3, 0.2], [0.1, 0.9], [0.7, 0.8], [0.4, 0.6]])),
            RoutingBatch(np.empty((0, 3)), np.empty((0, 2), dtype=int), np.empty((0, 2))),
        )
        counts = tuple(map(sum, zip(*routing_counts(batches, 3))))
        packed = dispatch(batches, balanced_plan(counts, 2))
        outputs = combine(batches, packed, [payload.hidden for payload in packed])
        for batch, output in zip(batches, outputs):
            np.testing.assert_allclose(output, batch.hidden * batch.scores.sum(axis=1)[:, None])
        identities = [(int(source), int(token), int(slot)) for payload in packed
                      for source, token, slot in zip(payload.source_ranks,
                                                      payload.token_indices, payload.topk_slots)]
        self.assertTrue(all(count == 1 for count in Counter(identities).values()))
        self.assertEqual(len(identities), 8)

    def test_combine_rejects_missing_duplicate_and_changed_routes(self):
        batch = RoutingBatch(np.ones((2, 2)), np.zeros((2, 1), dtype=int), np.ones((2, 1)))
        packed = dispatch([batch], balanced_plan([2], 1))
        for name, value in [("token_indices", np.array([0, 0])),
                            ("scores", np.array([1.0, 2.0]))]:
            corrupt = dataclasses.replace(packed[0], **{name: value})
            with self.subTest(name=name), self.assertRaises(ValueError):
                combine([batch], [corrupt], [np.ones((2, 2))])
        with self.assertRaises(ValueError):
            combine([batch], packed, [np.ones((1, 2))])

    def test_softmax_router_ties_and_renormalization(self):
        hidden = np.ones((2, 3))
        logits = np.array([[1000.0, 1000.0, -1000.0], [0.0, 1.0, 2.0]])
        batch = RoutingBatch.from_logits(hidden, logits, 2)
        np.testing.assert_array_equal(batch.expert_ids, [[0, 1], [2, 1]])
        np.testing.assert_allclose(batch.scores.sum(axis=1), 1)
        raw = RoutingBatch.from_logits(hidden, logits, 1, renormalize=False)
        self.assertAlmostEqual(raw.scores[0, 0], 0.5)
        empty = RoutingBatch.from_logits(np.empty((0, 3)), np.empty((0, 4)), 2)
        self.assertEqual(empty.scores.shape, (0, 2))

    def test_bad_routes_and_nonfinite_scores(self):
        with self.assertRaises(ValueError):
            RoutingBatch(np.ones((1, 2)), np.array([[-1]]), np.ones((1, 1)))
        with self.assertRaises(ValueError):
            RoutingBatch(np.ones((1, 2)), np.array([[0]]), np.array([[np.nan]]))
        batch = RoutingBatch(np.ones((1, 2)), np.array([[3]]), np.ones((1, 1)))
        with self.assertRaises(ValueError):
            routing_counts([batch], 3)

    def test_mixed_integer_types_and_extreme_logits(self):
        limit = np.finfo(np.float64).max
        with np.errstate(all="raise"):
            batch = RoutingBatch.from_logits(np.ones((1, 2)), np.array([[limit, -limit]]), 1)
        np.testing.assert_array_equal(batch.scores, [[1.0]])
        other = RoutingBatch(np.ones((1, 2)), np.array([[1]], dtype=np.uint64), np.ones((1, 1)))
        packed = dispatch([batch, other], balanced_plan([1, 1], 2))
        outputs = combine([batch, other], packed, [payload.hidden for payload in packed])
        for output in outputs:
            np.testing.assert_array_equal(output, [[1.0, 1.0]])


class RuntimeTests(unittest.TestCase):
    def make_layer(self, experts=7, ranks=3, hidden=4, intermediate=5, owners=None):
        rng = np.random.default_rng(112)
        w13 = rng.normal(0, 0.3, (experts, intermediate * 2, hidden))
        w2 = rng.normal(0, 0.3, (experts, hidden, intermediate))
        store = SingleLayerExpertStore(w13, w2)
        return PrefillLayer(store, ranks, owners=owners), w13, w2

    def test_numerical_equivalence_for_skewed_unequal_batches(self):
        layer, w13, w2 = self.make_layer()
        rng = np.random.default_rng(9)
        for _ in range(20):
            batches = []
            for count in (19, 0, 7):
                ids = rng.choice(7, (count, 3), p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
                scores = rng.uniform(0.1, 1, (count, 3))
                batches.append(RoutingBatch(rng.normal(size=(count, 4)), ids, scores))
            result = layer.forward(batches)
            for batch, output in zip(batches, result.outputs):
                np.testing.assert_allclose(
                    output, direct_moe(batch, w13, w2), rtol=1e-12, atol=1e-12)
            self.assertLessEqual(max(result.plan.loads) - min(result.plan.loads), 1)
            for rank, ids in enumerate(result.missing_experts):
                self.assertTrue(all(layer.owners[e] != rank for e in ids))
                expected_bytes = len(ids) * (w13[0].nbytes + w2[0].nbytes)
                self.assertEqual(result.transferred_bytes[rank], expected_bytes)

    def test_more_than_ten_missing_experts(self):
        layer, w13, w2 = self.make_layer(experts=64, ranks=2)
        batches = [RoutingBatch(np.ones((32, 4)), np.arange(32).reshape(-1, 1), np.ones((32, 1))),
                   RoutingBatch(np.empty((0, 4)), np.empty((0, 1), dtype=int), np.empty((0, 1)))]
        result = layer.forward(batches)
        self.assertEqual(len(result.missing_experts[1]), 16)
        np.testing.assert_allclose(result.outputs[0], direct_moe(batches[0], w13, w2), atol=1e-12)

    def test_empty_single_rank_and_custom_ownership(self):
        for ranks, owners in [(1, None), (5, (4, 3))]:
            layer, w13, w2 = self.make_layer(experts=2, ranks=ranks, owners=owners)
            empty = [RoutingBatch(np.empty((0, 4)), np.empty((0, 2), dtype=int), np.empty((0, 2)))
                     for _ in range(ranks)]
            result = layer.forward(empty)
            self.assertEqual(result.plan.loads, (0,) * ranks)
            self.assertTrue(all(output.shape == (0, 4) for output in result.outputs))
            empty[0] = RoutingBatch(np.ones((1, 4)), np.array([[0, 1]]), np.array([[0.2, 0.8]]))
            result = layer.forward(empty)
            np.testing.assert_allclose(result.outputs[0], direct_moe(empty[0], w13, w2), atol=1e-12)

    def test_shared_expert_is_added_once_outside_routed_scaling(self):
        layer, w13, w2 = self.make_layer(ranks=1)
        batch = RoutingBatch(np.ones((3, 4)), np.array([[0, 1]] * 3), np.full((3, 2), 0.5))
        calls = []
        def shared(hidden):
            calls.append(hidden.shape)
            return hidden * 7
        result = layer.forward([batch], shared_expert=shared, routed_scale=2.5)
        np.testing.assert_allclose(
            result.outputs[0], direct_moe(batch, w13, w2) * 2.5 + batch.hidden * 7)
        self.assertEqual(calls, [(3, 4)])

    def test_failed_transfer_propagates_and_next_request_is_clean(self):
        layer, w13, w2 = self.make_layer(experts=2, ranks=2)
        batch = RoutingBatch(np.ones((8, 4)), np.zeros((8, 1), dtype=int), np.ones((8, 1)))
        empty = RoutingBatch(np.empty((0, 4)), np.empty((0, 1), dtype=int), np.empty((0, 1)))
        with patch.object(layer.store, "load", side_effect=RuntimeError("transfer failed")):
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                layer.forward([batch, empty])
        result = layer.forward([batch, empty])
        np.testing.assert_allclose(result.outputs[0], direct_moe(batch, w13, w2), atol=1e-12)

    def test_loaded_weights_are_independent_and_validated(self):
        layer, w13, w2 = self.make_layer(experts=2, ranks=2)
        weights = layer.store.load([1, 0])
        np.testing.assert_array_equal(weights.w13, w13[[1, 0]])
        self.assertFalse(np.shares_memory(weights.w13, w13))
        self.assertFalse(weights.w13.flags.writeable)
        self.assertTrue(w13.flags.writeable)
        self.assertEqual(layer.store.load([]).w13.shape, (0, 10, 4))
        for ids in ([0, 0], [2], [-1]):
            with self.assertRaises(ValueError):
                layer.store.load(ids)
        with self.assertRaises(ValueError):
            SingleLayerExpertStore(w13, w2[:, :, :-1])

    def test_resident_computation_precedes_transfer_completion(self):
        layer, w13, w2 = self.make_layer(experts=2, ranks=2)
        batch = RoutingBatch(np.ones((8, 4)), np.zeros((8, 1), dtype=int), np.ones((8, 1)))
        empty = RoutingBatch(np.empty((0, 4)), np.empty((0, 1), dtype=int), np.empty((0, 1)))
        loading, computed = Event(), Event()
        original_load, original_compute = layer.store.load, runtime_module._compute

        def delayed_load(ids):
            loading.set()
            if not computed.wait(timeout=5):
                raise RuntimeError("resident work was blocked by the transfer")
            return original_load(ids)

        def compute_then_release(*args):
            self.assertTrue(loading.wait(timeout=5))
            original_compute(*args)
            computed.set()

        with patch.object(layer.store, "load", side_effect=delayed_load), \
                patch.object(runtime_module, "_compute", side_effect=compute_then_release):
            result = layer.forward([batch, empty])
        np.testing.assert_allclose(result.outputs[0], direct_moe(batch, w13, w2), atol=1e-12)

    def test_layer_instances_do_not_share_weights_or_routing_state(self):
        first, w13, w2 = self.make_layer(experts=2, ranks=2)
        second = PrefillLayer(SingleLayerExpertStore(w13 * 2, w2 * 3), 2)
        batches = [RoutingBatch(np.ones((5, 4)), np.zeros((5, 1), dtype=int), np.ones((5, 1))),
                   RoutingBatch(np.empty((0, 4)), np.empty((0, 1), dtype=int), np.empty((0, 1)))]
        first_result = first.forward(batches)
        second_result = second.forward(batches)
        np.testing.assert_allclose(
            second_result.outputs[0], direct_moe(batches[0], w13 * 2, w2 * 3))
        np.testing.assert_array_equal(first.forward(batches).outputs[0], first_result.outputs[0])

    def test_half_precision_accumulates_in_float32(self):
        _, w13, w2 = self.make_layer(experts=2, ranks=2)
        store = SingleLayerExpertStore(w13.astype(np.float16), w2.astype(np.float16))
        layer = PrefillLayer(store, 2)
        batches = [RoutingBatch(np.ones((5, 4), dtype=np.float16), np.zeros((5, 1), dtype=int),
                                np.ones((5, 1), dtype=np.float16)),
                   RoutingBatch(np.empty((0, 4), dtype=np.float16), np.empty((0, 1), dtype=int),
                                np.empty((0, 1), dtype=np.float16))]
        result = layer.forward(batches)
        self.assertEqual(result.outputs[0].dtype, np.dtype(np.float32))
        expected = direct_moe(batches[0], w13.astype(np.float16).astype(np.float64),
                              w2.astype(np.float16).astype(np.float64))
        np.testing.assert_allclose(result.outputs[0], expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
