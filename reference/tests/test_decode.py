# SPDX-License-Identifier: Apache-2.0
import unittest
from concurrent.futures import Future
from unittest.mock import patch

import numpy as np

from dynamo_moe import DecodeLayer, Mode, RatePolicy, RoutingBatch, SingleLayerExpertStore
from test_prefill import direct_moe


class PolicyTests(unittest.TestCase):
    def test_hysteresis_and_equal_threshold(self):
        policy = RatePolicy(10, bits=2)
        self.assertEqual(policy.sample(0), Mode.EP)
        self.assertEqual(policy.sample(10), Mode.EP)
        self.assertEqual(policy.counter, 1)
        self.assertEqual(policy.sample(0), Mode.TP)
        self.assertEqual(policy.sample(20), Mode.TP)
        self.assertEqual(policy.sample(20), Mode.TP)
        self.assertEqual(policy.sample(20), Mode.EP)
        for _ in range(20):
            policy.sample(20)
        self.assertEqual(policy.counter, 3)

    def test_invalid_samples_do_not_change_state(self):
        policy = RatePolicy(10)
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                policy.sample(value)
            self.assertEqual(policy.counter, 4)


class DecodeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(917)
        self.w13 = rng.normal(size=(4, 12, 3)).astype(np.float32)
        self.w2 = rng.normal(size=(4, 3, 6)).astype(np.float32)
        self.store = SingleLayerExpertStore(self.w13, self.w2)
        self.batches = tuple(RoutingBatch.from_logits(
            rng.normal(size=(n, 3)), rng.normal(size=(n, 4)), 2)
            for n in (7, 0))

    def test_roundtrip_matches_direct_equation_and_shared_scaling(self):
        with DecodeLayer(self.store, 2, threshold=10, counter_bits=1) as layer:
            for rate, mode in ((20, Mode.EP), (0, Mode.TP), (20, Mode.EP)):
                layer.observe(rate)
                layer.synchronize()
                self.assertEqual(layer.mode, mode)
                outputs = layer.forward(self.batches, routed_scale=2.5,
                                        shared_expert=lambda x: x * 3)
                for output, batch in zip(outputs, self.batches):
                    expected = direct_moe(batch, self.w13, self.w2) * 2.5 + batch.hidden * 3
                    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)

    def test_gate_up_and_down_use_matching_tp_slices(self):
        parts = [self.store.load_tp(rank, 2) for rank in range(2)]
        gate = np.concatenate([p.w13[:, :3] for p in parts], axis=1)
        up = np.concatenate([p.w13[:, 3:] for p in parts], axis=1)
        np.testing.assert_array_equal(np.concatenate([gate, up], axis=1), self.w13)
        np.testing.assert_array_equal(np.concatenate([p.w2 for p in parts], axis=2), self.w2)
        self.assertFalse(np.shares_memory(parts[0].w13, self.w13))

    def test_unfinished_and_reversed_transfer_keep_active_layout(self):
        with DecodeLayer(self.store, 2, threshold=10, counter_bits=1) as layer:
            pending = Future()
            with patch.object(layer._transfers, "submit", return_value=pending):
                layer.observe(0)
                self.assertEqual(layer.mode, Mode.EP)
                layer.forward(self.batches)
                layer.observe(20)
                pending.set_result(layer._prepare(Mode.TP))
                layer.synchronize()
                self.assertEqual(layer.mode, Mode.EP)

    def test_failed_transfer_keeps_weights_and_can_retry(self):
        with DecodeLayer(self.store, 2, threshold=10, counter_bits=1) as layer:
            old = layer._active
            with patch.object(self.store, "load_tp", side_effect=OSError("read failed")):
                layer.observe(0)
                with self.assertRaisesRegex(OSError, "read failed"):
                    layer.synchronize()
            self.assertIs(layer._active, old)
            self.assertEqual(layer.mode, Mode.EP)
            layer.synchronize()
            self.assertEqual(layer.mode, Mode.TP)

    def test_independent_layers_and_close(self):
        with DecodeLayer(self.store, 2, threshold=10, counter_bits=1) as first:
            with DecodeLayer(self.store, 2, threshold=10) as second:
                first.observe(0)
                first.synchronize()
                self.assertEqual(second.mode, Mode.EP)
                self.assertFalse(np.shares_memory(first._active[0].w13,
                                                  second._active[0].w13))
        with self.assertRaises(RuntimeError):
            first.forward(self.batches)

    def test_invalid_dimension_and_expert_id(self):
        with self.assertRaises(ValueError):
            DecodeLayer(self.store, 3, threshold=1)
        bad = RoutingBatch(np.ones((1, 3)), np.array([[4]]), np.ones((1, 1)))
        with DecodeLayer(self.store, 2, threshold=1) as layer:
            with self.assertRaises(ValueError):
                layer.forward((bad, self.batches[1]))


if __name__ == "__main__":
    unittest.main()
