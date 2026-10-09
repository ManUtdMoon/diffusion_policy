import math
import unittest

import torch

from zprl.model.online import BatchedLinear


class BatchedLinearInitializationTest(unittest.TestCase):
    def test_rectangular_layers_use_input_width_for_initialization(self):
        for in_features, out_features in ((541, 512), (2104, 512), (512, 1), (8, 128)):
            for bias in (False, True):
                with self.subTest(in_features=in_features, out_features=out_features, bias=bias):
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(42)
                        layer = BatchedLinear(3, in_features, out_features, bias=bias)
                    bound = 1 / math.sqrt(in_features)
                    self.assertLessEqual(layer.weight.detach().abs().max().item(), bound)
                    expected_std = bound / math.sqrt(3)
                    self.assertAlmostEqual(layer.weight.detach().std().item(), expected_std,
                                           delta=0.1 * expected_std)
                    if bias:
                        self.assertLessEqual(layer.bias.detach().abs().max().item(), bound)
                    else:
                        self.assertIsNone(layer.bias)


if __name__ == '__main__':
    unittest.main()
