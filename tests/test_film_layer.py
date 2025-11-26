import unittest
import torch
from film_layer import FiLMLayer

class TestFiLMLayer(unittest.TestCase):

    def test_film_layer_linear(self):
        context_dim = 10
        feature_dim = 20
        film_layer = FiLMLayer(context_dim, feature_dim)

        x = torch.ones(5, feature_dim) # batch_size=5
        context = torch.randn(5, context_dim)

        output = film_layer(x, context)
        self.assertEqual(x.shape, output.shape)
        # Check if the transformation was applied (output should not be all ones)
        self.assertFalse(torch.all(output.eq(x)))

    def test_film_layer_conv(self):
        context_dim = 10
        feature_dim = 32 # num_channels
        film_layer = FiLMLayer(context_dim, feature_dim)

        x = torch.ones(5, feature_dim, 8, 8) # batch_size=5, C=32, H=8, W=8
        context = torch.randn(5, context_dim)

        output = film_layer(x, context)
        self.assertEqual(x.shape, output.shape)
        self.assertFalse(torch.all(output.eq(x)))

if __name__ == '__main__':
    unittest.main()
