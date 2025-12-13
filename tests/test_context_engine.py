import unittest
import pytest

torch = pytest.importorskip("torch")
from context_engine import ContextInferenceEngine
from memory import EpisodicMemory

class TestContextInferenceEngine(unittest.TestCase):

    def setUp(self):
        self.obs_shape = (1, 64, 64)
        self.action_dim = 4
        self.context_dim = 16
        self.engine = ContextInferenceEngine(self.obs_shape, self.action_dim, self.context_dim)

    def test_forward_pass_shape(self):
        batch_size = 5
        seq_len = 10
        obs_dim = torch.prod(torch.tensor(self.obs_shape)).item()

        obs_seq = torch.randn(batch_size, seq_len, obs_dim)
        action_seq = torch.randint(0, self.action_dim, (batch_size, seq_len, 1))
        next_obs_seq = torch.randn(batch_size, seq_len, obs_dim)

        context = self.engine(obs_seq, action_seq, next_obs_seq)
        self.assertEqual(context.shape, (batch_size, self.context_dim))

    def test_infer_context_shape(self):
        memory = EpisodicMemory(100, self.obs_shape, 1)
        for i in range(20):
            action = i % self.action_dim
            memory.push(torch.randn(*self.obs_shape), action, torch.randn(*self.obs_shape), 0.0)

        context = self.engine.infer_context(memory, batch_size=4, sequence_length=15)
        # Should be averaged to (1, context_dim)
        self.assertEqual(context.shape, (1, self.context_dim))

if __name__ == '__main__':
    unittest.main()
