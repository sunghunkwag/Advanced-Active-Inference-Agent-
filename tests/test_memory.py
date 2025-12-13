import unittest
import pytest

torch = pytest.importorskip("torch")
from memory import EpisodicMemory

class TestEpisodicMemory(unittest.TestCase):

    def setUp(self):
        self.capacity = 100
        self.obs_shape = (1, 64, 64)
        self.action_dim = 1
        self.memory = EpisodicMemory(self.capacity, self.obs_shape, self.action_dim)

    def test_push(self):
        self.assertEqual(len(self.memory), 0)
        obs = torch.randn(*self.obs_shape)
        action = 0
        next_obs = torch.randn(*self.obs_shape)
        reward = 0.0
        self.memory.push(obs, action, next_obs, reward)
        self.assertEqual(len(self.memory), 1)

    def test_capacity(self):
        obs = torch.randn(*self.obs_shape)
        for i in range(self.capacity + 10):
            self.memory.push(obs, i, obs, 0.0)
        self.assertEqual(len(self.memory), self.capacity)

    def test_reset(self):
        obs = torch.randn(*self.obs_shape)
        self.memory.push(obs, 0, obs, 0.0)
        self.memory.reset()
        self.assertEqual(len(self.memory), 0)

    def test_sample_sequence(self):
        obs = torch.randn(*self.obs_shape)
        for i in range(20):
            self.memory.push(obs, i, obs, 0.0)

        sequences = self.memory.sample(batch_size=5, sequence_length=10)
        self.assertEqual(len(sequences), 5)
        self.assertEqual(len(sequences[0]), 10)
        # Check that the actions are sequential
        first_action_in_seq = sequences[0][0][1].item()
        second_action_in_seq = sequences[0][1][1].item()
        self.assertEqual(second_action_in_seq, first_action_in_seq + 1)

if __name__ == '__main__':
    unittest.main()
