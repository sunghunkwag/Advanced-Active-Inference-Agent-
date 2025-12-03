import torch
from collections import deque
import random

class EpisodicMemory:
    """
    A memory module to store recent experiences (trajectories) within a single episode.
    This is used by the Context Inference Engine to infer the current task.
    """
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_dim = action_dim

        # Use deque for efficient appends and pops from both ends
        self.memory = deque(maxlen=capacity)

    def push(self, obs, action, next_obs, reward):
        """Saves a transition."""
        # Ensure action is a tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=torch.long)

        self.memory.append((obs, action, next_obs, torch.tensor([reward])))

    def sample(self, batch_size, sequence_length=None):
        """
        Sample a batch of transitions.
        If sequence_length is provided, it samples sequences of transitions.
        """
        if sequence_length is None:
            # Sample random individual transitions
            return random.sample(self.memory, batch_size)
        else:
            # Sample random sequences of transitions
            sequences = []
            indices = range(len(self.memory) - sequence_length + 1)
            sampled_indices = random.sample(indices, min(batch_size, len(indices)))

            for start_idx in sampled_indices:
                sequence = [self.memory[i] for i in range(start_idx, start_idx + sequence_length)]
                sequences.append(sequence)
            return sequences

    def reset(self):
        """Clears the memory, typically called at the start of a new task."""
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
