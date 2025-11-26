import torch
import numpy as np

class PixelEnv:
    """
    A simple environment that generates image-like observations.
    The environment consists of a 2D canvas where a shape (e.g., a square) moves.
    """
    def __init__(self, size=64, shape_size=5, num_actions=4):
        self.size = size
        self.shape_size = shape_size
        self.num_actions = num_actions
        self.pos = np.array([size // 2, size // 2])
        self.action_map = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),   # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1]),   # Right
        }

    def _generate_observation(self):
        """
        Generates a 1xHxW tensor representing the current state.
        The image is black with a white square at the agent's position.
        """
        canvas = torch.zeros(1, self.size, self.size)
        x_start = self.pos[0] - self.shape_size // 2
        x_end = x_start + self.shape_size
        y_start = self.pos[1] - self.shape_size // 2
        y_end = y_start + self.shape_size

        # Clamp to bounds
        x_start, x_end = max(0, x_start), min(self.size, x_end)
        y_start, y_end = max(0, y_start), min(self.size, y_end)

        canvas[:, x_start:x_end, y_start:y_end] = 1.0
        return canvas

    def reset(self):
        """
        Resets the environment to a random position.
        """
        self.pos = np.random.randint(
            self.shape_size // 2,
            self.size - self.shape_size // 2,
            size=2
        )
        return self._generate_observation()

    def step(self, action):
        """
        Takes an action and updates the position.
        Returns the new observation.
        """
        if action in self.action_map:
            self.pos += self.action_map[action]

        # Clamp position to be within the canvas boundaries
        self.pos = np.clip(
            self.pos,
            self.shape_size // 2,
            self.size - 1 - self.shape_size // 2
        )
        
        return self._generate_observation()

    def get_num_actions(self):
        return self.num_actions
