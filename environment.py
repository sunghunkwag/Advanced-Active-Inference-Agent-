import torch
import numpy as np

import random

class PixelEnv:
    """
    A simple environment that generates image-like observations.
    The environment consists of a 2D canvas where a shape (e.g., a square) moves.
    For meta-learning, the dynamics (action mappings) can be changed by setting a new task.
    """
    def __init__(self, size=64, shape_size=5, num_actions=4, num_tasks=4):
        self.size = size
        self.shape_size = shape_size
        self.num_actions = num_actions
        self.pos = np.array([size // 2, size // 2])

        # Define a set of possible tasks (action mappings)
        self._tasks = self._create_tasks(num_actions, num_tasks)
        self.task_id = 0
        self.action_map = self._tasks[self.task_id]

    def _create_tasks(self, num_actions, num_tasks):
        base_map = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),   # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1]),   # Right
        }

        tasks = [base_map]
        available_mappings = list(base_map.values())

        for _ in range(num_tasks - 1):
            shuffled_mappings = available_mappings[:]
            random.shuffle(shuffled_mappings)
            new_task = {i: shuffled_mappings[i] for i in range(num_actions)}
            tasks.append(new_task)

        return tasks

    def reset_task(self, task_id=None):
        """
        Sets a new task for the agent to adapt to.
        If task_id is None, a random task is chosen.
        """
        if task_id is None:
            self.task_id = random.randint(0, len(self._tasks) - 1)
        else:
            self.task_id = task_id % len(self._tasks)

        self.action_map = self._tasks[self.task_id]
        return self.task_id

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

        # Add visual noise
        if self.noise_level > 0:
            noise = torch.rand_like(canvas) * self.noise_level
            canvas = torch.clamp(canvas + noise, 0, 1)

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
        Action execution is stochastic.
        Returns the new observation.
        """
        # Action Stochasticity
        if random.random() < self.action_stochasticity:
            action = random.randint(0, self.num_actions - 1)

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
