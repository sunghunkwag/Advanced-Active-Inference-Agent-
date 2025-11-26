import unittest
import torch
from collections import deque

from environment import PixelEnv
from world_model import VAE, vae_loss_function
from transition_model import TransitionModel
from agent import AgentController

class IntegrationTest(unittest.TestCase):

    def test_training_step_runs(self):
        """
        A simple integration test to ensure a single training step
        can be executed without errors.
        """
        try:
            # --- Initialization ---
            env = PixelEnv(size=32, num_actions=4) # Smaller size for faster test
            vae = VAE(latent_dim=16, img_size=32)
            transition_model = TransitionModel(latent_dim=16, num_actions=4)
            agent = AgentController(vae, transition_model, num_actions=4, latent_dim=16)
            
            vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
            transition_optimizer = torch.optim.Adam(transition_model.parameters(), lr=1e-4)
            
            # --- Create dummy data ---
            replay_buffer = deque(maxlen=100)
            obs = env.reset()
            for _ in range(10): # Collect a few samples
                action = agent.select_action(obs)
                next_obs = env.step(action)
                replay_buffer.append((obs, action, next_obs))
                obs = next_obs
            
            # --- Execute a single training step ---
            from main import train_step
            train_step(vae, transition_model, replay_buffer, vae_optimizer, transition_optimizer)

            self.assertTrue(True)

        except Exception as e:
            self.fail(f"Integration test failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
