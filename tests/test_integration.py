import unittest
import pytest

torch = pytest.importorskip("torch")
from environment import PixelEnv
from world_model import VAE
from transition_model import TransitionModel
from context_engine import ContextInferenceEngine
from memory import EpisodicMemory
from agent import AgentController

class TestIntegration(unittest.TestCase):

    def test_full_agent_flow(self):
        # Hyperparameters for test
        img_size = 16 # Use smaller image for faster test
        latent_dim = 8
        context_dim = 4
        num_actions = 4
        mem_capacity = 50

        # Initialization
        env = PixelEnv(size=img_size, num_actions=num_actions)
        vae = VAE(latent_dim=latent_dim, context_dim=context_dim, img_channels=1, img_size=img_size)
        transition_model = TransitionModel(latent_dim, num_actions, context_dim)
        context_engine = ContextInferenceEngine((1, img_size, img_size), num_actions, context_dim)
        memory = EpisodicMemory(mem_capacity, (1, img_size, img_size), 1)
        agent = AgentController(
            vae, transition_model, context_engine, memory, num_actions,
            latent_dim, context_dim, planning_horizon=2, num_candidates=5,
            context_batch_size=4, context_seq_len=5
        )

        try:
            # --- Test a single step ---
            obs = env.reset()

            # 1. Agent selects an action
            action = agent.select_action(obs)
            self.assertIsInstance(action, int)

            # 2. Environment steps
            next_obs = env.step(action)
            self.assertEqual(obs.shape, next_obs.shape)

            # 3. Agent records experience
            agent.record_experience(obs, action, next_obs, 0.0)
            self.assertEqual(len(memory), 1)

            # --- Test context inference ---
            for i in range(10): # Fill memory a bit
                action_to_record = i % num_actions
                agent.record_experience(obs, action_to_record, next_obs, 0.0)

            context = context_engine.infer_context(memory, 4, 5)
            self.assertIsNotNone(context)
            self.assertEqual(context.shape, (1, context_dim))

            # --- Test model predictions with context ---
            z, _ = vae.encode(obs.unsqueeze(0))
            next_z_pred = transition_model(z, torch.tensor([action]), context)
            self.assertEqual(next_z_pred.shape, (1, latent_dim))

            recon_obs, _, _ = vae(obs.unsqueeze(0), context)
            self.assertEqual(recon_obs.shape, (1, 1, img_size, img_size))

        except Exception as e:
            self.fail(f"Integration test failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
