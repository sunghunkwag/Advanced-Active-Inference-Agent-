import torch
import random
from baseline_models import BaselineVAE, BaselineTransitionModel

class BaselineAgentController:
    """
    A standard RL agent controller for baseline comparison.
    It uses a world model for planning but lacks the meta-learning
    capabilities (context inference, episodic memory) of the main agent.

    This agent needs to be retrained for every new task.
    """
    def __init__(self, vae: BaselineVAE, transition_model: BaselineTransitionModel,
                 num_actions, latent_dim, planning_horizon=10, num_candidates=100):
        self.vae = vae
        self.transition_model = transition_model
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.planning_horizon = planning_horizon
        self.num_candidates = num_candidates

    def select_action(self, observation):
        """
        Selects an action using a random-shooting planning algorithm.
        This process does NOT use a context vector.
        """
        self.vae.eval()
        self.transition_model.eval()

        with torch.no_grad():
            mu, _ = self.vae.encode(observation.unsqueeze(0))
            current_z = mu

        candidate_actions = [
            [random.randint(0, self.num_actions - 1) for _ in range(self.planning_horizon)]
            for _ in range(self.num_candidates)
        ]

        best_score = -float('inf')
        best_action = 0

        for action_sequence in candidate_actions:
            z = current_z.clone()
            imagined_zs = []
            with torch.no_grad():
                for action in action_sequence:
                    action_tensor = torch.tensor([action])
                    z = self.transition_model(z, action_tensor)
                    imagined_zs.append(z)

            # The scoring objective is to find a goal, so we use the VAE
            # to decode the imagined states and see how close they are to a "goal" state.
            # For simplicity here, we stick to the original exploration objective.
            imagined_trajectory = torch.stack(imagined_zs)
            score = torch.var(imagined_trajectory, dim=0).mean().item()

            if score > best_score:
                best_score = score
                best_action = action_sequence[0]

        return best_action
