import torch
import random

class AgentController:
    """
    The agent's decision-making component (Controller).
    It uses the learned models to plan and select actions.
    """
    def __init__(self, vae, transition_model, num_actions, latent_dim,
                 planning_horizon=10, num_candidates=100):
        self.vae = vae
        self.transition_model = transition_model
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.planning_horizon = planning_horizon
        self.num_candidates = num_candidates

    def select_action(self, observation):
        """
        Selects an action using a simple planning algorithm.
        This uses a random-shooting method:
        1. Generate a number of random action sequences.
        2. "Imagine" the outcome of each sequence using the transition model.
        3. Score each sequence based on a simple objective (e.g., novelty).
        4. Execute the first action of the best sequence.
        """
        # Ensure models are in evaluation mode
        self.vae.eval()
        self.transition_model.eval()

        # Get the initial latent state from the current observation
        with torch.no_grad():
            mu, _ = self.vae.encode(observation.unsqueeze(0))
            current_z = mu

        # 1. Generate random action sequences (candidates)
        candidate_actions = [
            [random.randint(0, self.num_actions - 1) for _ in range(self.planning_horizon)]
            for _ in range(self.num_candidates)
        ]
        
        best_score = -float('inf')
        best_action = 0

        # 2. Evaluate each candidate sequence
        for action_sequence in candidate_actions:
            z = current_z.clone()
            imagined_zs = []

            # "Imagine" the trajectory
            with torch.no_grad():
                for action in action_sequence:
                    action_tensor = torch.tensor([action])
                    z = self.transition_model(z, action_tensor)
                    imagined_zs.append(z)
            
            # 3. Score the sequence.
            # A simple scoring objective: reward variance in the latent space.
            # This encourages exploration of diverse latent states.
            imagined_trajectory = torch.stack(imagined_zs)
            score = torch.var(imagined_trajectory, dim=0).mean().item()

            if score > best_score:
                best_score = score
                best_action = action_sequence[0]
                
        return best_action
