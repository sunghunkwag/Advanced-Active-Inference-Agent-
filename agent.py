import torch
import random
import torch.nn.functional as F

class AgentController:
    """
    The agent's decision-making component, upgraded for meta-learning.
    It infers a context vector from recent experience and uses it to plan actions
    with the context-aware world model.
    """
    def __init__(self, vae, transition_model, context_engine, memory, num_actions,
                 latent_dim, context_dim, planning_horizon=10, num_candidates=100,
                 context_batch_size=16, context_seq_len=10, device=None):
        self.vae = vae
        self.transition_model = transition_model
        self.context_engine = context_engine
        self.memory = memory
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.planning_horizon = planning_horizon
        self.num_candidates = num_candidates
        self.context_batch_size = context_batch_size
        self.context_seq_len = context_seq_len
        self.device = device or next(vae.parameters()).device

        self.current_context = torch.zeros(1, context_dim, device=self.device)

    def select_action(self, observation):
        """
        Selects an action by inferring context and then planning.
        """
        # 1. Infer the context from recent memory
        inferred_context = self.context_engine.infer_context(
            self.memory, self.context_batch_size, self.context_seq_len, device=self.device
        )
        if inferred_context is not None:
            self.current_context = inferred_context

        # 2. Get the initial latent state from the current observation
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(observation.unsqueeze(0).to(self.device))
            current_z = mu

        # 3. Plan using the random-shooting method, now conditioned on context
        self.transition_model.eval()
        candidate_actions = [
            [random.randint(0, self.num_actions - 1) for _ in range(self.planning_horizon)]
            for _ in range(self.num_candidates)
        ]
        
        best_score = -float('inf')
        best_action = 0

        for action_sequence in candidate_actions:
            z = current_z.clone()
            with torch.no_grad():
                imagined_zs = []
                for action in action_sequence:
                    action_tensor = torch.tensor(
                        [action], device=self.device, dtype=torch.long
                    )
                    # Crucially, pass the context to the transition model
                    z = self.transition_model(z, action_tensor, self.current_context)
                    imagined_zs.append(z)
            
            imagined_trajectory = torch.stack(imagined_zs)
            score = torch.var(imagined_trajectory, dim=0).mean().item()

            if score > best_score:
                best_score = score
                best_action = action_sequence[0]
                
        return best_action

    def record_experience(self, obs, action, next_obs, reward):
        """
        Stores the latest transition in the episodic memory.
        """
        self.memory.push(obs, action, next_obs, reward)
