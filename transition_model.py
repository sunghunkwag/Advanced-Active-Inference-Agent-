import torch
import torch.nn as nn
import torch.nn.functional as F

from film_layer import FiLMLayer
from tri_neuro_model import TriNeuroHybrid


class TransitionModel(nn.Module):
    """
    A context-aware model that predicts the next latent state (z_{t+1}) given the
    current latent state (z_t), the action taken (a_t), and a context vector.

    Optionally integrates a Tri‑Neuro Hybrid module to capture richer dynamics.
    When enabled via ``use_tri_neuro``, predictions are a combination of the
    original FiLM-modulated MLP and the Tri‑Neuro output.
    """

    def __init__(self, latent_dim: int, num_actions: int, context_dim: int,
                 hidden_dim: int = 256, use_tri_neuro: bool = False,
                 tri_hidden_dim: int | None = None, tri_ema_decay: float = 0.8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.use_tri_neuro = use_tri_neuro

        # FiLM-modulated feedforward net
        self.fc1 = nn.Linear(latent_dim + num_actions, hidden_dim)
        self.film1 = FiLMLayer(context_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film2 = FiLMLayer(context_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

        # Optional Tri-Neuro module
        if self.use_tri_neuro:
            tri_hidden = tri_hidden_dim if tri_hidden_dim is not None else hidden_dim // 2
            input_dim = latent_dim + num_actions
            self.tri_neuro = TriNeuroHybrid(
                input_dim=input_dim,
                context_dim=context_dim,
                latent_dim=latent_dim,
                hidden_dim=tri_hidden,
                ema_decay=tri_ema_decay,
            )

    def forward(self, z: torch.Tensor, action: torch.Tensor, context_vector: torch.Tensor) -> torch.Tensor:
        # Ensure action is 1D
        if action.dim() > 1:
            action = action.view(-1)

        # One-hot encode actions
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=self.num_actions).float()
        z_action = torch.cat([z, action_one_hot], dim=1)

        # FiLM-modulated MLP
        x = self.fc1(z_action)
        x = self.film1(x, context_vector)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.film2(x, context_vector)
        x = self.relu(x)

        next_z_mlp = self.fc3(x)

        # Tri-Neuro path
        if self.use_tri_neuro:
            tri_out = self.tri_neuro(z_action, context_vector)
            next_z = next_z_mlp + tri_out
        else:
            next_z = next_z_mlp
        return next_z

    def reset_states(self) -> None:
        """Reset Tri‑Neuro internal states, if used."""
        if self.use_tri_neuro:
            self.tri_neuro.reset_state()
