import torch
import torch.nn as nn
import torch.nn.functional as F


class TriNeuroHybrid(nn.Module):
    """
    A simplified implementation of the Tri‑Neuro Hybrid Architecture.

    This module combines three distinct processing paths—semantic, spatial and
    dynamic—into a single latent representation. The semantic path uses a small
    Transformer encoder, the spatial path is represented by a multi‑layer
    perceptron similar to predictive coding models (e.g. JEPA), and the dynamic
    path is a recurrent cell inspired by liquid neural networks. A gating
    network determines how much each path contributes to the final
    representation based on a context vector. A global manifold state is
    maintained via exponential moving average to provide a persistent memory
    across calls.

    Args:
        input_dim (int): Dimension of the input feature vector.
        context_dim (int): Dimension of the context vector.
        latent_dim (int): Desired dimensionality of the output latent representation.
        hidden_dim (int, optional): Hidden size used inside the JEPA‑like MLP and Transformer.
        ema_decay (float, optional): Decay factor for the exponential moving average.
    """

    def __init__(self, input_dim: int, context_dim: int, latent_dim: int,
                 hidden_dim: int = 128, ema_decay: float = 0.8):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay

        # Semantic path: tiny Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=4, dim_feedforward=hidden_dim, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Spatial path: JEPA-like feedforward net
        self.jepa_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Dynamic path: GRU cell
        self.gru_cell = nn.GRUCell(input_dim, latent_dim)
        self.register_buffer("_gru_state", None)

        # Gating network: produce three weights
        self.gating_net = nn.Linear(context_dim, 3)

        # Projection to latent_dim (for transformer output)
        self.proj = nn.Linear(input_dim, latent_dim)

        # Global manifold state (EMA buffer)
        self.register_buffer("global_state", None)

    def reset_state(self):
        """Reset the internal recurrent and global states."""
        self._gru_state = None
        self.global_state = None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Semantic path
        sem_input = x.unsqueeze(0)
        sem_out = self.transformer(sem_input).squeeze(0)
        sem_out = self.proj(sem_out)

        # Spatial path
        spa_out = self.jepa_net(x)

        # Dynamic path
        if self._gru_state is None or self._gru_state.size(0) != batch_size:
            self._gru_state = torch.zeros(batch_size, self.latent_dim, device=x.device)
        dyn_out = self.gru_cell(x, self._gru_state)
        self._gru_state = dyn_out.detach()

        # Gating
        gates = F.softmax(self.gating_net(context), dim=-1)
        w_sem, w_spa, w_dyn = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]

        sem_weighted = w_sem * sem_out
        spa_weighted = w_spa * spa_out
        dyn_weighted = w_dyn * dyn_out

        integrated = sem_weighted + spa_weighted + dyn_weighted

        # Update global state (EMA)
        if self.global_state is None or self.global_state.size(0) != batch_size:
            self.global_state = integrated.detach().clone()
        else:
            self.global_state = (
                self.ema_decay * self.global_state + (1 - self.ema_decay) * integrated.detach()
            )

        return integrated
