import torch
import torch.nn as nn
from film_layer import FiLMLayer

class TransitionModel(nn.Module):
    """
    A context-aware model that predicts the next latent state (z_t+1) given the
    current latent state (z_t), the action taken (a_t), and a context vector.
    The context vector allows the model to adapt its predictions to the current
    environment dynamics (task).
    """
    def __init__(self, latent_dim, num_actions, context_dim, hidden_dim=256):
        super(TransitionModel, self).__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.context_dim = context_dim

        self.fc1 = nn.Linear(latent_dim + num_actions, hidden_dim)
        self.film1 = FiLMLayer(context_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film2 = FiLMLayer(context_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        
        self.relu = nn.ReLU()

    def forward(self, z, action, context_vector):
        """
        Forward pass to predict the next latent state, conditioned on the context.
        
        Args:
            z (torch.Tensor): The current latent state, shape (batch_size, latent_dim).
            action (torch.Tensor): The action taken, shape (batch_size,).
            context_vector (torch.Tensor): The inferred context, shape (batch_size, context_dim).
            
        Returns:
            torch.Tensor: The predicted next latent state, shape (batch_size, latent_dim).
        """
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=self.num_actions).float()
        z_action = torch.cat([z, action_one_hot], dim=1)

        x = self.fc1(z_action)
        x = self.film1(x, context_vector)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.film2(x, context_vector)
        x = self.relu(x)
        
        next_z = self.fc3(x)
        
        return next_z
