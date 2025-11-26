import torch
import torch.nn as nn

class TransitionModel(nn.Module):
    """
    A model that predicts the next latent state given the current latent state and an action.
    This model learns the dynamics of the environment in the latent space.
    """
    def __init__(self, latent_dim, num_actions, hidden_dim=256):
        super(TransitionModel, self).__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        
        # We use a simple Multi-Layer Perceptron (MLP) for this.
        # The input is the concatenation of the latent state and a one-hot encoded action.
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim) # Output is the predicted next latent state
        )

    def forward(self, z, action):
        """
        Forward pass to predict the next latent state.
        
        Args:
            z (torch.Tensor): The current latent state, shape (batch_size, latent_dim).
            action (torch.Tensor): The action taken, shape (batch_size,).
            
        Returns:
            torch.Tensor: The predicted next latent state, shape (batch_size, latent_dim).
        """
        # Convert action indices to one-hot vectors
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=self.num_actions).float()
        
        # Concatenate the latent state and the one-hot action vector
        z_action = torch.cat([z, action_one_hot], dim=1)
        
        # Predict the next latent state
        next_z = self.model(z_action)
        
        return next_z
