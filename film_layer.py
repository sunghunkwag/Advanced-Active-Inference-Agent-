import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.

    This layer dynamically adjusts the output of a preceding layer by applying
    a feature-wise affine transformation. The parameters of this transformation
    (gamma for scaling, beta for shifting) are generated from a context vector.

    y = gamma * x + beta
    """
    def __init__(self, context_dim, feature_dim):
        """
        Args:
            context_dim (int): The dimensionality of the context vector.
            feature_dim (int): The number of features in the input tensor `x`.
                               This is equivalent to `num_channels` for convolutional layers.
        """
        super(FiLMLayer, self).__init__()
        self.feature_dim = feature_dim

        # A linear layer to project the context vector to produce gamma and beta.
        # We need 2 * feature_dim outputs: one set for gamma, one for beta.
        self.context_projection = nn.Linear(context_dim, feature_dim * 2)

    def forward(self, x, context_vector):
        """
        Apply the FiLM transformation.

        Args:
            x (Tensor): The input tensor from the previous layer.
                        Shape for CNNs: (batch_size, feature_dim, height, width)
                        Shape for Linear: (batch_size, feature_dim)
            context_vector (Tensor): The context vector. Shape: (batch_size, context_dim)

        Returns:
            Tensor: The modulated tensor with the same shape as `x`.
        """
        # Generate gamma and beta from the context vector
        # projected_context shape: (batch_size, feature_dim * 2)
        projected_context = self.context_projection(context_vector)

        # Split the projected context into gamma and beta
        # gamma, beta shape: (batch_size, feature_dim)
        gamma, beta = torch.chunk(projected_context, 2, dim=-1)

        # Reshape gamma and beta to match the input tensor `x` for broadcasting
        # This is crucial for applying the transformation element-wise.
        if x.dim() == 4:  # Convolutional layer output (B, C, H, W)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply the affine transformation
        return gamma * x + beta
