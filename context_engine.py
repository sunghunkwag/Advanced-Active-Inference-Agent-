import torch
import torch.nn as nn

class ContextInferenceEngine(nn.Module):
    """
    Infers a latent context vector from a sequence of recent experiences.
    This context vector should capture the unobserved dynamics (the current task)
    of the environment.
    It uses a Gated Recurrent Unit (GRU) to process the sequence.
    """
    def __init__(self, observation_shape, action_dim, context_dim, hidden_dim=128):
        super(ContextInferenceEngine, self).__init__()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Assuming observation is a flattened vector for the RNN
        # In a real scenario with images, a CNN encoder would be used first.
        obs_dim = torch.prod(torch.tensor(observation_shape)).item()

        # The input to the GRU will be the concatenation of obs, action, and next_obs
        input_dim = obs_dim + action_dim + obs_dim

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, context_dim)

    def forward(self, obs_sequence, action_sequence, next_obs_sequence):
        """
        Forward pass to infer the context vector.

        Args:
            obs_sequence (Tensor): (batch_size, sequence_length, obs_dim)
            action_sequence (Tensor): (batch_size, sequence_length, action_dim)
            next_obs_sequence (Tensor): (batch_size, sequence_length, obs_dim)

        Returns:
            Tensor: The inferred context vector (batch_size, context_dim)
        """
        # Flatten observations
        batch_size, seq_len, _ = obs_sequence.shape
        obs_flat = obs_sequence.view(batch_size, seq_len, -1)
        next_obs_flat = next_obs_sequence.view(batch_size, seq_len, -1)

        # One-hot encode actions
        action_one_hot = torch.nn.functional.one_hot(
            action_sequence.squeeze(-1), num_classes=self.action_dim
        ).float()

        # Concatenate inputs along the feature dimension
        gru_input = torch.cat([obs_flat, action_one_hot, next_obs_flat], dim=-1)

        # Pass through GRU
        # h_n shape: (num_layers, batch_size, hidden_dim)
        _, h_n = self.gru(gru_input)

        # Get the hidden state of the last time step
        # Squeeze to remove the num_layers dimension: (batch_size, hidden_dim)
        last_hidden_state = h_n.squeeze(0)

        # Pass through the final fully connected layer
        context_vector = self.fc_out(last_hidden_state)

        return context_vector

    def infer_context(self, memory, batch_size, sequence_length):
        """
        Samples from memory and infers the context.
        A convenience method for inference.
        """
        if len(memory) < sequence_length:
            return None # Not enough data to infer

        sequences = memory.sample(batch_size, sequence_length)
        if not sequences:
            return None

        # --- Prepare batch for the model ---
        # Unzip the sequences
        obs_seq, action_seq, next_obs_seq, _ = zip(*[zip(*s) for s in sequences])

        # Stack and convert to tensors
        obs_tensor = torch.stack([torch.cat(list(s), 0) for s in obs_seq])
        action_tensor = torch.stack([torch.cat(list(s), 0) for s in action_seq])
        next_obs_tensor = torch.stack([torch.cat(list(s), 0) for s in next_obs_seq])

        # Reshape to (batch_size, sequence_length, feature_dim)
        obs_tensor = obs_tensor.view(len(sequences), sequence_length, -1)
        action_tensor = action_tensor.view(len(sequences), sequence_length, -1)
        next_obs_tensor = next_obs_tensor.view(len(sequences), sequence_length, -1)

        with torch.no_grad():
            context = self.forward(obs_tensor, action_tensor, next_obs_tensor)

        # For simplicity during inference, we average the context vectors if batch_size > 1
        return context.mean(dim=0, keepdim=True)
