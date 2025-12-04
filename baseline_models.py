# This file contains the standard, non-context-aware versions of the VAE and Transition Model.
# It is used by the Baseline Agent for a fair comparison against the Meta-Learning Agent.

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineVAE(nn.Module):
    def __init__(self, latent_dim=32, img_channels=1, img_size=64):
        super(BaselineVAE, self).__init__()
        self.latent_dim = latent_dim
        final_conv_size = img_size // (2**4)
        self.flattened_size = 256 * final_conv_size * final_conv_size

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc(z))
        final_conv_size = int((self.flattened_size // 256)**0.5)
        h = h.view(-1, 256, final_conv_size, final_conv_size)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class BaselineTransitionModel(nn.Module):
    def __init__(self, latent_dim, num_actions, hidden_dim=256):
        super(BaselineTransitionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_actions, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.num_actions = num_actions

    def forward(self, z, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        z_action = torch.cat([z, action_one_hot], dim=1)
        return self.model(z_action)
