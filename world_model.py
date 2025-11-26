import torch
import torch.nn as nn
import torch.nn.functional as F
from film_layer import FiLMLayer

class VAE(nn.Module):
    """
    A context-aware Variational Autoencoder (VAE).
    The decoder is conditioned on a context vector via FiLM layers, allowing it to
    reconstruct observations based on the inferred environment dynamics.
    """
    def __init__(self, latent_dim=32, context_dim=16, img_channels=1, img_size=64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        # Encoder
        final_conv_size = img_size // (2**4)
        self.flattened_size = 256 * final_conv_size * final_conv_size

        self.encoder_net = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.flattened_size)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.film1 = FiLMLayer(context_dim, 128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.film2 = FiLMLayer(context_dim, 64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.film3 = FiLMLayer(context_dim, 32)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        h = self.encoder_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, context_vector):
        h = F.relu(self.decoder_fc(z))
        final_conv_size = int((self.flattened_size // 256)**0.5)
        h = h.view(-1, 256, final_conv_size, final_conv_size)

        x = self.deconv1(h)
        x = self.film1(x, context_vector)
        x = F.relu(x)

        x = self.deconv2(x)
        x = self.film2(x, context_vector)
        x = F.relu(x)

        x = self.deconv3(x)
        x = self.film3(x, context_vector)
        x = F.relu(x)

        recon_x = torch.sigmoid(self.deconv4(x))
        return recon_x

    def forward(self, x, context_vector):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, context_vector), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
    """
    # Reconstruction loss (Binary Cross-Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
