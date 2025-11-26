import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    A Variational Autoencoder (VAE) to learn a latent representation of the environment.
    """
    def __init__(self, latent_dim=32, img_channels=1, img_size=64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        # Calculate the size of the flattened feature map after the encoder
        final_conv_size = img_size // (2**4) # 4 conv layers with stride 2
        self.flattened_size = 256 * final_conv_size * final_conv_size

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1), # -> img_size/2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # -> img_size/4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # -> img_size/8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),         # -> img_size/16
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent space mean and log-variance
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1), # -> 64x64
            nn.Sigmoid() # Use Sigmoid for pixel values between 0 and 1
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

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

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
    """
    # Reconstruction loss (Binary Cross-Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
