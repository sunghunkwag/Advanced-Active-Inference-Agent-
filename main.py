import torch
import torch.optim as optim
from collections import deque
import random
import numpy as np

from environment import PixelEnv
from world_model import VAE, vae_loss_function
from transition_model import TransitionModel
from agent import AgentController

# --- Hyperparameters ---
IMG_SIZE = 64
LATENT_DIM = 32
NUM_ACTIONS = 4
BUFFER_SIZE = 10000
BATCH_SIZE = 32
TRAINING_STEPS = 1000
EPISODE_LENGTH = 100

def train_step(vae, transition_model, replay_buffer, vae_optimizer, transition_optimizer):
    if len(replay_buffer) < BATCH_SIZE:
        return  # Not enough data to train

    # --- Sample a batch from the replay buffer ---
    batch = random.sample(replay_buffer, BATCH_SIZE)
    obs_batch, action_batch, next_obs_batch = zip(*batch)

    obs_batch = torch.stack(obs_batch)
    action_batch = torch.tensor(action_batch, dtype=torch.long)
    next_obs_batch = torch.stack(next_obs_batch)

    # --- Train the VAE ---
    vae.train()
    vae_optimizer.zero_grad()
    
    recon_obs, mu, logvar = vae(obs_batch)
    vae_loss = vae_loss_function(recon_obs, obs_batch, mu, logvar)
    vae_loss.backward()
    vae_optimizer.step()

    # --- Train the Transition Model ---
    transition_model.train()
    transition_optimizer.zero_grad()
    
    with torch.no_grad():
        z, _ = vae.encode(obs_batch)
        next_z, _ = vae.encode(next_obs_batch)

    predicted_next_z = transition_model(z, action_batch)
    
    transition_loss = torch.nn.functional.mse_loss(predicted_next_z, next_z)
    transition_loss.backward()
    transition_optimizer.step()
    
    print(f"VAE Loss: {vae_loss.item():.2f}, Transition Loss: {transition_loss.item():.4f}")

def main():
    # --- Initialization ---
    env = PixelEnv(size=IMG_SIZE, num_actions=NUM_ACTIONS)
    vae = VAE(latent_dim=LATENT_DIM, img_size=IMG_SIZE)
    transition_model = TransitionModel(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS)
    agent = AgentController(vae, transition_model, num_actions=NUM_ACTIONS, latent_dim=LATENT_DIM)
    
    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    transition_optimizer = optim.Adam(transition_model.parameters(), lr=1e-4)
    
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    
    obs = env.reset()
    
    print("--- Starting Data Collection and Training ---")
    
    for step in range(TRAINING_STEPS):
        # --- Interact with the environment ---
        action = agent.select_action(obs)
        next_obs = env.step(action)
        
        replay_buffer.append((obs, action, next_obs))
        
        obs = next_obs
        
        if (step + 1) % EPISODE_LENGTH == 0:
            obs = env.reset() # Reset periodically
            
        # --- Perform a training step ---
        if (step + 1) % 10 == 0: # Train every 10 steps
            train_step(vae, transition_model, replay_buffer, vae_optimizer, transition_optimizer)
            
    print("--- Training Finished ---")

if __name__ == "__main__":
    main()
