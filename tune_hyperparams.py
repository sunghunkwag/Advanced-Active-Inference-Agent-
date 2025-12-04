import optuna
import torch
import torch.optim as optim
from itertools import chain

from environment import PixelEnv
from world_model import VAE, vae_loss_function
from transition_model import TransitionModel
from context_engine import ContextInferenceEngine
from memory import EpisodicMemory
from agent import AgentController

# --- Constants for Tuning ---
# Use smaller values for faster tuning trials
N_TRIALS = 30  # Number of different hyperparameter sets to try
META_EPISODES_PER_TRIAL = 20 # Run for fewer episodes during tuning
EPISODE_LENGTH = 50
IMG_SIZE = 32 # Use smaller images for speed

def objective(trial):
    """
    This function is called by Optuna for each trial.
    It trains the meta-RL agent with a given set of hyperparameters
    and returns the final performance metric (e.g., average loss).
    """
    # --- 1. Suggest Hyperparameters ---
    # Optuna will pick values from these ranges for each trial
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    latent_dim = trial.suggest_categorical('latent_dim', [16, 32, 64])
    context_dim = trial.suggest_categorical('context_dim', [8, 16, 32])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])

    # --- 2. Initialize Models and Environment with Suggested Hyperparameters ---
    env = PixelEnv(size=IMG_SIZE)
    vae = VAE(latent_dim=latent_dim, context_dim=context_dim, img_channels=1, img_size=IMG_SIZE)
    transition_model = TransitionModel(latent_dim, env.get_num_actions(), context_dim, hidden_dim)
    context_engine = ContextInferenceEngine((1, IMG_SIZE, IMG_SIZE), env.get_num_actions(), context_dim, hidden_dim)
    memory = EpisodicMemory(200, (1, IMG_SIZE, IMG_SIZE), 1)
    agent = AgentController(
        vae, transition_model, context_engine, memory, env.get_num_actions(),
        latent_dim, context_dim, context_seq_len=10
    )

    meta_optimizer = optim.Adam(
        chain(vae.parameters(), transition_model.parameters(), context_engine.parameters()), lr=lr
    )

    # --- 3. Run a shortened Meta-Training Loop ---
    final_losses = []
    for meta_episode in range(META_EPISODES_PER_TRIAL):
        env.reset_task()
        memory.reset()
        obs = env.reset()

        episode_total_loss = 0
        steps_with_training = 0

        for step in range(EPISODE_LENGTH):
            action = agent.select_action(obs)
            next_obs = env.step(action)
            agent.record_experience(obs, action, next_obs, 0.0)
            obs = next_obs

            if len(memory) >= 15:
                # Re-use the training logic from main.py
                sequences = memory.sample(16, 10)
                if not sequences: continue

                obs_seqs, act_seqs, next_obs_seqs, _ = zip(*[zip(*s) for s in sequences])
                obs_b = torch.stack([torch.stack(s) for s in obs_seqs])
                act_b = torch.stack([torch.stack(s) for s in act_seqs])
                next_obs_b = torch.stack([torch.stack(s) for s in next_obs_seqs])

                b, s, c, h, w = obs_b.shape

                obs_b = obs_b.view(b * s, c, h, w)
                act_b = act_b.view(b * s, -1)
                next_obs_b = next_obs_b.view(b * s, c, h, w)

                meta_optimizer.zero_grad()
                context = context_engine(obs_b.view(b, s, -1), act_b.view(b, s, -1), next_obs_b.view(b, s, -1))
                context = context.repeat_interleave(10, dim=0)

                recon_obs, mu, logvar = vae(obs_b, context)
                vae_loss = vae_loss_function(recon_obs, obs_b, mu, logvar)

                with torch.no_grad():
                    z, _ = vae.encode(obs_b)
                    next_z, _ = vae.encode(next_obs_b)
                predicted_next_z = transition_model(z, act_b.squeeze(), context)
                transition_loss = torch.nn.functional.mse_loss(predicted_next_z, next_z)

                total_loss = vae_loss + transition_loss
                total_loss.backward()
                meta_optimizer.step()

                episode_total_loss += total_loss.item()
                steps_with_training += 1

        if steps_with_training > 0:
            final_losses.append(episode_total_loss / steps_with_training)

    # --- 4. Return the final performance metric ---
    # We want to minimize the average loss over the last few episodes
    if not final_losses:
        return float('inf') # Should not happen if training goes well
    return sum(final_losses[-5:]) / len(final_losses[-5:])

if __name__ == '__main__':
    # --- Create and run the Optuna study ---
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n--- Hyperparameter Tuning Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Min Avg Loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
