import torch
import torch.optim as optim
from itertools import chain

from environment import PixelEnv
from world_model import VAE, vae_loss_function
from transition_model import TransitionModel
from context_engine import ContextInferenceEngine
from memory import EpisodicMemory
from agent import AgentController

# --- Hyperparameters ---
IMG_SIZE = 64
IMG_CHANNELS = 1
LATENT_DIM = 32
CONTEXT_DIM = 16
NUM_ACTIONS = 4
NUM_TASKS = 10
EPISODIC_MEM_CAPACITY = 200
HIDDEN_DIM = 128

# Meta-Training Loop
META_EPISODES = 500
EPISODE_LENGTH = 100 # Steps per task
TRAIN_BATCH_SIZE = 16
CONTEXT_SEQ_LEN = 15

# Agent Planning
PLANNING_HORIZON = 5
NUM_CANDIDATES = 50

def train_meta_step(models, optimizers, memory, batch_size, context_seq_len):
    vae, transition_model, context_engine = models
    meta_optimizer = optimizers[0]

    # Sample a batch of sequences from memory
    sequences = memory.sample(batch_size, context_seq_len)
    if not sequences:
        return 0.0, 0.0, 0.0

    # --- Prepare data ---
    obs_seqs, act_seqs, next_obs_seqs, _ = zip(*[zip(*s) for s in sequences])
    
    obs_b = torch.stack([torch.cat(list(s), 0) for s in obs_seqs])
    act_b = torch.stack([torch.cat(list(s), 0) for s in act_seqs])
    next_obs_b = torch.stack([torch.cat(list(s), 0) for s in next_obs_seqs])

    b, s, c, h, w = obs_b.shape
    obs_b, act_b, next_obs_b = obs_b.view(b*s,c,h,w), act_b.view(b*s,-1), next_obs_b.view(b*s,c,h,w)

    # --- Jointly train all models ---
    meta_optimizer.zero_grad()

    # 1. Infer context from the sequences
    context = context_engine(
        obs_b.view(b, s, -1), act_b.view(b, s, -1), next_obs_b.view(b, s, -1)
    )
    # Repeat context for each step in the sequence
    context = context.repeat_interleave(context_seq_len, dim=0)

    # 2. VAE Loss (Reconstruction)
    recon_obs, mu, logvar = vae(obs_b, context)
    vae_loss = vae_loss_function(recon_obs, obs_b, mu, logvar)

    # 3. Transition Model Loss (Prediction)
    with torch.no_grad():
        z, _ = vae.encode(obs_b)
        next_z, _ = vae.encode(next_obs_b)
    
    predicted_next_z = transition_model(z, act_b.squeeze(), context)
    transition_loss = torch.nn.functional.mse_loss(predicted_next_z, next_z)

    # Total loss is the sum of individual losses
    total_loss = vae_loss + transition_loss
    total_loss.backward()
    meta_optimizer.step()
    
    return vae_loss.item(), transition_loss.item(), total_loss.item()

def main():
    # --- Initialization ---
    env = PixelEnv(size=IMG_SIZE, num_actions=NUM_ACTIONS, num_tasks=NUM_TASKS)
    
    vae = VAE(latent_dim=LATENT_DIM, context_dim=CONTEXT_DIM, img_channels=IMG_CHANNELS, img_size=IMG_SIZE)
    transition_model = TransitionModel(LATENT_DIM, NUM_ACTIONS, CONTEXT_DIM, HIDDEN_DIM)
    context_engine = ContextInferenceEngine((IMG_CHANNELS, IMG_SIZE, IMG_SIZE), NUM_ACTIONS, CONTEXT_DIM, HIDDEN_DIM)
    
    memory = EpisodicMemory(EPISODIC_MEM_CAPACITY, (IMG_CHANNELS, IMG_SIZE, IMG_SIZE), 1)
    
    agent = AgentController(
        vae, transition_model, context_engine, memory, NUM_ACTIONS, LATENT_DIM, CONTEXT_DIM,
        PLANNING_HORIZON, NUM_CANDIDATES, TRAIN_BATCH_SIZE, CONTEXT_SEQ_LEN
    )
    
    # A single optimizer for all models to encourage joint learning
    meta_optimizer = optim.Adam(
        chain(vae.parameters(), transition_model.parameters(), context_engine.parameters()),
        lr=1e-4
    )
    
    print("--- Starting Meta-Training ---")

    for meta_episode in range(META_EPISODES):
        # Sample a new task and reset memory
        task_id = env.reset_task()
        memory.reset()
        obs = env.reset()
        
        total_v_loss, total_t_loss = 0, 0
        
        # Interact with the environment for one episode
        for step in range(EPISODE_LENGTH):
            action = agent.select_action(obs)
            next_obs = env.step(action)
            
            # For simplicity, we use a constant reward.
            agent.record_experience(obs, action, next_obs, reward=0.0)
            obs = next_obs

            # --- Perform a meta-training step ---
            if len(memory) >= CONTEXT_SEQ_LEN:
                v_loss, t_loss, _ = train_meta_step(
                    (vae, transition_model, context_engine),
                    (meta_optimizer,),
                    memory,
                    TRAIN_BATCH_SIZE,
                    CONTEXT_SEQ_LEN
                )
                total_v_loss += v_loss
                total_t_loss += t_loss

        avg_v_loss = total_v_loss / (EPISODE_LENGTH) if EPISODE_LENGTH > 0 else 0
        avg_t_loss = total_t_loss / (EPISODE_LENGTH) if EPISODE_LENGTH > 0 else 0

        print(f"Meta-Episode {meta_episode+1}/{META_EPISODES} | Task: {task_id} | "
              f"Avg VAE Loss: {avg_v_loss:.2f} | Avg Trans Loss: {avg_t_loss:.4f}")
            
    print("--- Meta-Training Finished ---")

if __name__ == "__main__":
    main()
