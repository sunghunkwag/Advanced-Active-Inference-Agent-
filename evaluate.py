import torch
import numpy as np
import os
import json
from datetime import datetime
import importlib

from environment import PixelEnv
from agent import AgentController
from baseline_agent import BaselineAgentController
from world_model import VAE, TransitionModel, ContextInferenceEngine
from memory import EpisodicMemory
from baseline_models import BaselineVAE, BaselineTransitionModel

# --- Configuration ---
EVAL_TASKS = [10, 11, 12, 13, 14] # Use task IDs unseen during training
EPISODE_LENGTH = 200
IMG_SIZE = 64
VAE_PATH = "saved_models/vae.pth"
TRANSITION_PATH = "saved_models/transition.pth"
CONTEXT_PATH = "saved_models/context.pth"

def run_evaluation_episode(agent, env, episode_length, agent_type="meta"):
    """Runs one episode of evaluation for a given agent."""
    if agent_type == "meta":
        agent.memory.reset()

    obs = env.reset()
    total_reward = 0

    for _ in range(episode_length):
        action = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)

        if agent_type == "meta":
            agent.record_experience(obs, action, next_obs, reward)

        obs = next_obs
        total_reward += reward
        if done:
            break
    return total_reward

def main():
    """Main evaluation script."""
    print("--- Starting Agent Evaluation ---")

    eval_module_name = os.environ.get("EVAL_MODULE", "baseline")
    is_challenger = "challenger" in eval_module_name

    env = PixelEnv(size=IMG_SIZE, num_tasks=max(EVAL_TASKS) + 1)

    # --- Load Agent ---
    latent_dim, context_dim, hidden_dim, num_actions = 32, 16, 256, env.get_num_actions()

    if is_challenger:
        try:
            challenger_module = importlib.import_module(eval_module_name)
            vae = challenger_module.VAE(latent_dim, context_dim, 1, IMG_SIZE)
            print(f"Loaded challenger VAE from {eval_module_name}")
        except Exception as e:
            print(f"ERROR: Could not load challenger module {eval_module_name}. Reverting to default. Error: {e}")
            vae = VAE(latent_dim, context_dim, 1, IMG_SIZE)
    else: # Baseline or default meta-agent
        vae = VAE(latent_dim, context_dim, 1, IMG_SIZE)

    transition_model = TransitionModel(latent_dim, num_actions, context_dim, hidden_dim)
    context_engine = ContextInferenceEngine((1, IMG_SIZE, IMG_SIZE), num_actions, context_dim, hidden_dim)
    memory = EpisodicMemory(200, (1, IMG_SIZE, IMG_SIZE), 1)

    if not is_challenger and os.path.exists(VAE_PATH):
        vae.load_state_dict(torch.load(VAE_PATH))
        transition_model.load_state_dict(torch.load(TRANSITION_PATH))
        context_engine.load_state_dict(torch.load(CONTEXT_PATH))
        print("Loaded pre-trained models for meta-agent.")

    agent = AgentController(vae, transition_model, context_engine, memory, num_actions, latent_dim, context_dim)

    # --- Load Baseline Agent ---
    baseline_vae = BaselineVAE(latent_dim, 1, IMG_SIZE)
    baseline_transition = BaselineTransitionModel(latent_dim, num_actions, hidden_dim)
    baseline_agent = BaselineAgentController(baseline_vae, baseline_transition, num_actions, latent_dim)

    # --- Run Evaluation ---
    agent_rewards = []
    baseline_rewards = []

    for task_id in EVAL_TASKS:
        print(f"\n--- Evaluating on Unseen Task #{task_id} ---")
        env.reset_task(task_id)

        reward = run_evaluation_episode(agent, env, EPISODE_LENGTH)
        agent_rewards.append(reward)
        print(f"  Agent Total Reward: {reward:.2f}")

        baseline_reward = run_evaluation_episode(baseline_agent, env, EPISODE_LENGTH, agent_type="baseline")
        baseline_rewards.append(baseline_reward)
        print(f"  Baseline Agent Total Reward: {baseline_reward:.2f}")

    avg_agent_reward = np.mean(agent_rewards)
    avg_baseline_reward = np.mean(baseline_rewards)

    print("\n--- Evaluation Summary ---")
    print(f"Average Reward (Agent): {avg_agent_reward:.2f}")
    print(f"Average Reward (Baseline): {avg_baseline_reward:.2f}")

    # Output result for the daemon
    result = {
        "module": eval_module_name,
        "avg_reward": avg_agent_reward,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open("temp_eval_result.json", "w") as f:
        json.dump(result, f)

if __name__ == '__main__':
    main()
