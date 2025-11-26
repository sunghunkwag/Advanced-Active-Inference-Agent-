# Advanced Active Inference Agent (PyTorch)

## Overview

This repository contains a sophisticated, deep-learning-based implementation of an autonomous agent using the Active Inference framework. Built with PyTorch, this agent is designed to operate in a pixel-based environment, learning a comprehensive world model to guide its actions.

The agent's core objective is to learn a generative model of its environment, which it then uses to plan actions that minimize its uncertainty about the world (surprise). This project serves as a functional, advanced prototype demonstrating a scalable approach to Active Inference.

## Architecture

The architecture is composed of several deep learning modules that work in concert:

-   **Pixel-Based Environment (`environment.py`)**: A simple environment that generates image-like `torch` tensors as observations, providing a high-dimensional sensory input for the agent.
-   **Variational Autoencoder (VAE) World Model (`world_model.py`)**: A convolutional VAE that serves as the agent's perceptual system. It learns to compress the high-dimensional image observations into a low-dimensional probabilistic latent space (`z`). This allows the agent to form an efficient, internal representation of the world.
-   **Latent Space Transition Model (`transition_model.py`)**: An MLP-based model that learns the environment's dynamics within the VAE's latent space. It predicts the next latent state (`z'`) given the current latent state (`z`) and an action (`a`), enabling the agent to simulate future outcomes.
-   **Agent Controller (`agent.py`)**: The agent's decision-making module. It uses the VAE and transition model to perform planning. By "imagining" future trajectories in the latent space, it selects the action sequence that is expected to be most informative or rewarding.

## Getting Started

### Prerequisites

-   Python 3.x
-   PyTorch
-   NumPy

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install the required dependencies:
    ```bash
    pip install torch numpy
    ```

### Execution

To run the agent's training loop, execute the `main.py` script:

```bash
python3 main.py
```

This will initialize all models and the environment, then begin the process of data collection and model training. The script will periodically print the VAE and transition model losses to the console.

## Project Structure

```
.
├── agent.py              # Controller for planning and action selection
├── environment.py        # Pixel-based environment for the agent
├── main.py               # Main script to run training and execution
├── transition_model.py   # Predicts future latent states
└── world_model.py        # VAE for learning a world representation
```
