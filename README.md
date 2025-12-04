# Memory-Augmented Meta-Reinforcement Learning Agent

## Overview

This repository contains a cutting-edge implementation of a meta-reinforcement learning agent designed to adapt to new environments in real-time. Built with PyTorch, the agent operates in a pixel-based world where the rules of interaction (i.e., the environment dynamics) can change.

The core of this project is the agent's ability to learn **how to learn**. Instead of just mastering a single task, it learns to quickly infer the hidden rules of a new, noisy, and unpredictable environment from a handful of experiences. This is achieved through a novel architecture based on **Memory-Augmented Meta-RL with Contextual Inference**.

## Architecture

The agent's architecture is a sophisticated system of neural modules designed for rapid adaptation:

-   **Enhanced Meta-Learning Environment (`environment.py`)**: The environment is designed to generate multiple "tasks". Each task has different dynamics (e.g., the effect of an action is shuffled). To increase robustness, the environment is enhanced with:
    -   **Visual Noise**: Random noise is added to observations to challenge the agent's perception.
    -   **Action Stochasticity**: Actions have a chance to fail or result in a random outcome, forcing the agent to develop more robust plans.

-   **Episodic Memory (`memory.py`)**: A short-term memory buffer that stores the agent's recent experiences within a single task.

-   **Context Inference Engine (`context_engine.py`)**: A recurrent neural network (GRU) that analyzes sequences of experience from memory to infer a latent **context vector**, which represents the hidden rules of the current task.

-   **FiLM Layer (`film_layer.py`)**: Implements Feature-wise Linear Modulation, allowing the context vector to dynamically adjust the behavior of the agent's core models without changing their weights.

-   **Context-Aware World Model (`world_model.py` & `transition_model.py`)**: The VAE and Transition Model are modulated by FiLM layers, enabling them to perceive and predict the world according to the inferred context of the current task.

-   **Agent Controller (`agent.py`)**: The agent's decision-making core. It orchestrates context inference and planning to select optimal actions in the face of uncertainty.

## Getting Started

### Prerequisites

-   Python 3.x
-   PyTorch
-   NumPy
-   Optuna (for hyperparameter tuning)

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install the required dependencies:
    ```bash
    pip install torch numpy optuna
    ```

### Running the Meta-Training

To start the meta-training process with a strong set of baseline hyperparameters, run the `main.py` script:

```bash
python3 main.py
```

This will initialize all models and the enhanced environment, then begin the meta-learning loop.

### Hyperparameter Tuning (Optional, Recommended)

To find the optimal hyperparameters for the models and training loop, you can use the automated tuning script powered by Optuna. This will run multiple training trials to find the best-performing configuration.

```bash
python3 tune_hyperparams.py
```

After the script finishes, it will print the best set of hyperparameters found. You can then apply these values to `main.py` for maximum performance.

### Running Tests

This project includes a suite of unit and integration tests. To run them, use the `unittest` module:

```bash
python3 -m unittest discover tests
```

## Project Structure

```
.
├── agent.py              # Controller for context inference and planning
├── context_engine.py     # RNN-based engine to infer context from memory
├── environment.py        # Enhanced meta-learning environment
├── film_layer.py         # FiLM layer for dynamic model modulation
├── main.py               # Main script for the meta-training loop
├── memory.py             # Episodic memory for the current task
├── transition_model.py   # Context-aware model for predicting future states
├── tune_hyperparams.py   # Optuna-based hyperparameter tuning script
├── world_model.py        # Context-aware VAE for world representation
└── tests/
    ├── test_context_engine.py
    ├── test_film_layer.py
    ├── test_integration.py
    └── test_memory.py
```
