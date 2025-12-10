# Advanced-Active-Inference-Agent-

## Overview

This repository contains a cutting-edge implementation of a meta-reinforcement learning agent designed to adapt to new environments in real-time. Built with PyTorch, the agent operates in a pixel-based world where the rules of interaction (i.e., the environment dynamics) can change.

The core of this project is the agent's ability to learn **how to learn**. Instead of just mastering a single task, it learns to quickly infer the hidden rules of a new environment from a handful of experiences. This is achieved through a novel architecture based on **Memory-Augmented Meta-RL with Contextual Inference**.

## Architecture

The agent's architecture is a sophisticated system of neural modules designed for rapid adaptation:

-   **Meta-Learning Environment (`environment.py`)**: The environment is designed to generate multiple "tasks". Each task has different dynamics (e.g., the effect of an action is shuffled). The agent's goal is to quickly adapt to the dynamics of the current task.

-   **Episodic Memory (`memory.py`)**: A short-term memory buffer that stores the agent's recent experiences (state, action, next_state) within a single task. This memory provides the raw data for understanding the current environment.

-   **Context Inference Engine (`context_engine.py`)**: The "brain" of the adaptation mechanism. This module is a recurrent neural network (GRU) that analyzes the sequence of experiences in the episodic memory to infer a latent **context vector**. This vector is a compressed representation of the current task's hidden rules.

-   **FiLM Layer (`film_layer.py`)**: Implements Feature-wise Linear Modulation. This is a powerful conditioning technique that allows the `context_vector` to dynamically adjust the behavior of the agent's core models without changing their weights.

-   **Context-Aware World Model (`world_model.py` & `transition_model.py`)**:
    -   The **VAE** (`world_model.py`) learns a compressed representation of the world. Its decoder is modulated by the `FiLM` layer, allowing it to reconstruct the environment in a way that is consistent with the inferred context.
    -   The **Transition Model** (`transition_model.py`) predicts future states in the latent space. It is also modulated by the `FiLM` layer, enabling it to predict the future according to the specific rules of the current task.

-   **Agent Controller (`agent.py`)**: The agent's decision-making core. It first uses the `ContextInferenceEngine` to understand the current situation, then uses the `Context-Aware World Model` to "imagine" future outcomes and select the best action.

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

To start the meta-training process, where the agent learns how to adapt, run the `main.py` script:

```bash
python3 main.py
```

This will initialize all models and the environment, then begin the meta-learning loop. The script will print the average losses for each meta-episode.

### Running Tests

This project includes a suite of unit and integration tests to ensure the correctness of each component. To run the tests, use the `unittest` module:

```bash
python3 -m unittest discover tests
```

## Project Structure

```
.
├── agent.py              # Controller for context inference and planning
├── context_engine.py     # RNN-based engine to infer context from memory
├── environment.py        # Meta-learning environment with multiple tasks
├── film_layer.py         # FiLM layer for dynamic model modulation
├── main.py               # Main script for the meta-training loop
├── memory.py             # Episodic memory for the current task
├── transition_model.py   # Context-aware model for predicting future states
├── world_model.py        # Context-aware VAE for world representation
└── tests/
    ├── test_context_engine.py
    ├── test_film_layer.py
    ├── test_integration.py
    └── test_memory.py
```

```
