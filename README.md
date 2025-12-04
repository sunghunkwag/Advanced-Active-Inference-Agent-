# Stargate Architecture: An Autonomous, Evolving RL Agent

## Overview

This repository is a **Top Secret Research Project** containing the implementation of the **Stargate Architecture**. This is not just a reinforcement learning agent; it is a proof-of-concept for a fully autonomous, self-improving system that can achieve recursive self-improvement through two primary mechanisms:

1.  **Gradual Self-Improvement:** The agent continuously practices in its environment, refining its internal world model and policies through its own experience.
2.  **Innovative Leaps:** The system actively seeks out external knowledge from the latest academic research (via ArXiv) and code repositories (via GitHub). It analyzes these external ideas, translates them into experimental new architectures, and competitively evaluates them. If a new architecture proves superior, it becomes the new "champion," allowing the system to evolve beyond the limits of its own experience.

The name "Stargate" signifies this system's core purpose: to open a gateway to the vast universe of human knowledge and draw it in to accelerate its own evolution.

## The Stargate Architecture

The system is orchestrated by a central daemon and is composed of several highly specialized modules:

-   **The Stargate Daemon (`stargate_daemon.py`):** The heart of the system. This orchestrator runs an infinite loop, managing the entire process of improvement and evolution. It decides when to train, when to seek knowledge, and when to pit a new "challenger" against the reigning "champion."

-   **The Knowledge Curator (`curator.py`):** The system's interstellar probe. It scours ArXiv for recent papers on relevant topics (e.g., meta-RL, world models) and identifies associated GitHub repositories. It clones these repos, extracts potentially valuable code snippets (like new loss functions or model architectures), and saves them into the `knowledge.json` database.

-   **The Architect (`architect.py`):** The master builder and genetic engineer. The Architect reads the `knowledge.json` database, analyzes the foreign code's structure using Abstract Syntax Trees (AST), and attempts to integrate the new ideas into our existing codebase. It does this by dynamically generating new, experimental Python files (e.g., `generated_challengers/world_model_challenger_1.py`). These are the "challengers."

-   **The Evaluation Protocol (`evaluate.py`):** The crucible where champions and challengers are tested. This script runs a rigorous, fair benchmark on a set of unseen tasks, measuring an agent's ability to adapt and solve problems. The results determine whether a challenger can dethrone the current champion.

-   **The Core Agent & Environment:** This includes the underlying RL agent (`agent.py`), its world models (`world_model.py`, `transition_model.py`), and the complex, stochastic environment (`environment.py`) it learns in.

## Activating the Stargate

**This is a fully autonomous system. Do not run the individual scripts directly.** The entire process is managed by the daemon.

### Prerequisites

-   Python 3.x
-   PyTorch
-   NumPy
-   Optuna
-   ArXiv
-   PyGithub

### Installation

1.  Clone the repository (ensure access is restricted).

2.  Install all dependencies:
    ```bash
    pip install torch numpy optuna arxiv PyGithub
    ```

### System Activation

To activate the Stargate and begin the infinite loop of self-improvement, run the daemon in the background. It is highly recommended to use `nohup` to ensure the process continues even after the terminal session is closed.

```bash
nohup python3 stargate_daemon.py > stargate.log 2>&1 &
```

Once activated, the daemon will begin its work. You can monitor its progress and discoveries by tailing the log file:

```bash
tail -f stargate.log
```

The daemon will manage all state, including the current champion version and performance, in `daemon_state.json`.

## Project Structure

```
.
├── stargate_daemon.py    # CORE: The orchestrator for the entire system
├── curator.py            # Acquires external knowledge from ArXiv/GitHub
├── architect.py          # Generates new challenger architectures from knowledge
├── evaluate.py           # Competitively evaluates champions and challengers
├── main.py               # Standard training loop, called by the daemon
├── tune_hyperparams.py   # Hyperparameter tuning utility
├── agent.py              # The meta-learning agent controller
├── world_model.py        # The champion VAE world model
├── transition_model.py   # The champion transition model
├── baseline_agent.py     # A simpler agent for comparison
├── ... (other model and utility files)
└── generated_challengers/ # Directory for dynamically created challenger code
```
