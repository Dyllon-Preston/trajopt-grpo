# trajopt-grpo
Implementation of Group Relative Policy Optimization for Quadrotor Applications

## Overview

This project implements reinforcement learning (RL) algorithms with a focus on Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO). It is designed for a variety of applications—including classical control (e.g., CartPole, Pendulum) and more advanced tasks (e.g., quadrotors, quadrotor swarms)—using a modular, extensible, and well-tested codebase.

## Table of Contents

- [Overview](#overview)
- [Visualizations](#visualizations)
- [Core Components](#core-components)
  - [Algorithms](#algorithms)
  - [Environments](#environments)
  - [Models](#models)
  - [Policies](#policies)
  - [Buffers](#buffers)
  - [Rollout](#rollout)
  - [Training](#training)
- [Scripts](#scripts)
- [Tests](#tests)
- [Notable Features](#notable-features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)

## Visualizations

![CartPole PPO Simulation](https://github.com/Dyllon-Preston/trajopt-grpo/blob/main/reports/CartPole/cartpole_nn_ppo/001/simulation.gif)

## Core Components
### Algorithms
- **Base Algorithm Class:** An abstract class defining the RL algorithm interface.
- **PPO Implementation:** Implementation of Proximal Policy Optimization.
- **GRPO Implementation:** Implementation of Group Relative Policy Optimization.

### Environments
- **Base Env Class:** An abstract class extending `gymnasium.Env`.
- **CartPole, Pendulum, Quadrotor, Test Environment:** Multiple environments for diverse applications.

### Models
- **NeuralNetwork:** A fully configurable multi-layer perceptron (MLP) for approximating policies and value functions.

### Policies
- **GaussianActor_NeuralNetwork:** A Gaussian policy network for continuous action spaces.
- **GaussianActorCritic_NeuralNetwork:** Combined actor-critic architecture.

### Buffers
- **Base Buffer Class:** An abstract class for experience storage.
- **Rollout_Buffer:** Collects trajectory data during rollouts.
- **TokenizedBuffer:** Alternative buffer for specialized applications.

### Rollout
- **RolloutManager:** Manages parallel rollout workers to simulate environments.
- **RolloutWorker:** Collects episodes and trajectories from individual environment instances.

### Training
- **Trainer:** Orchestrates the training process, including rollout collection, buffer management, and model updates.
- **Checkpoint Management:** Save and load model checkpoints and training metadata.

## Scripts

- `cartpole_nn_ppo.py` → Train PPO on CartPole.
- `cartpole_nn_grpo.py` → Train GRPO on CartPole.
- `pendulum_nn_ppo.py` → Train PPO on Pendulum.
- `pendulum_nn_grpo.py` → Train GRPO on Pendulum.

## Tests

The project includes a comprehensive test suite covering:
- **Environments:** Validates behavior of custom environments.
- **Models:** Ensures network architectures are built correctly.
- **Policies:** Verifies that policy networks return valid actions and distributions.
- **Rollout Mechanisms:** Tests the functionality of rollout managers and workers.
- **Buffer Implementations:** Checks data storage, RTG computations, and retrieval functions.


## Notable Features

- **Multiprocessing Support:** Efficient, parallel rollout generation.
- **Visualization:** Tools for visualizing rollouts and environment interactions.
- **Checkpoint Management:** Organized saving/loading of model checkpoints and metadata.
- **Configurable Architectures:** Flexible neural network architectures for diverse applications.
- **Action Space Support:** Compatible with both continuous and discrete action spaces.
- **Robust Engineering Practices:** Abstract base classes, separation of concerns, comprehensive testing, type hints, and extensive documentation.



## Usage

Run a script to start training, for example:

`python scripts/cartpole_nn_ppo.py`

## Running Tests

Execute unit tests:

`pytest tests/ -v`

## Model Checkpoints & Logs

Checkpoints are saved in the archive/ directory, organized by:
- Environment (CartPole, Quadrotors, etc.)
- Experiment Name
- Checkpoints (001, 002, ...)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

