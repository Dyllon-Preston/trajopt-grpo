import numpy as np
import multiprocessing as mp
from rollout import RolloutWorker
from environments import CartPole


def test_rollout_worker():
    """Tests the RolloutWorker class."""

    # Create environment instance
    env = CartPole()

    # Manager for tracking episodes completed
    manager = mp.Manager()
    episodes_completed = manager.list([0])

    # Random policy for testing
    random_policy = lambda state: (env.action_space.sample(), 0)

    # Initialize the worker
    worker = RolloutWorker(worker_id=0, env=env, policy=random_policy, episodes_completed=episodes_completed)

    # Run rollouts
    num_episodes = 3
    observations, actions, log_probs, rewards, episode_lengths = worker.run_episodes(num_episodes=num_episodes)

    # --- Validations ---
    assert observations.shape == (num_episodes, env.max_steps, env.observation_space.shape[0]), \
        f"Unexpected observation shape: {observations.shape}"

    assert actions.shape == (num_episodes, env.max_steps, env.action_space.shape[0]), \
        f"Unexpected action shape: {actions.shape}"
    
    assert log_probs.shape == (num_episodes, env.max_steps), \
        f"Unexpected log_probs shape: {log_probs.shape}"

    assert rewards.shape == (num_episodes, env.max_steps), \
        f"Unexpected reward shape: {rewards.shape}"

    assert episode_lengths.shape == (num_episodes,), \
        f"Unexpected episode length shape: {episode_lengths.shape}"

    # Ensure episodes_completed is updated correctly
    assert episodes_completed[0] == num_episodes, \
        f"episodes_completed should be {num_episodes}, but got {episodes_completed[0]}"