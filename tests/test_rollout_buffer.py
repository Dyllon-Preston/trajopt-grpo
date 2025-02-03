import pytest
import numpy as np
import torch
from buffers import Rollout_Buffer
from environments import CartPole
from rollout import RolloutWorker
from rollout import RolloutManager

import matplotlib as plt

num_workers = 4

# Define the environment creation function outside of the test for pickling
def env_fn():
    return CartPole(max_steps=10)

env = env_fn()

# Define a random policy for testing outside of the test for pickling
def random_policy(state):
    return (env.action_space.sample(), 0)

# Define the worker class outside of the test for pickling
def worker_class(worker_id, env, policy, episodes_completed):
    return RolloutWorker(worker_id, env, policy, episodes_completed)

@pytest.fixture(scope="module")
def rollout_data():
    # Initialize RolloutManager with the random policy
    manager = RolloutManager(
        env_fn=env_fn,  # Pass the class method for creating the env
        worker_class=worker_class,
        policy=random_policy,  # Use the class method for policy
        num_workers=num_workers,  # Using 2 workers for testing
        num_episodes_per_worker=3,  # Each worker will run 3 episodes
    )
    # Run rollout process
    data = manager.rollout()

    manager.shutdown()

    yield data

@pytest.fixture
def rollout_buffer():
    """Fixture to initialize Rollout_Buffer with the CartPole environment."""
    return Rollout_Buffer(env_fn())

def test_initialization(rollout_buffer):
    """Test if Rollout_Buffer initializes properly."""
    assert hasattr(rollout_buffer, 'group_actions'), "group_actions attribute missing."
    assert hasattr(rollout_buffer, 'group_rtgs'), "group_rtgs attribute missing."
    assert hasattr(rollout_buffer, 'group_rewards'), "group_rewards attribute missing."
    assert hasattr(rollout_buffer, 'group_lengths'), "group_lengths attribute missing."

def test_store_and_retrieve(rollout_buffer, rollout_data):
    """Test storing and retrieving rollout data."""

    group_observations, group_actions, group_log_probs, group_rewards, group_lengths = rollout_data

    rollout_buffer.store(group_observations, group_actions, group_log_probs, group_rewards, group_lengths)

    observations, actions, log_probs, rtgs, lengths = rollout_buffer.retrieve()

    assert observations.shape == (num_workers, 3, env.max_steps, 5), "Incorrect shape for stored observations."
    assert actions.shape == (num_workers, 3, env.max_steps, env.action_space.shape[0]), "Incorrect shape for stored actions."
    assert log_probs.shape == (num_workers, 3, env.max_steps), "Incorrect shape for stored log_probs."
    assert rtgs.shape == (num_workers, 3, env.max_steps), "Incorrect shape for stored RTGs."
    assert lengths.shape == (num_workers, 3), "Incorrect shape for stored lengths."

    assert isinstance(actions, torch.Tensor), "Actions should be a torch.Tensor."
    assert isinstance(log_probs, torch.Tensor), "Log_probs should be a torch.Tensor."
    assert isinstance(rtgs, torch.Tensor), "RTGs should be a torch.Tensor."
    assert isinstance(lengths, torch.Tensor), "Lengths should be a torch.Tensor."

def test_rtg_computation(rollout_buffer):
    """Test reward-to-go (RTG) computation for correctness."""
    group_rewards = np.array([[[1, 2, 3], [0, 1, 2]],  # First group
                              [[3, 2, 1], [1, 0, 1]]]) # Second group
    expected_rtgs = np.zeros_like(group_rewards)

    gamma = 0.99
    for i in range(group_rewards.shape[0]):
        for j in range(group_rewards.shape[2] - 1, -1, -1):
            if j == group_rewards.shape[2] - 1:
                expected_rtgs[:, :, j] = group_rewards[:, :, j]
            else:
                expected_rtgs[:, :, j] = group_rewards[:, :, j] + gamma * expected_rtgs[:, :, j + 1]

    computed_rtgs = rollout_buffer.rtg(group_rewards)

    assert np.allclose(computed_rtgs, expected_rtgs, atol=1e-5), "RTG computation incorrect."

# def test_visualize_does_not_crash(rollout_buffer, rollout_data):
#     """Ensure visualize() runs without crashing."""

#     group_observations, group_actions, group_log_probs, group_rewards, group_lengths = rollout_data

#     rollout_buffer.store(group_observations, group_actions, group_log_probs, group_rewards, group_lengths)

#     try:
#         rollout_buffer.visualize(concurrent=False, pause_interval=0.01)
#     except Exception as e:
#         pytest.fail(f"Visualization function crashed with error: {e}")

#     try:
#         rollout_buffer.visualize(concurrent=True, pause_interval=0.01)
#     except Exception as e:
#         pytest.fail(f"Visualization function crashed with error: {e}")

#     rollout_buffer.close()

def test_visualize_without_display_concurrent(rollout_buffer, rollout_data, monkeypatch):
        """
        Test that visualize() in concurrent mode runs without hanging
        or actually displaying a window by monkey-patching plt.pause.
        """
        group_obs, group_actions, group_log_probs, group_rewards, group_lengths = rollout_data
        rollout_buffer.store(group_obs, group_actions, group_log_probs, group_rewards, group_lengths)
        
        # Monkey-patch plt.pause to bypass real-time waiting and display.
        monkeypatch.setattr("matplotlib.pyplot.pause", lambda interval: None)
        
        try:
            rollout_buffer.visualize(concurrent=True, pause_interval=0.01)
        except Exception as e:
            pytest.fail(f"Visualization function crashed with error: {e}")
        finally:
            rollout_buffer.close()
        # Ensure that a figure was created.
        assert rollout_buffer.fig is not None, "Figure was not created during visualization."

def test_visualize_without_display_sequential(rollout_buffer, rollout_data, monkeypatch):
        """
        Test that visualize() in sequential mode runs without error 
        and does not actually require a display by monkey-patching plt.pause.
        """
        group_obs, group_actions, group_log_probs, group_rewards, group_lengths = rollout_data
        rollout_buffer.store(group_obs, group_actions, group_log_probs, group_rewards, group_lengths)
        
        # Monkey-patch plt.pause to avoid delays during testing.
        monkeypatch.setattr("matplotlib.pyplot.pause", lambda interval: None)
        
        try:
            rollout_buffer.visualize(concurrent=False, pause_interval=0.01)
        except Exception as e:
            pytest.fail(f"Visualization function crashed with error: {e}")
        finally:
            rollout_buffer.close()
        # Check that the figure attribute was set.
        assert rollout_buffer.fig is not None, "Figure was not created during visualization."