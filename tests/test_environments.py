import pytest
import numpy as np
from environments.test_env import EnvTest  # Assuming you have a test environment

@pytest.fixture
def env():
    """Creates an instance of the test environment."""
    return EnvTest()

def test_environment_initialization(env):
    """Ensure the environment initializes correctly."""
    assert env is not None  # Environment should be instantiated
    assert hasattr(env, 'reset')  # Should have a reset method
    assert hasattr(env, 'step')  # Should have a step method

def test_environment_reset(env):
    """Test if environment resets correctly and returns an initial state."""
    env.reset()
    initial_state = env._initial_state
    assert initial_state is not None  # Initial state should exist
    assert isinstance(initial_state, dict)  # Assuming state is returned as a dictionary

def test_environment_step(env):
    """Test if the environment step function works correctly."""
    env.reset()  # Reset before stepping
    action = np.array([0.1, -0.2])  # Example action (Modify based on action space)
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(observation, np.ndarray)  # Observation should be a numpy array
    assert isinstance(reward, float)  # Reward should be a float
    assert isinstance(terminated, bool)  # terminated should be a boolean
    assert isinstance(truncated, bool)  # truncated should be a boolean
    assert isinstance(info, dict)  # Info should be a dictionary

def test_environment_termination(env):
    """Ensure the environment correctly handles episode termination."""
    env.reset()
    done = False
    while not done:
        action = np.random.rand(2)
        _, _, terminated, truncated, _ = env.step(action)  # Example neutral action
        done = terminated | truncated

    assert done is True  # Ensure the environment eventually terminates