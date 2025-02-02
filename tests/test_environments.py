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
    
    # Check the types of the outputs
    assert isinstance(observation, np.ndarray), \
        f"Observation type is {type(observation)}, expected np.ndarray"
    assert isinstance(reward, float), \
        f"Reward type is {type(reward)}, expected float"
    assert isinstance(terminated, bool), \
        f"Terminated type is {type(terminated)}, expected bool"
    assert isinstance(truncated, bool), \
        f"Truncated type is {type(truncated)}, expected bool"
    assert isinstance(info, dict), \
        f"Info type is {type(info)}, expected dict"

def test_environment_termination(env):
    """Ensure the environment correctly handles episode termination."""
    env.reset()
    done = False

    limit = env.max_steps + 1  # Maximum number of steps before termination
    i = 0
    while not done and i < limit:
        action = np.random.rand(2)
        _, _, terminated, truncated, _ = env.step(action)  # Example neutral action
        done = terminated | truncated
        i += 1

    assert done is True, \
        f"Environment did not terminate after {env.max_steps} steps"