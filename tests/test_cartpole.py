import pytest
import numpy as np
import matplotlib.pyplot as plt
from environments import CartPole

@pytest.fixture
def env():
    """Creates an instance of the CartPole environment."""
    return CartPole()

def test_cartpole_initialization(env):
    """Test if the CartPole environment initializes correctly."""
    assert env is not None
    assert isinstance(env.masscart, float), \
        f"masscart is {type(env.masscart)}, expected float"
    assert isinstance(env.masspole, float), \
        f"masspole is {type(env.masspole)}, expected float"
    assert isinstance(env.length, float), \
        f"length is {type(env.length)}, expected float"
    assert isinstance(env.gravity, float), \
        f"gravity is {type(env.gravity)}, expected float"
    assert isinstance(env.timestep, float), \
        f"timestep is {type(env.timestep)}, expected float"

def test_cartpole_reset(env):
    """Test if reset initializes the state properly."""
    env.reset()
    state = env._get_obs()
    assert isinstance(state, np.ndarray), \
        f"State type is {type(state)}, expected np.ndarray"
    assert state.shape == (5,), \
        f"State shape is {state.shape}, expected (5,)"
    assert env.observation_space.contains(state), \
        f"State {state} is not in the observation space"

def test_cartpole_dynamics(env):
    """Test if the _dynamics function returns a valid next state."""
    state = np.array([0, 0, 0, 1, 0])  # Initial upright state
    action = 0.0  # No control input
    next_state = env._dynamics(state, action)
    
    assert isinstance(next_state, np.ndarray), \
        f"Next state type is {type(next_state)}, expected np.ndarray"
    assert next_state.shape == (5,), \
        f"Next state shape is {next_state.shape}, expected (5,)"

def test_cartpole_step(env):
    """Test if step function works correctly."""
    env.reset()
    action = 0.1  # Small force applied to the cart
    observation, reward, terminated, truncated, info = env.step(action)

    assert isinstance(observation, np.ndarray), \
        f"Observation type is {type(observation)}, expected np.ndarray"
    assert observation.shape == (5,), \
        f"Observation shape is {observation.shape}, expected (5,)"
    assert isinstance(reward, float), \
        f"Reward type is {type(reward)}, expected float"
    assert isinstance(terminated, bool), \
        f"Terminated type is {type(terminated)}, expected bool"
    assert isinstance(truncated, bool), \
        f"Truncated type is {type(truncated)}, expected bool"
    assert isinstance(info, dict), \
        f"Info type is {type(info)}, expected dict"

def test_cartpole_step_progression(env):
    """Test if the environment state evolves with steps."""
    env.reset()
    initial_state = env._get_obs().copy()
    
    for _ in range(10):
        env.step(0.1)  # Apply small force repeatedly
    
    new_state = env._get_obs()
    assert not np.array_equal(initial_state, new_state), \
        "State should change after applying control input"

def test_cartpole_termination(env):
    """Test if the environment terminates when conditions are met."""
    env.reset()
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        
        _, _, terminated, truncated, _ = env.step(0.1)
    
    assert terminated or truncated, \
        "Environment should terminate when conditions are met"

def test_cartpole_reward_function(env):
    """Test if the reward function behaves reasonably."""
    env.reset()
    initial_state = {
        'cartpole': np.array([0, 0, 0, 1, 0]),
    }
    env._initial_state = initial_state
    env.restart()
    state, reward, _, _, _ = env.step(0.0)

    assert reward > 0, \
        f"Reward is {reward}, expected positive for being upright"
    assert reward < 5, \
        f"Reward is {reward}, expected less than 5 for being upright"

def test_cartpole_render(env):
    """Test if rendering runs without errors."""
    fig, ax = plt.subplots()
    try:
        env.render(ax)
        assert True
    except Exception as e:
        pytest.fail(f"Render function failed: {e}")

