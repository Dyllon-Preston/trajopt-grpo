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
    assert isinstance(env.masscart, float)
    assert isinstance(env.masspole, float)
    assert isinstance(env.length, float)
    assert isinstance(env.gravity, float)
    assert isinstance(env.timestep, float)

def test_cartpole_reset(env):
    """Test if reset initializes the state properly."""
    env.reset()
    state = env._get_obs()
    assert isinstance(state, np.ndarray)
    assert state.shape == (5,)  # Should have 5 state variables
    assert np.all(np.abs(state) <= 1)  # Values should be within reasonable range

def test_cartpole_dynamics(env):
    """Test if the _dynamics function returns a valid next state."""
    state = np.array([0, 0, 0, 1, 0])  # Initial upright state
    action = 0.0  # No control input
    next_state = env._dynamics(state, action)
    
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (5,)

def test_cartpole_step(env):
    """Test if step function works correctly."""
    env.reset()
    action = 0.1  # Small force applied to the cart
    observation, reward, terminated, truncated, info = env.step(action)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_cartpole_step_progression(env):
    """Test if the environment state evolves with steps."""
    env.reset()
    initial_state = env._get_obs().copy()
    
    for _ in range(10):
        env.step(0.1)  # Apply small force repeatedly
    
    new_state = env._get_obs()
    assert not np.array_equal(initial_state, new_state)  # State should change

def test_cartpole_termination(env):
    """Test if the environment terminates when conditions are met."""
    env.reset()
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        
        _, _, terminated, truncated, _ = env.step(0.1)
    
    assert terminated or truncated  # One of them should eventually be True

def test_cartpole_truncation(env):
    """Test if the environment truncates when cart goes out of bounds."""
    env.reset()
    env.state_dict['cartpole'][0] = 2  # Set x-position out of bounds
    _, _, terminated, truncated, _ = env.step(0)

    assert truncated  # Truncation should trigger

def test_cartpole_reward_function(env):
    """Test if the reward function behaves reasonably."""
    env.reset()
    initial_state = {
        'cartpole': np.array([0, 0, 0, 1, 0]),
    }
    env._initial_state = initial_state
    env.restart()
    state, reward, _, _, _ = env.step(0.0)

    assert reward > 0  # Should be positive for staying upright
    assert reward < 5  # Should not be unrealistically high

def test_cartpole_render(env):
    """Test if rendering runs without errors."""
    fig, ax = plt.subplots()
    try:
        env.render(ax)
        assert True  # If no error occurs, test passes
    except Exception as e:
        pytest.fail(f"Render function failed: {e}")

