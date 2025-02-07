import numpy as np
from environments.env import Env
from matplotlib import patches
import gymnasium as gym

class CartPole(Env):
    def __init__(
            self,
            env_name: str = 'CartPole',
            mass: float=1.0, # kg
            length: float=0.5, # m
            gravity: float=9.80665, # m/s²
            timestep: float=0.05, # s
            max_steps: int=200,
            ):
        
        # Initialize the environment
        super().__init__(env_name)

        # CartPole parameters
        self.mass = mass
        self.length = length
        self.gravity = gravity
        self.timestep = timestep
        self.max_steps = max_steps

        self.state_dict = {
            'pendulum': np.zeros(3)
        }

        # Define the observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32)
        
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        


    def _wrap_action(self, action):
        return 5*np.clip(action, -1, 1)

    def _dynamics(
            self,
            state,
            control):
        
        gravity = self.gravity
        mass = self.mass
        length = self.length

        
        sin_theta, cos_theta, thetadot = state
        theta = np.arctan2(sin_theta, cos_theta)

        alpha = 1/(mass*length**2)*(control - mass*gravity*length*np.sin(theta))

        thetadot_new = thetadot + alpha*self.timestep
        theta_new = theta + thetadot*self.timestep
        sin_theta_new = np.sin(theta_new)
        cos_theta_new = np.cos(theta_new)
        
        return np.array([sin_theta_new, cos_theta_new, thetadot_new])
    
    def _propegate_pendulum(
            self, 
            state, 
            control):
        
        state = self._dynamics(state, control)
        self.state_dict['pendulum'] = state

    
    def reset(self):

        theta = np.random.uniform(-np.pi, np.pi)
        
        self.state_dict['pendulum'] = np.array([
            np.sin(theta),
            np.cos(theta),
            0
        ])

        self._initial_state = self.state_dict.copy()

        self._steps = 0
        self._time = 0
        self._time_balanced = 0

        return self._get_obs(), self._get_info()

    def restart(self):
        self.state_dict = self._initial_state.copy()

        self._steps = 0
        self._time = 0
        self._time_balanced = 0

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        return self.state_dict['pendulum']
    
    def _get_info(self):
        return {
            'time_balanced': self._time_balanced,
        }

    def step(self, action):

        action = self._wrap_action(action)
        self._propegate_pendulum(self.state_dict['pendulum'], action)

        state = self.state_dict['pendulum']

        sin_theta, cos_theta, thetadot = state
        theta = np.arctan2(sin_theta, cos_theta)

        # Increment the time
        self._steps += 1
        self._time += self.timestep
        self._time_balanced = self._time_balanced + self.timestep if np.abs(theta) < 0.05 else 0

        # Get the observation and info
        observation = self._get_obs()
        info = self._get_info()

        reward = 0 
        reward += np.sum([
            -theta**2,
            -0.1*thetadot**2,
            -0.001*action**2
        ])

        truncated = self._time > self.max_time
        terminated = self._time_balanced > 5

        if truncated:
            reward -= 50
        if terminated:
            reward += 50

        return observation, reward, truncated, terminated, info

    def render(
            self, 
            ax,
            observation=None,
            color='black',
            alpha=1.0):
        pass