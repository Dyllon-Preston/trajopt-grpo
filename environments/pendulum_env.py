import numpy as np
from environments.env import Env
import gymnasium as gym
import matplotlib.pyplot as plt


class Pendulum(Env):
    def __init__(
            self,
            env_name: str = 'Pendulum',
            swingup: bool=False,
            mass: float=1.0, # kg
            length: float=0.5, # m
            gravity: float=9.80665, # m/s²
            timestep: float=0.05, # s
            max_steps: int=200,
            ):
        
        # Initialize the environment
        super().__init__(env_name)

        # CartPole parameters
        self.swingup = swingup
        self.mass = mass
        self.length = length
        self.gravity = gravity
        self.timestep = timestep
        self.max_steps = max_steps
        self.max_time = max_steps*timestep

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
        return np.clip(action, -1, 1)

    def _dynamics(
            self,
            state,
            control):
        
        gravity = self.gravity
        mass = self.mass
        length = self.length

        sin_theta, cos_theta, thetadot = state
        thetadot = np.clip(thetadot, -10, 10)

        theta = np.arctan2(sin_theta, cos_theta)

        alpha = 1/(mass*length**2)*(control - mass*gravity*length*np.sin(theta))

        thetadot = thetadot + alpha*self.timestep
        theta = theta + thetadot*self.timestep
        sin_theta_new = np.sin(theta)
        cos_theta_new = np.cos(theta)
        
        next_state = np.array([
            sin_theta_new, 
            cos_theta_new, 
            thetadot
        ]).flatten()

        return next_state
    
    def _propegate_pendulum(
            self, 
            state, 
            control):
        
        state = self._dynamics(state, control)
        self.state_dict['pendulum'] = state

    
    def reset(self):

        if self.swingup:
            theta = np.random.uniform(-np.pi, np.pi)
        else:
            theta = np.random.uniform(np.pi - 0.05, np.pi + 0.05)

        
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
        self._time_balanced = self._time_balanced + self.timestep if cos_theta <= -0.99 else 0

        # Get the observation and info
        observation = self._get_obs()
        info = self._get_info()

        reward = 0 
        reward += self.timestep*np.sum([
            -10*abs(-1 - cos_theta)**0.5, # Reward for keeping the pendulum upright (cos(pi) = -1)
            -0.1*thetadot**2,
            -0.001*(action**2).sum()
        ])

        if self._time_balanced > 0:
            reward += 1

        truncated = self._time > self.max_time
        terminated = self._time_balanced > 5

        # if truncated:
        #     reward -= 50
        # if terminated:
        #     reward += 50

        return observation, reward, truncated, terminated, info

    def render(
            self, 
            ax = None,
            observation=None,
            color='black',
            alpha=1.0):
        
        if observation is None:
            observation = self.state_dict['pendulum']

        sin_theta, cos_theta, thetadot = observation

        theta = np.arctan2(sin_theta, cos_theta)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_frame_on(False)
        ax.set_aspect('equal')

        # Draw pendulum
        pendulum_x = [0, self.length*sin_theta]
        pendulum_y = [0, -self.length*cos_theta]

        ax.plot(pendulum_x, pendulum_y, color=color, alpha=alpha, linewidth=5)

        # Add Mass
        ax.plot(
            self.length*sin_theta, -self.length*cos_theta, 
            marker='o', markersize=10, color=color, alpha=alpha
        )

        # Add the base
        ax.plot([0], [0], marker='o', markersize=10, color='black')
        
        