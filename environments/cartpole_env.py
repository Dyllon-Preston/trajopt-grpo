import numpy as np
from environments.env import Env
from matplotlib import patches
import gymnasium as gym

class CartPole(Env):
    def __init__(
            self,
            env_name: str = 'CartPole',
            masscart: float=1.0, # kg
            masspole: float=0.1, # kg
            length: float=0.5, # m
            gravity: float=9.80665, # m/s²
            timestep: float=0.05, # s
            max_steps: int=200,
            ):
        
        # Initialize the environment
        super().__init__(env_name)

        # CartPole parameters
        self.masscart = masscart
        self.masspole = masspole
        self.length = length
        self.gravity = gravity
        self.timestep = timestep
        self.max_steps = max_steps
        self.max_time = max_steps*timestep

        self._initial_state = None

        self._steps = 0
        self._time = 0
        self._time_balanced = 0

        self.state_dict = {
            'cartpole': np.zeros(5)
        }

        # Define the observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)

    def _wrap_action(self, action):
        return np.clip(action, -1, 1)

    def _dynamics(
            self,
            state,
            control):
        
        x, xdot, sin_theta, cos_theta, thetadot = state

        control = control[0]

        masscart = self.masscart
        masspole = self.masspole
        length = self.length
        gravity = self.gravity
        timestep = self.timestep

        theta = np.arctan2(sin_theta, cos_theta)

        # Angular acceleration of the pendulum
        alpha = (gravity*sin_theta + cos_theta*(
                (-control - masspole*length*thetadot**2*sin_theta)/(masscart + masspole)))/(
                length*(4/3 - (masspole*cos_theta**2)/(masscart + masspole)))
        
        # Translational acceleration
        a = (control + masspole*length*(thetadot**2*sin_theta - alpha*cos_theta))/(masscart + masspole)

        # Semi-implicit Euler integration
        xdot = xdot + a*timestep
        x = x + xdot*timestep

        thetadot = thetadot + alpha*timestep
        theta = theta + thetadot * timestep

        next_state = np.array([
            x,
            xdot,
            np.sin(theta),
            np.cos(theta),
            thetadot
        ])
        return next_state
    
    def _propegate_cartpole(
            self, 
            state, 
            control):
        
        state = self._dynamics(state, control)
        self.state_dict['cartpole'] = state
    
    def reset(self):
        theta = np.random.uniform(-np.pi, np.pi)
        self.state_dict['cartpole'] = np.array([
            0,
            0,
            np.sin(theta),
            np.cos(theta),
            0
        ])

        self._initial_state = self.state_dict.copy()

        self._steps = 0
        self._time = 0

        return self._get_obs(), self._get_info()

    def restart(self):
        self.state_dict = self._initial_state.copy()

        self._steps = 0
        self._time = 0

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        return self.state_dict['cartpole']
    
    def _get_info(self):
        return {
            'time_balanced': self._time_balanced,
        }

    def step(self, action):

        reward = 0

        action = self._wrap_action(action)

        self._propegate_cartpole(self.state_dict['cartpole'], action)

        state = self.state_dict['cartpole']
        x, xdot, sin_theta, cos_theta, thetadot = state
        theta = np.arctan2(sin_theta, cos_theta)

        # Increment the time steps
        self._steps += 1
        self._time += self.timestep
        self._time_balanced = self._time_balanced + self.timestep if np.abs(theta) < 0.05 else 0

        # Get the observation and info
        observation = self._get_obs()
        info = self._get_info()

        reward += self.timestep*np.sum([
            1, # Reward for staying alive
            1/(1 + x**2), # Reward for staying near the center
            1/(1 + theta**2), # Reward for keeping the pole upright
            1/(1 + np.sum(action**2)) # Reward for using less control
        ])

        truncated = (np.abs(x) > 1) or (self._time > self.max_time)
        terminated = self._time_balanced > 5

        if truncated:
            reward -= 200
        if terminated:
            reward += 200

        return observation, reward, terminated, truncated, info

    def render(
            self, 
            ax,
            observation=None,
            color='black',
            alpha=1.0):
        """
        Plot the state of the cartpole
        """

        if observation is None:
            observation = self.state_dict['cartpole']

        x, xdot, sin_theta, cos_theta, thetadot = observation
        theta = np.arctan2(sin_theta, cos_theta)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_frame_on(False)
        ax.set_aspect('equal')

        cart_width = 0.3
        cart_height = 0.1

        # Draw floor
        ax.axhline(y=0, color='black', lw=1, linestyle='-.', alpha=1)

        # Draw cart
        cart = patches.Rectangle((x - cart_width/2, -cart_height/2), cart_width, cart_height, 
                                color=color, ec=color, lw=2, alpha=alpha)

        ax.add_patch(cart)

        # Compute pole end position
        pole_x = x + self.length * np.sin(theta)
        pole_y = self.length * np.cos(theta)
        
        # Draw pole
        ax.plot([x, pole_x], [0, pole_y], color=color, lw=4, solid_capstyle='round', alpha=alpha)
        
        # Draw pivot point
        ax.scatter([x], [0], color=color, s=50, zorder=3, alpha=alpha)