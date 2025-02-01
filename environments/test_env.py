import numpy as np
from environments.env import Env

"""
A test environment for the purpose of testing the environment class.
"""

class EnvTest(Env):
    def __init__(
            self,
            timestep: float=0.05, # s
            max_time: float=10.0, # s      
            ):
        
        self.timestep = timestep
        self.max_time = max_time

        self.state_dict = {
            'test': np.zeros(2)
        }

        self._initial_state = None

        self._time = 0
        self._steps = 0

    def _dynamics(
            self,
            state,
            control):
        return state + control*self.timestep
    
    def _propegate_test(
            self, 
            state, 
            control):
        
        state = self._dynamics(state, control)
        self.state_dict['test'] = state
    
    def reset(self):
        self.state_dict['test'] = np.random.rand(2)
        self._initial_state = self.state_dict.copy()
    
    def restart(self):
        self.state_dict = self._initial_state.copy()

    def _get_obs(self):
        return self.state_dict['test']

    def _get_info(self):
        return {
            'time': self._time,
        }
    
    def step(self, action):
        self._propegate_test(self.state_dict['test'], action)

        state = self.state_dict['test']

        self._steps += 1
        self._time += self.timestep

        observation = self._get_obs()
        info = self._get_info()
        reward = 0

        reward += self.timestep*np.sum([
            1, # Reward for staying alive
            1/(1 + np.sum(state**2)), # Reward for staying near the center
            1/(1 + np.sum(action**2)) # Reward for using less control
        ])

        truncated = bool(state[0] > 1)
        terminated = self._time > self.max_time
        return observation, reward, terminated, truncated, info
    
    def render(
            self, 
            ax):
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        ax.scatter(self.state_dict['test'][0], self.state_dict['test'][1], color='black', s=50, zorder=3)