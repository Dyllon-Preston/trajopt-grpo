import gymnasium as gym
import abc
from typing import Any, Tuple, Dict


"""
Extension of the gym.Env class to include a restart method.
"""

class Env(gym.Env, abc.ABC):
    def __init__(self, env_name: str) -> None:
        """
        Initialize the environment with a name and any necessary setup.
        """
        super().__init__()  # Initialize gym.Env
        self.env_name = env_name

    @abc.abstractmethod
    def reset(self) -> Any:
        """
        Reset the environment to an initial state and return the first observation.
        Gym's reset() method has been modified to return the observation.

        Args:
            None

        Returns:
            observation: The initial observation of the space.
        """
        pass

    @abc.abstractmethod
    def restart(self) -> Any:
        """
        Reset the environment to the state it was in at the last call to reset.
        This is useful when you want to restart from a known initial state.

        Args:
            None

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: The action to perform in the environment.
            
        Returns:
            A tuple containing:
                - observation: The new state observed after taking the action.
                - reward: The reward received after taking the action.
                - done: A boolean flag indicating whether the episode has ended.
                - info: A dictionary of extra diagnostic information.
        """
        pass

    @abc.abstractmethod
    def render(self) -> None:
        """
        Render or display the environment to the screen.
        """
        pass

    # def close(self) -> None:
    #     """
    #     Clean up the environment's resources.
    #     """
    #     pass

    # def seed(self, seed: int) -> None:
    #     """
    #     Seed the environment's random number generator.
    #     """
    #     pass