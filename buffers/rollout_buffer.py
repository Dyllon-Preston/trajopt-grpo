import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Union
from buffers.buffer import Buffer
import os

class Rollout_Buffer(Buffer):
    def __init__(
            self,
            rollout_manager,
            rtg: bool = True
            ):
        
        self.rollout_manager = rollout_manager
        self.env = rollout_manager.env_fn()
        self.rtg = rtg

        self.group_observations = None
        self.group_actions = None
        self.group_rewards = None
        self.group_lengths = None

        self.avg_reward = []

        self.fig = None
        self.axs = None

    def load(self, path: str):
        """
        Load the average reward per episode from a CSV file.
        
        Parameters:
            path (str): Path to the CSV file.
        
        Returns:
            int: Number of epochs loaded.
        """
        self.avg_reward = np.loadtxt(os.path.join(path, "reward.csv"), delimiter=",").tolist()
        return len(self.avg_reward)


    def sample(self):
        group_observations, group_actions, group_rewards, group_lengths, group_masks = self.rollout_manager.rollout()
        self.store(
            group_observations,
            group_actions,
            group_rewards,
            group_lengths,
            group_masks
        )

    def store(
            self,
            group_observations: np.ndarray,
            group_actions: np.ndarray,
            group_rewards: np.ndarray,
            group_lengths: np.ndarray,
            group_masks: np.ndarray
            ):
        
        self.group_observations = group_observations
        self.group_actions = group_actions
        self.group_rewards = group_rewards
        self.group_lengths = group_lengths
        self.group_masks = group_masks

        self.avg_reward.append(group_rewards.sum(2).mean().detach().numpy())

            
    def retrieve(self):
        return self.group_observations, self.group_actions, self.group_log_probs, self.group_values, self.group_lengths

    def metadata(self):
        return {
            'avg_reward': float(self.avg_reward[-1]) if len(self.avg_reward) > 0 else None
        }
    
    def save(self, path: str):
        """
        Save the buffer to a file.
        
        Parameters:
            path (str): Path to the file.
        """
        # Save reward csv
        avg_reward = self.avg_reward
        with open(os.path.join(path, "reward.csv"), "w") as f:
            for reward in avg_reward:
                f.write(f"{reward}\n")