import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Union
from buffers.buffer import Buffer
import os
import pandas as pd

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

    def save_trajectory(self, path: str):
        """
        Save the trajectory to a CSV file.
        """
        group_observations = self.group_observations.detach().numpy()
        group_actions = self.group_actions.detach().numpy()
        group_lengths = self.group_lengths.detach().numpy().astype(int)


        observations = np.empty((0, group_observations.shape[3]))
        actions = np.empty((0, group_actions.shape[3]))
        episode_id = []

        for i in range(group_lengths.shape[0]):
            for j in range(group_lengths.shape[1]):
                observations = np.vstack((observations, group_observations[i, j, :group_lengths[i, j]]))
                actions = np.vstack((actions, group_actions[i, j, :group_lengths[i, j]]))
                episode_id.extend([j + i*group_lengths.shape[1]] * group_lengths[i, j])

        episode_id = np.array(episode_id).reshape(-1, 1)

        # Header for the CSV file
        header = ['episode_id']
        header += ["observation_{}".format(i) for i in range(group_observations.shape[3])]
        header += ["action_{}".format(i) for i in range(group_actions.shape[3])]

        # Create a dataframe
        df = pd.DataFrame(np.hstack([episode_id, observations, actions]), columns=header)
        df['episode_id'] = df['episode_id'].astype(int)
        # Save the dataframe to a CSV file
        df.to_csv(os.path.join(path, "trajectory.csv"), index=False)



            
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