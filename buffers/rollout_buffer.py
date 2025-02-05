import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Union
from buffers.buffer import Buffer

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
        self.group_log_probs = None
        self.group_values = None
        self.group_rewards = None
        self.group_lengths = None

        self.avg_reward = []

        self.fig = None
        self.axs = None


    def sample(self):
        group_observations, group_actions, group_rewards, group_rtgs, group_lengths = self.rollout_manager.rollout()
        self.store(
            group_observations,
            group_actions,
            group_rewards,
            group_rtgs,
            group_lengths
        )


    def store(
            self,
            group_observations: np.ndarray,
            group_actions: np.ndarray,
            group_rewards: np.ndarray,
            group_rtgs: np.ndarray,
            group_lengths: np.ndarray
            ):
        
        self.group_observations = group_observations
        self.group_actions = group_actions
        self.group_rewards = group_rewards
        self.group_rtgs = group_rtgs
        self.group_lengths = group_lengths

        self.avg_reward.append(np.mean(group_rewards))
        
    # def calculate_rtg(self, 
    #         group_rewards: np.ndarray, 
    #         gamma: float = 0.99):

    #     max_steps = group_rewards.shape[2]

    #     group_rtgs = np.zeros_like(group_rewards)

    #     for i in range(len(group_rewards)): # For each group
    #         for j in range(max_steps - 1, -1, -1): # For each step in the group
    #             if j == max_steps - 1:
    #                 group_rtgs[i,:,j] = group_rewards[i,:,j]
    #             else:
    #                 group_rtgs[i,:,j] = group_rewards[i,:,j] + gamma * group_rtgs[i,:,j+1]
        
    #     return group_rtgs
            
    def retrieve(self):
        return self.group_observations, self.group_actions, self.group_log_probs, self.group_values, self.group_rtgs, self.group_lengths

    def plot_reward(self, ax: plt.Axes = None):
        """
        Plot the average reward per episode.
        
        Parameters:
            ax (plt.Axes): Axis to plot the reward.
        """
        ax.cla()

        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(self.avg_reward)
        ax.set_title("Average Reward per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        plt.pause(0.001)

    def visualize(self, sim_axes = None, title_ax = None, concurrent: bool = False, pause_interval: float = 0.5):
        """
        Visualize the rollout observations sequentially using plt.pause instead of animation.
        
        For non-concurrent mode:
            The function displays one episode at a time across all subplots.
        For concurrent mode:
            The function displays all episodes simultaneously on each subplot.
        
        Parameters:
            concurrent (bool): If True, updates show all episodes concurrently at each time step.
                            If False, updates show one episode at a time across the subplots.
            pause_interval (float): Time in seconds to pause between frame updates.
        """

        env = self.env

        # Plot up to 6 subplots.
        num_groups = self.group_observations.shape[0]
        
         # Grid dimensions for subplots.

        n = min(max(num_groups//3, 1), 2) # Number of rows. Can be 1 or 2.
        m = min(num_groups//n, 3)
        
        # Determine maximum time steps and number of episodes.
        max_steps = self.group_observations.shape[2]
        episodes = self.group_observations.shape[1]
        
        if self.fig is None:
            # Create the figure and subplots.
            fig, axs = plt.subplots(n, m, figsize=(8, 5))
            sim_axes = axs.flatten()

            self.fig = fig
            self.axs = axs
        else:
            fig = self.fig
            sim_axes = self.axs
        
        # Define distinct colors for each episode.
        colors = [plt.cm.jet(i / episodes) for i in range(episodes)]
        
        def update(frame):
            """
            Update function for non-concurrent mode.
            Displays a single episode across subplots by computing the current episode and time step.
            
            Parameters:
                frame (int): Global frame index where each episode shows for max_steps frames.
            """

            # Calculate episode and step based on the frame index.
            episode = frame // max_steps
            step = frame % max_steps
            
            for i, ax in enumerate(sim_axes):
                # Render the observation if within valid length.
                if step < self.group_lengths[i, episode]:
                    obs = self.group_observations[i, episode, step]
                    ax.cla()  # Clear the axis.
                    env.render(
                        ax=ax,
                        observation=obs,
                        color=colors[episode],
                        alpha=2/episodes
                    )
                title_ax.set_title(f"Subplot {i+1}")
            
            fig.suptitle(f"Episode {episode + 1} | Step {step}", fontsize=16)
            plt.tight_layout()
            fig.canvas.draw()
        
        def update_concurrent(frame):
            """
            Update function for concurrent mode.
            Displays all episodes concurrently on each subplot.
            
            Parameters:
                frame (int): The current time step index.
            """
            for i, ax in enumerate(sim_axes):
                # Render observation for each episode that has the frame.
                ax.cla()  # Clear the axis.
                for ep in range(episodes):
                    if frame < self.group_lengths[i, ep]:
                        obs = self.group_observations[i, ep, frame]
                    else:
                        obs = self.group_observations[i, ep, int(self.group_lengths[i, ep]) - 1]
                    env.render(
                        ax=ax,
                        observation=obs,
                        color=colors[ep],
                        alpha=2/episodes
                    )

                title_ax.set_title(f"Subplot {i+1}")
                
            fig.suptitle(f"Step {frame}", fontsize=16)
            plt.tight_layout()
            fig.canvas.draw()
        
        # Loop over frames using plt.pause.
        if concurrent:
            for frame in range(max_steps):
                update_concurrent(frame)
                plt.pause(pause_interval)
        else:
            total_frames = episodes * max_steps
            for frame in range(total_frames):
                update(frame)
                plt.pause(pause_interval)
        

    def close(self):
        """
        Closes the figure window associated with the rollout visualization.
        """
        plt.close(self.fig)