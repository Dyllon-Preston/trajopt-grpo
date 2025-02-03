import numpy as np
import torch
import matplotlib.pyplot as plt

class Rollout_Buffer():
    def __init__(
            self,
            env):
        
        self.env = env

        self.group_observations = None
        self.group_actions = None
        self.group_rtgs = None
        self.group_rewards = None
        self.group_lengths = None

        self.fig = None
        self.axs = None

        
    def store(
            self,
            group_observations: np.ndarray,
            group_actions: np.ndarray,
            group_log_probs: np.ndarray,
            group_rewards: np.ndarray,
            group_lengths: np.ndarray
            ):
        
        group_rtgs = self.rtg(group_rewards)

        self.group_observations = torch.tensor(group_observations, dtype=torch.float32)
        self.group_actions = torch.tensor(group_actions, dtype=torch.float32)
        self.group_log_probs = torch.tensor(group_log_probs, dtype=torch.float32)
        self.group_rtgs = torch.tensor(group_rtgs, dtype=torch.float32)
        self.group_lengths = torch.tensor(group_lengths, dtype=torch.float32)



    def rtg(self, 
            group_rewards: np.ndarray, 
            gamma: float = 0.99):

        max_steps = group_rewards.shape[2]

        group_rtgs = np.zeros_like(group_rewards)

        for i in range(len(group_rewards)): # For each group
            for j in range(max_steps - 1, -1, -1): # For each step in the group
                if j == max_steps - 1:
                    group_rtgs[i,:,j] = group_rewards[i,:,j]
                else:
                    group_rtgs[i,:,j] = group_rewards[i,:,j] + gamma * group_rtgs[i,:,j+1]
        
        return group_rtgs
            
    def retrieve(self):
        return self.group_observations, self.group_actions, self.group_log_probs, self.group_rtgs, self.group_lengths

    def visualize(self, concurrent: bool = False, pause_interval: float = 0.5):
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
            axs = axs.flatten()

            self.fig = fig
            self.axs = axs
        else:
            fig = self.fig
            axs = self.axs
        
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
            
            for i, ax in enumerate(axs):
                # Render the observation if within valid length.
                if step < self.group_lengths[i, episode]:
                    obs = self.group_observations[i, episode, step]
                    ax.cla()  # Clear the axis.
                    self.env.render(
                        ax=ax,
                        observation=obs,
                        color=colors[episode],
                        alpha=2/episodes
                    )
                ax.set_title(f"Subplot {i+1}")
            
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
            for i, ax in enumerate(axs):
                # Render observation for each episode that has the frame.
                ax.cla()  # Clear the axis.
                for ep in range(episodes):
                    if frame < self.group_lengths[i, ep]:
                        obs = self.group_observations[i, ep, frame]
                    else:
                        obs = self.group_observations[i, ep, int(self.group_lengths[i, ep]) - 1]
                    self.env.render(
                        ax=ax,
                        observation=obs,
                        color=colors[ep],
                        alpha=2/episodes
                    )

                ax.set_title(f"Subplot {i+1}")
                
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