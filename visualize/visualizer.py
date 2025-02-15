from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

class Visualizer(ABC):
    def __init__(
            self,
            buffer):
        
        self.fontfamily = "monospace"
        self.env = buffer.rollout_manager.env_fn()

    @abstractmethod
    def initialize(self, metadata: dict):
        """
        Initialize the visualizer with metadata.
        """
        pass

    @abstractmethod
    def plot(self):
        """
        Update the plotable data.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Update the renderable data.
        """
        pass

    @abstractmethod
    def frames(self):
        """
        Generate frames for the visualization.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the visualization to a file.
        """
        pass

    @abstractmethod
    def show(self):
        """
        Display the visualization.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the visualization.
        """
        pass
    
    def _update_sequential(
            self, 
            frame,
            sim_axes,
            title_ax,
            buffer,
            max_episodes,):
        """
        Update function for sequential mode.
        Displays a single episode across subplots by computing the current episode and time step.
        
        Parameters:
            frame (int): Global frame index where each episode shows for max_steps frames.
        """

        group_observations = buffer.group_observations
        group_actions = buffer.group_actions
        group_lengths = buffer.group_lengths

        colors = [plt.cm.jet(i / max_episodes) for i in range(max_episodes)]

        # Calculate episode and step based on the frame index.
        max_steps = group_lengths.shape[2]
        episode = frame // max_steps
        step = frame % max_steps
        
        for i, ax in enumerate(sim_axes):
            # Render the observation if within valid length.
            if step < group_lengths[i, episode]:
                obs = group_observations[i, episode, step]
                ax.cla()  # Clear the axis.
                self.env.render(
                    ax=ax,
                    observation=obs,
                    color=colors[episode],
                    alpha=min(1.0, 2/max_episodes)
                )
            title_ax.set_title(f"Subplot {i+1}")
        
        title_ax.set_title(f"Episode {episode + 1} | Step {step}", fontsize=12, fontfamily=self.fontfamily)

        pass

    def _update_concurrent(
            self, 
            frame: int, 
            sim_axes: list, 
            title_ax: object,
            buffer,
            max_episodes: int,):
        """
        Update function for concurrent mode.
        Displays all episodes concurrently on each subplot.
        
        Parameters:
            frame (int): The current time step index.
        """
    
        group_observations = buffer.group_observations
        group_actions = buffer.group_actions
        group_lengths = buffer.group_lengths

        colors = [plt.cm.jet(i / max_episodes) for i in range(max_episodes)]


        for i, ax in enumerate(sim_axes):
            # Render observation for each episode that has the frame.
            ax.cla()  # Clear the axis.
            for ep in range(max_episodes):
                if frame < group_lengths[i, ep]:
                    obs = group_observations[i, ep, frame]
                else:
                    obs = group_observations[i, ep, int(group_lengths[i, ep]) - 1]
                self.env.render(
                    ax=ax,
                    observation=obs,
                    color=colors[ep],
                    alpha=min(1.0, 2/max_episodes)
                )
            
        title_ax.set_title(f"Step {frame}", fontsize=12, fontfamily=self.fontfamily)