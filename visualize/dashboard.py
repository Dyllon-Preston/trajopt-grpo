from .visualizer import Visualizer

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from datetime import date
from typing import Any, List, Iterable
import os
import numpy as np
import io
from PIL import Image

class Dashboard(Visualizer):
    """
    Dashboard for visualizing training simulations, rewards, and metadata.
    
    This class encapsulates a matplotlib figure that displays:
      - A 2x2 grid of simulation plots.
      - A reward plot.
      - A table of metadata.
    
    Attributes:
        fig (plt.Figure): The main figure containing the dashboard.
        sim_axes (List[plt.Axes]): List of axes for simulation plots.
        ax_sim_title (plt.Axes): Axis for the simulation plots title.
        reward_ax (plt.Axes): Axis for the reward plot (detailed view).
        table_ax (plt.Axes): Axis for the metadata table.
    """

    def __init__(
        self,
        env: object,
        buffer: object,
        max_episodes_per_render: int = 5,
        dpi: int = 200,
        skip: int = 1,
    ) -> None:
        """
        Initialize the Dashboard.
        
        Parameters:
            env (object): The environment object.
            buffer (object): The buffer object.
            max_episodes_per_render (int): Maximum number of episodes to display in the simulation plots.
        """

        super().__init__(buffer)

        self.env = env
        self.buffer = buffer
        self.max_episodes_per_render = max_episodes_per_render
        self.dpi = dpi
        self.skip = skip

    def initialize(self, metadata: dict) -> None:
        """
        Create the full dashboard layout with simulation plots, reward plot, and metadata table.
        """

        metadata
        env_name = metadata.get("env_name", "N/A")
        test_name = metadata.get("test_name", "N/A")
        checkpoint_name = metadata.get("checkpoint_name", "N/A")
        algorithm_name = metadata['algorithm'].get("algorithm", "N/A")
        author_name = metadata['publisher'].get("author", "N/A")
        num_parameters = metadata['policy'].get("num_parameters", "N/A")
        if isinstance(num_parameters, int):
            num_parameters = f"{num_parameters:,}"


        # If fig exists, close it.
        if hasattr(self, "fig"):
            plt.close(self.fig)

        self.fig = plt.figure(figsize=(12, 6))

        fontfamily="monospace"

        # Main grid: left for simulation plots, right for reward plot and table.
        gs_main = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[0.6, 0.4], wspace=0.05)

        # Left panel: Nested grid with a title and a 2x2 grid for simulation plots.
        gs_left_main = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_main[0], height_ratios=[0.01, 0.99], hspace=0
        )
        self.ax_sim_title = self.fig.add_subplot(gs_left_main[0])
        self.ax_sim_title.axis("off")
        self.ax_sim_title.set_title("Simulation Plots", fontsize=12, fontfamily=fontfamily)

        gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_left_main[1], hspace=0.1, wspace=-0.17
        )
        self.sim_axes = [
            self.fig.add_subplot(gs_left[0, 0]),
            self.fig.add_subplot(gs_left[0, 1]),
            self.fig.add_subplot(gs_left[1, 0]),
            self.fig.add_subplot(gs_left[1, 1]),
        ]

        # Set axis properties.
        for ax in self.sim_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        # Right panel: Vertical grid for the reward plot and a metadata table.
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_main[1], height_ratios=[0.7, 0.3], hspace=0.25, wspace=0.1
        )
        self.reward_ax = self.fig.add_subplot(gs_right[0])
        self.reward_ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        self.reward_ax.set_title("Average Total Reward per Episode", fontfamily=fontfamily)
        self.reward_ax.set_xlabel("Epoch", fontfamily=fontfamily)
        self.reward_ax.set_ylabel("Reward", fontfamily=fontfamily)
        self.reward_ax.grid(True)

        self.table_ax = self.fig.add_subplot(gs_right[1])
        self.table_ax.axis("off")

        # Define table_data
        table_data = {
            "Experiment Date": str(date.today()),
            "Environment Name": env_name,
            "Model Checkpoint": f"{test_name} | {checkpoint_name}",
            "Model Parameters": num_parameters,
            "Algorithm": algorithm_name,
            "Author": author_name,
        }

        # Compute max label width for alignment
        max_label_length = max(len(label) for label in table_data.keys())

        # Format text with padding
        formatted_text = "\n".join(
            f"{label.ljust(max_label_length)}: {value}" for label, value in table_data.items()
        )

        # Display text in the subplot
        self.table_ax.text(
            -0.01, 0.85,  # Position (left-aligned, top-down)
            formatted_text,
            fontsize=10,
            fontfamily=fontfamily,  # Monospaced font ensures alignment
            verticalalignment="top",
        )

    def _update_reward_plot(self) -> None:
        """
        Update the reward plot with new data.
        """
        ax = self.reward_ax
        ax.cla()

        avg_reward = self.buffer.avg_reward

        # Moving average
        window_size = 5
        if len(avg_reward) > window_size:
            avg_reward_ma = np.convolve(avg_reward, np.ones(window_size) / window_size, mode="valid")
            x = np.arange(window_size-1, len(avg_reward))
        else:
            avg_reward_ma = 0
            x = 0


        ax.plot(avg_reward, alpha=0.8, label="Average Reward")
        ax.plot(x, avg_reward_ma, color="red", label=f"Moving Average ({window_size})", alpha=0.8)
        ax.grid(True)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_xlabel("Epoch", fontfamily=self.fontfamily)
        ax.set_ylabel("Reward", fontfamily=self.fontfamily)
        ax.set_title("Average Total Reward per Episode", fontfamily=self.fontfamily)
        ax.legend(prop={'size': 8, 'family': f'{self.fontfamily}'}, loc="upper left")

        

    def plot(
        self,
        show: bool = True,
        ) -> None:
        """
        Update plotable data and display the dashboard.
        """
        self._update_reward_plot()
        if show:
            plt.pause(0.1)
        

    def render(
            self, 
            show = True,
            ) -> None:
        """
        Update renderable data and display the dashboard.
        """

        group_lengths = self.buffer.group_lengths

        max_frame = int(group_lengths[:,:self.max_episodes_per_render].max())
        for frame in range(0, max_frame, self.skip):
            self._update_concurrent(
                frame,
                sim_axes=self.sim_axes,
                title_ax=self.ax_sim_title,
                buffer=self.buffer,
                max_episodes=self.max_episodes_per_render,)
            if show:
                plt.pause(0.1)

    def frames(self):
        """
        Return the frames for the dashboard.
        """

        self._update_reward_plot()

        group_lengths = self.buffer.group_lengths

        max_frame = int(group_lengths[:,:self.max_episodes_per_render].max())

        frames = []
        for frame in range(max_frame):
            self._update_concurrent(
                frame,
                sim_axes=self.sim_axes,
                title_ax=self.ax_sim_title,
                buffer=self.buffer,
                max_episodes=self.max_episodes_per_render,)
            # Return frame as np array with tight bounding box
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", bbox_inches="tight", dpi=self.dpi)
            buf.seek(0)

            # Convert to PIL image
            pil_image = Image.open(buf)
            frames.append(pil_image)
        
        return frames

    def show(self) -> None:
        """
        Display the dashboard.
        """
        plt.show()

    def metadata(self) -> dict:
        """
        Return the metadata of the dashboard.
        """
        metadata = {
            "max_episodes_per_render": self.max_episodes_per_render,
        }
        return metadata
    
    def save(self, path: str) -> None:
        """
        Save the dashboard to a file.
        """
        self.fig.savefig(os.path.join(path, "dashboard.png"), bbox_inches="tight")

    def close(self) -> None:
        """
        Close the dashboard.
        """
        plt.close(self.fig)