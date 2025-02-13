import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from datetime import date
from typing import Any, List, Iterable


class Dashboard:
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
        reward_ax_solo (plt.Axes): Axis for reward plot when not rendering simulations.
        table_ax (plt.Axes): Axis for the metadata table.
    """

    def __init__(
        self,
        test_name: str,
        checkpoint_name: str,
        env_name: str,
        algorithm_name: str,
        policy_metadata: dict,
        creation_date: str = "-",
        render: bool = True,
        max_episodes_per_render: int = 5,
    ) -> None:
        """
        Initialize the Dashboard.
        
        Parameters:
            test_name (str): The name of the test.
            checkpoint_name (str): The model checkpoint name.
            env_name (str): The environment name.
            algorithm_name (str): The name of the algorithm used.
            policy_metadata (dict): Metadata from the policy (e.g., {'num_parameters': 123456}).
            creation_date (str): Creation date of the model/dashboard.
            render (bool): If True, creates a full dashboard with simulation plots; 
                           otherwise, only a simple reward plot is created.
            max_episodes_per_render (int): Maximum number of episodes to render in simulation plots.
        """
        self.test_name = test_name
        self.checkpoint_name = checkpoint_name
        self.env_name = env_name
        self.algorithm_name = algorithm_name
        self.policy_metadata = policy_metadata
        self.creation_date = creation_date
        self.render = render
        self.max_episodes_per_render = max_episodes_per_render

        if self.render:
            self._initialize_dashboard()
        else:
            # Create a simple figure with a single reward axis.
            self.fig, self.reward_ax_solo = plt.subplots(figsize=(8, 6))

    def _initialize_dashboard(self) -> None:
        """
        Create the full dashboard layout with simulation plots, reward plot, and metadata table.
        """
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

        self.table_ax = self.fig.add_subplot(gs_right[1])
        self.table_ax.axis("off")

        # Define metadata
        metadata = {
            "Experiment Date": str(date.today()),
            "Environment Name": self.env_name,
            "Model Checkpoint": f"{self.test_name} | {self.checkpoint_name}",
            "Model Parameters": str(self.policy_metadata.get("num_parameters", "N/A")),
            "Algorithm": self.algorithm_name,
        }

        # Compute max label width for alignment
        max_label_length = max(len(label) for label in metadata.keys())

        # Format text with padding
        formatted_text = "\n".join(
            f"{label.ljust(max_label_length)}: {value}" for label, value in metadata.items()
        )

        # Display text in the subplot
        self.table_ax.text(
            -0.01, 0.85,  # Position (left-aligned, top-down)
            formatted_text,
            fontsize=10,
            fontfamily=fontfamily,  # Monospaced font ensures alignment
            verticalalignment="top",
        )

    def _show(self) -> None:
        """
        Debug function to display the dashboard.
        """
        # self.fig.savefig("dashboard.png", dpi=300, bbox_inches="tight")
        self.fig.show()

    def update_reward_plot(self, reward_data: Iterable[Any]) -> None:
        """
        Update the reward plot with new data.
        
        Parameters:
            reward_data (Iterable[Any]): A sequence of reward values.
        """
        # Choose the appropriate axis based on rendering mode.
        ax = self.reward_ax if self.render else self.reward_ax_solo
        ax.clear()
        ax.plot(reward_data, marker="o", linestyle="-", color="blue")
        ax.set_title("Reward Progress")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reward")
        self.fig.canvas.draw_idle()

    def update_simulation_plots(self, simulation_data: List[Iterable[Any]], title: str = "Simulation Plots") -> None:
        """
        Update the simulation plots with new data.
        
        Parameters:
            simulation_data (List[Iterable[Any]]): A list of data sequences for each simulation subplot.
            title (str): The title to display above the simulation plots.
        """
        if not self.render:
            return

        self.ax_sim_title.set_title(title, fontsize=16)
        for ax, sim_data in zip(self.sim_axes, simulation_data):
            ax.clear()
            # Plot each simulation data (customize as needed).
            ax.plot(sim_data, marker=".", linestyle="-", color="green")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
        self.fig.canvas.draw_idle()

    def refresh(self) -> None:
        """
        Redraw the dashboard. Call after updating any plots.
        """
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """
        Display the dashboard.
        """
        plt.show()
