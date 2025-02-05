import os
import gymnasium as gym
import torch

from buffers import Buffer

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Trainer():
    def __init__(
            self,
            test_name: str,
            checkpoint_name: str,
            buffer: Buffer,
            policy: torch.nn.Module,
            ref_model: torch.nn.Module,
            algorithm: str,

            epochs: int = 100,

            save_freq: int = 10,
            render_freq: int = 10,
            render_reward: bool = False,
            render_visuals: bool = False,
            log_freq: int = 1,

            ):
        
        self.test_name = test_name # name of the test (usually the date)
        self.checkpoint_name = checkpoint_name # name of the checkpoint (an important checkpoint in the policy training)

        
        self.buffer = buffer
        self.policy = policy
        self.ref_model = ref_model
        self.algorithm = algorithm

        self.epochs = epochs

        self.env = self.buffer.rollout_manager.env_fn()
        self.env_name = self.env.env_name

        self.save_freq = save_freq
        self.render_freq = render_freq
        self.log_freq = log_freq

        self.initialize_dashboard()

        


        if not os.path.exists("/archive"):
            os.makedirs("/archive")

    def initialize_dashboard(self):
        fig = plt.figure(figsize=(16, 8))
        # Main grid: left for simulations and right for reward plot/table.
        gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.6, 0.4], wspace=0.15)

        # Left panel: Create a nested GridSpec with 2 rows.
        # The top row is for the super title, the bottom row for the 2x2 simulation plots.
        gs_left_main = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0], height_ratios=[0.01, 0.99], hspace=0)
        ax_sim_title = fig.add_subplot(gs_left_main[0])
        ax_sim_title.axis("off")
        ax_sim_title.set_title("Simulation Plots", fontsize=16)
        self.ax_sim_title = ax_sim_title

        # 2x2 grid for simulations.
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_left_main[1],
                                                    hspace=0.1, wspace=0.3)
        ax_sim1 = fig.add_subplot(gs_left[0, 0])
        ax_sim2 = fig.add_subplot(gs_left[0, 1])
        ax_sim3 = fig.add_subplot(gs_left[1, 0])
        ax_sim4 = fig.add_subplot(gs_left[1, 1])

        self.sim_axes = [ax_sim1, ax_sim2, ax_sim3, ax_sim4]

        # Right panel: vertical grid for reward plot (upper) and table (lower).
        gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[1],
                                                    height_ratios=[0.8, 0.2], hspace=0.3)

        ax_reward = fig.add_subplot(gs_right[0])
        self.reward_ax = ax_reward

        ax_table = fig.add_subplot(gs_right[1])
        ax_table.axis("off")
        table_data = [
            ["128x128"],            # Model dimensions
            ["checkpoint_01.pth"],  # Model checkpoint
            ["2023-10-05"],          # Creation date
            ["Today"]          # Creation date
        ]
        # Use bbox to let the table fill the axis.
        ax_table.table(cellText=table_data,
                    rowLabels=["Publish Date", 
                               "Environment Name", 
                               "Model Checkpoint", 
                               "Model Description"],
                    loc="center", bbox=[0.32, 0, 0.65, 1])


    def run(self):

        for i in range(self.epochs):
            self.buffer.sample()

            self.algorithm.learn(
                group_observations = self.buffer.group_observations,
                group_actions = self.buffer.group_actions,
                group_rewards = self.buffer.group_rewards,
            )

            breakpoint()

            if i % self.render_freq == 0:
                self.buffer.plot_reward(self.reward_ax)
                self.buffer.visualize(sim_axes = self.sim_axes, title_ax = self.ax_sim_title)

            if i % self.save_freq == 0:
                if not os.path.exists(f"/archive/{self.env_name}/{self.test_name}/{self.checkpoint_name}"):
                    os.makedirs(f"/archive/{self.env_name}/{self.test_name}/{self.checkpoint_name}")
                self.policy.save()

                # Save config file

    def shutdown(self):
        # self.env.close()
        self.buffer.rollout_manager.shutdown()
                
    
    def config(self):
        return {
            "env_name": self.env_name,
            "test_name": self.test_name,
            "checkpoint_name": self.checkpoint_name,
            "num_workers": self.num_workers,
            "num_episodes_per_worker": self.num_episodes_per_worker,
            "epochs": self.epochs,
            "save_freq": self.save_freq,
            "render_freq": self.render_freq,
            "log_freq": self.log_freq,
        }


    

        