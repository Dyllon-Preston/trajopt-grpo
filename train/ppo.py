import torch
import copy

"""
Proximal Policy Optimization Algorithms
https://arxiv.org/abs/1707.06347

"""

class PPO:
    
    def __init__(
        self,
        group_size: int,
        epsilon: float,
        policy: torch.nn.Module,
        ref_model: torch.nn.Module,
        beta: float,
        updates_per_iter: int,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Initialize GRPO with the specified parameters.
        """
        self.group_size = group_size
        self.epsilon = epsilon
        self.policy = policy
        self.ref_model = ref_model
        self.beta = beta
        self.updates_per_iter = updates_per_iter
        self.optimizer = optimizer

    def train(self, group_observations, group_actions, group_log_probs, group_rewards):
        pass