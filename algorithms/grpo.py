from algorithms.algorithm import Algorithm

import torch
import copy
import os

"""
DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
https://arxiv.org/abs/2402.03300
"""

class GRPO(Algorithm):
    """
    Group Relative Policy Optimization (GRPO) implementation.

    Attributes:
        group_size (int): Number of groups.
        epsilon (float): Clipping parameter for the policy update.
        policy (torch.nn.Module): The policy network.
        ref_model (torch.nn.Module): Reference model for KL divergence calculation.
        beta (float): Coefficient for KL divergence penalty.
        updates_per_iter (int): Number of training updates per iteration.
        optimizer (torch.optim.Optimizer): Optimizer for policy updates.
        old_policy (torch.nn.Module): Frozen copy of the policy used for stable updates.
    """
    def __init__(
        self,
        epsilon: float,
        beta: float,
        gamma: float,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ref_model: torch.nn.Module = None,
        updates_per_iter: int = 10,
    ):
        """
        Initialize GRPO with the specified parameters.
        """
        self.epsilon = epsilon
        self.policy = policy
        self.ref_model = ref_model
        self.beta = beta
        self.gamma = gamma
        self.updates_per_iter = updates_per_iter
        self.optimizer = optimizer

        # Create a deep copy of the policy for stable old_policy
        self.old_policy = copy.deepcopy(self.policy)

    def learn(self, buffer) -> None:
        

        group_observations = buffer.group_observations
        group_actions = buffer.group_actions
        group_rewards = buffer.group_rewards
        group_masks = buffer.group_masks

        group_masks_float = group_masks.float()


        num_groups, num_episodes, max_steps, _ = group_observations.shape

        gamma = self.gamma

        # Calculate rtg
        group_rtgs = torch.zeros_like(group_rewards)
        for i in reversed(range(max_steps)):
            if i < max_steps - 1:
                group_rtgs[:, :, i] = (
                    group_rewards[:, :, i] * group_masks_float[:, :, i]
                    + gamma * group_rtgs[:, :, i + 1] * group_masks_float[:, :, i + 1]
                )
            else:
                group_rtgs[:, :, i] = group_rewards[:, :, i] * group_masks_float[:, :, i]
        



        group_observations = group_observations.view(
            group_observations.size(0),
            -1,
            group_observations.size(-1)
        )
        group_actions = group_actions.view(
            group_actions.size(0),
            -1,
            group_actions.size(-1)
        )
        group_rewards = group_rewards.view(
            group_rewards.size(0),
            -1
        )
        group_rtgs = group_rtgs.view(
            group_rtgs.size(0),
            -1
        )
        group_masks = group_masks.view(
            group_masks.size(0),
            -1
        )   


        group_size = len(group_observations)

        # Perform multiple update iterations
        for _ in range(self.updates_per_iter):
            J = 0 # Initialize loss for the current update iteration
            for i in range(group_size):

                observations = group_observations[i][group_masks[i].bool()]
                rtgs = group_rtgs[i][group_masks[i].bool()]
                actions = group_actions[i][group_masks[i].bool()]

                # Compute advantage by normalizing rewards
                A_i = (rtgs - torch.mean(rtgs)) / torch.std(rtgs + 1e-8)

                # Old log probabilities from stored group log probs
                with torch.no_grad():
                    old_log_probs, _ = self.old_policy.log_prob(observations, actions)
                
                # Current log probabilities from stored group log probs
                log_probs, entropy = self.policy.log_prob(observations, actions)
                
                # Compute probability ratios for policy update
                ratios = torch.exp(log_probs - old_log_probs)
                ratios = ratios # Apply mask to ratios

                # If reference model is provided, compute the KL divergence penalty
                if self.ref_model is not None:
                    _, ref_log_probs = self.ref_model(group_observations[i])
                    # Compute a simple form of KL divergence penalty
                    D_kl = torch.exp(ref_log_probs - log_probs) - torch.log(torch.exp(ref_log_probs - log_probs)) - 1
                else:
                    D_kl = 0
                
                # Policy loss using a clipped surrogate objective minus a KL-divergence penalty
                J += torch.min(ratios*A_i, torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon)*A_i).sum() - self.beta*D_kl
            
            # Normalize loss over groups (note: division by negative group_size to perform gradient ascent)
            J /= group_size

            # Perform a gradient descent step (note: optimizer minimizes, so negative loss is used)
            self.optimizer.zero_grad()
            J.backward()
            self.optimizer.step()

        # Update policy
        self.old_policy.load_state_dict(self.policy.state_dict())

    def save(self, path: str) -> None:
        """
        Save the optimizer state to the specified path.
        """
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pth"))

    def load(self, path: str) -> None:
        """
        Load the optimizer state from the specified path.
        """
        self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pth")))

    def metadata(self):
        return {
            "algorithm": "GRPO",
            'epsilon': self.epsilon,
            'beta': self.beta,
            'updates_per_iter': self.updates_per_iter,
        }
    
                