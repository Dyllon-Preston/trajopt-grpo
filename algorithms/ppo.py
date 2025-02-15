import copy
import os

import torch
from algorithms.algorithm import Algorithm


class PPO(Algorithm):
    """
    Proximal Policy Optimization (PPO) implementation.

    Reference:
        Schulman et al., "Proximal Policy Optimization Algorithms", https://arxiv.org/abs/1707.06347
    """

    def __init__(
        self,
        epsilon: float,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ref_model: torch.nn.Module,
        updates_per_iter: int,
        c1: float = 0.5,
        kl_coeff: float = 0.5,
        gamma: float = 0.99,
        lam: float = 0.95,
        entropy: float = 0.01,
        batch_size: int = 64,
        monte_carlo: bool = True,
    ):
        """
        Initialize the PPO algorithm.

        Args:
            epsilon (float): Clipping parameter for the PPO objective.
            policy (torch.nn.Module): The policy network.
            optimizer (torch.optim.Optimizer): Optimizer for policy parameters.
            ref_model (torch.nn.Module): Reference model (e.g., used for KL control).
            updates_per_iter (int): Number of epochs per iteration.
            c1 (float, optional): Coefficient for the critic (value) loss. Defaults to 0.5.
            kl_coeff (float, optional): Coefficient for the KL divergence penalty. Defaults to 0.5.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            lam (float, optional): Lambda for Generalized Advantage Estimation. Defaults to 0.95.
            entropy (float, optional): Coefficient for the entropy bonus. Defaults to 0.01.
            batch_size (int, optional): Mini-batch size for PPO updates. Defaults to 64.
            monte_carlo (bool, optional): Flag to compute returns using Monte Carlo. Defaults to True.
        """
        self.epsilon = epsilon
        self.c1 = c1
        self.policy = policy
        self.ref_model = ref_model
        self.updates_per_iter = updates_per_iter
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam
        self.entropy = entropy
        self.batch_size = batch_size
        self.kl_coeff = kl_coeff
        self.monte_carlo = monte_carlo

        # Create a deep copy of the policy for stable "old policy" reference.
        self.old_policy = copy.deepcopy(self.policy)

    def learn(self, buffer) -> None:
        """
        Perform PPO training using data from the provided rollout buffer.

        Args:
            buffer: Rollout buffer containing group observations, actions, rewards, and masks.
        """
        gamma = self.gamma
        lam = self.lam
        kl_coeff = self.kl_coeff
        c1 = self.c1

        # Extract group data from the buffer.
        group_observations = buffer.group_observations
        group_actions = buffer.group_actions
        group_rewards = buffer.group_rewards
        group_masks = buffer.group_masks

        num_groups, num_episodes, max_steps, _ = group_observations.shape

        # Flatten the group data.
        observations = group_observations.view(-1, group_observations.size(-1))
        actions = group_actions.view(-1, group_actions.size(-1))
        rewards = group_rewards.view(-1)
        mask = group_masks.view(-1)

        group_masks_float = group_masks.float()

        # Compute state values using the policy's value function.
        values = self.policy.value(observations)
        group_values = values.view(num_groups, num_episodes, max_steps)

        # Initialize tensors for advantages and return-to-go (RTG).
        group_advantages = torch.zeros_like(group_rewards)
        group_rtgs = torch.zeros_like(group_rewards)

        if self.monte_carlo:
            # Compute Monte Carlo returns.
            for i in reversed(range(max_steps)):
                if i < max_steps - 1:
                    group_rtgs[:, :, i] = (
                        group_rewards[:, :, i] * group_masks_float[:, :, i]
                        + gamma * group_rtgs[:, :, i + 1] * group_masks_float[:, :, i + 1]
                    )
                else:
                    group_rtgs[:, :, i] = group_rewards[:, :, i] * group_masks_float[:, :, i]

            group_advantages = group_rtgs - group_values
        else:
            # Compute advantages using Generalized Advantage Estimation (GAE).
            for i in reversed(range(max_steps)):
                if i < max_steps - 1:
                    next_value = group_values[:, :, i + 1] * group_masks_float[:, :, i + 1]
                    delta = group_rewards[:, :, i] + gamma * next_value - group_values[:, :, i]
                    group_advantages[:, :, i] = (
                        delta + gamma * lam * group_advantages[:, :, i + 1] * group_masks_float[:, :, i + 1]
                    )
                else:
                    delta = group_rewards[:, :, i] - group_values[:, :, i]
                    group_advantages[:, :, i] = delta
            group_rtgs = group_values + group_advantages

        # Flatten RTGs and advantages and detach them from the computation graph.
        rtgs = group_rtgs.view(-1).detach()
        advantages = group_advantages.view(-1).detach()

        # Filter valid steps using the mask.
        valid_indices = mask.bool()
        observations = observations[valid_indices]
        actions = actions[valid_indices]
        rtgs = rtgs[valid_indices]
        advantages = advantages[valid_indices]

        # Normalize advantages and returns.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-8)

        # Compute old log probabilities without tracking gradients.
        with torch.no_grad():
            old_log_probs, _ = self.policy.log_prob(observations, actions)

        data_size = observations.size(0)
        # Perform PPO updates using mini-batches.
        for epoch in range(self.updates_per_iter):
            permutation = torch.randperm(data_size)
            current_batch_size = self.batch_size if self.batch_size is not None else data_size

            for start in range(0, data_size, current_batch_size):
                batch_indices = permutation[start : start + current_batch_size]
                batch_observations = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_rtgs = rtgs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                batch_log_probs, entropy = self.policy.log_prob(batch_observations, batch_actions)
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)

                # Compute the surrogate loss.
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic (value) loss.
                batch_values = self.policy.value(batch_observations)
                critic_loss = torch.nn.MSELoss()(batch_values, batch_rtgs)

                # Entropy bonus (to encourage exploration).
                entropy_bonus = self.entropy * entropy.mean()

                # KL divergence penalty.
                kl_div = (torch.exp(batch_old_log_probs) * (batch_old_log_probs - batch_log_probs)).mean()
                kl_loss = kl_coeff * kl_div

                # Total loss combines actor, critic, entropy, and KL terms.
                total_loss = actor_loss + c1 * critic_loss - entropy_bonus + kl_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # Update the old policy with the latest policy parameters.
        self.old_policy.load_state_dict(self.policy.state_dict())

    def metadata(self) -> dict:
        """
        Retrieve PPO configuration metadata.

        Returns:
            dict: A dictionary containing algorithm parameters.
        """
        return {
            "algorithm": "PPO",
            "epsilon": self.epsilon,
            "c1": self.c1,
            "kl_coeff": self.kl_coeff,
            "gamma": self.gamma,
            "lam": self.lam,
            "entropy": self.entropy,
            "batch_size": self.batch_size,
            "updates_per_iter": self.updates_per_iter,
        }

    def save(self, path: str) -> None:
        """
        Save the optimizer state.

        Args:
            path (str): Directory where the state will be saved.
        """
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

    def load(self, path: str) -> None:
        """
        Load the optimizer state.

        Args:
            path (str): Directory from which to load the state.
        """
        self.optimizer.load_state_dict(
            torch.load(os.path.join(path, "optimizer.pt"), weights_only=True)
        )
