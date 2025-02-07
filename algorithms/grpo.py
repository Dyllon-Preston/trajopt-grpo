from algorithms.algorithm import Algorithm

import torch
import copy

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
        self.updates_per_iter = updates_per_iter
        self.optimizer = optimizer

        # Create a deep copy of the policy for stable old_policy
        self.old_policy = copy.deepcopy(self.policy)

    def learn(
            self, 
            group_observations, 
            group_actions, 
            group_rewards, 
            group_masks):
        """
        Perform training using group data.

        Args:
            group_observations (list of torch.Tensor): Observations for each group.
            group_actions (list of torch.Tensor): Actions taken for each group.
            group_values (list of torch.Tensor): Value estimates for each group (Not used in this algorithm).
            group_rewards (list of torch.Tensor): Rewards received for each group.
        """

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
        group_masks = group_masks.view(
            group_masks.size(0),
            -1
        )   

        group_size = len(group_observations)

        group_old_log_probs = torch.zeros((group_size, group_observations[0].size(0)))
        for i in range(group_size):
            old_log_probs = self.old_policy.log_prob(group_observations[i], group_actions[i])
            group_old_log_probs[i] = old_log_probs.detach()

        # Perform multiple update iterations
        for _ in range(self.updates_per_iter):
            J = 0 # Initialize loss for the current update iteration
            for i in range(group_size):

                # Mask for valid data points
                mask = group_masks[i]

                # Compute advantage by normalizing rewards
                A_i = (group_rewards[i] - torch.mean(group_rewards[i])) / torch.std(group_rewards[i] + 1e-8)

                # Old log probabilities from stored group log probs
                old_log_probs = group_old_log_probs[i]
                
                # Current log probabilities from stored group log probs
                log_probs = self.policy.log_prob(group_observations[i], group_actions[i])
                
                # Compute probability ratios for policy update
                ratios = torch.exp(log_probs - old_log_probs)
                ratios = ratios*mask # Apply mask to ratios

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
            J /= -group_masks.sum()*group_size

            print(J)

            # Perform a gradient descent step (note: optimizer minimizes, so negative loss is used)
            self.optimizer.zero_grad()
            J.backward()
            self.optimizer.step()

        # Update policy
        self.old_policy.load_state_dict(self.policy.state_dict())
                