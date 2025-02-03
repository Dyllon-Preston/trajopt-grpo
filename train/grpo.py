import torch
import copy

"""
DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
https://arxiv.org/abs/2402.03300
"""

class GRPO:
    """
    Group Regularized Policy Optimization (GRPO) implementation.

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

        # Create a deep copy of the policy for stable old_policy
        self.old_policy = copy.deepcopy(self.policy)

    def train(self, group_observations, group_actions, group_log_probs, group_rewards):
        """
        Perform training using group data.

        Args:
            group_observations (list of torch.Tensor): Observations for each group.
            group_actions (list of torch.Tensor): Actions taken for each group.
            group_log_probs (list of torch.Tensor): Log probabilities from the policy.
            group_rewards (list of torch.Tensor): Rewards received for each group.
        """
        # Perform multiple update iterations
        for _ in range(self.updates_per_iter):
            J = 0 # Initialize loss for the current update iteration
            for i in range(self.group_size):
                # Compute advantage by normalizing rewards
                A_i = (group_rewards[i] - torch.mean(group_rewards[i])) / torch.std(group_rewards[i] + 1e-8)

                # Forward pass on old_policy to get old log probabilities
                _, old_log_probs = self.old_policy(group_observations[i])
                
                # Current log probabilities from stored group log probs
                log_probs = group_log_probs[i]

                # Compute probability ratios for policy update
                ratios = torch.exp(log_probs - old_log_probs)

                # If reference model is provided, compute the KL divergence penalty
                if self.ref_model is not None:
                    _, ref_log_probs = self.ref_model(group_observations[i])
                    # Compute a simple form of KL divergence penalty
                    D_kl = torch.exp(ref_log_probs - log_probs) - torch.log(torch.exp(ref_log_probs - log_probs)) - 1
                else:
                    D_kl = 0
                
                # Policy loss using a clipped surrogate objective minus a KL-divergence penalty
                J += min(ratios*A_i, torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon)*A_i) - self.beta*D_kl
            
            # Normalize loss over groups (note: division by negative group_size to perform gradient ascent)
            J /= -self.group_size

            # Perform a gradient descent step (note: optimizer minimizes, so negative loss is used)
            self.optimizer.zero_grad()
            J.backward()
            self.optimizer.step()

            # Update policy
            self.old_policy.load_state_dict(self.policy.state_dict())
                