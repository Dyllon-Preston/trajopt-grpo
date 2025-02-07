from algorithms.algorithm import Algorithm

import torch

"""
Proximal Policy Optimization Algorithms
https://arxiv.org/abs/1707.06347

"""

class PPO(Algorithm):
    
    def __init__(
        self,
        epsilon: float,
        c1: float,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ref_model: torch.nn.Module,
        updates_per_iter: int,
    ):
        """
        Initialize PPO with the specified parameters.
        """
        self.epsilon = epsilon
        self.c1 = c1
        self.policy = policy
        self.ref_model = ref_model
        self.updates_per_iter = updates_per_iter
        self.optimizer = optimizer

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

        c1 = self.c1

        observations = group_observations.view(
            -1,
            group_observations.size(-1)
        )
        actions = group_actions.view(
            -1,
            group_actions.size(-1)
        )
        rewards = group_rewards.view(
            -1
        )
        mask = group_masks.view(
            -1
        )

        observations = observations[mask.bool()]
        actions = actions[mask.bool()]
        rewards = rewards[mask.bool()]
        
        log_probs = self.policy.log_prob(observations, actions).detach()
        values = self.policy.value(observations)

        rewards  = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        A = rewards - values.detach()

        A = (A - A.mean()) / (A.std() + 1e-8)

        for _ in range(self.updates_per_iter):
            curr_log_probs = self.policy.log_prob(observations, actions)
            V = self.policy.value(observations)

            ratio = torch.exp(curr_log_probs - log_probs)
            ratio = ratio

            actor_loss = -torch.min(ratio*A, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)*A).mean()
            critic_loss = torch.nn.MSELoss()(V, rewards)

            total_loss = actor_loss + c1*critic_loss

            print(total_loss)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()