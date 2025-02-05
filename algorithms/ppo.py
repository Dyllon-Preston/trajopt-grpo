from algorithms.algorithm import Algorithm

import torch

"""
Proximal Policy Optimization Algorithms
https://arxiv.org/abs/1707.06347

"""

class PPO(Algorithm):
    
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
        Initialize PPO with the specified parameters.
        """
        self.group_size = group_size
        self.epsilon = epsilon
        self.policy = policy
        self.ref_model = ref_model
        self.beta = beta
        self.updates_per_iter = updates_per_iter
        self.optimizer = optimizer

    def train(self):

        observations = buffer.observations
        actions = buffer.actions
        log_probs = buffer.log_probs
        rewards = buffer.rewards
        values = buffer.values

        rewards  = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        A = rewards - values.detach()

        A = (A - A.mean()) / (A.std() + 1e-8)

        for _ in range(self.updates_per_iter):
            _, curr_log_probs, V = self.policy(observations)

            ratio = torch.exp(curr_log_probs - log_probs)

            actor_loss = -min(ratio*A, torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)*A).mean()
            critic_loss = torch.nn.MSELoss()(V, rewards)

            total_loss = actor_loss + c1*critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


        


        

        pass