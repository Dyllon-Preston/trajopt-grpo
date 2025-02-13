from algorithms.algorithm import Algorithm

import torch
import copy

"""
Proximal Policy Optimization Algorithms
https://arxiv.org/abs/1707.06347

"""

class PPO(Algorithm):
    
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
        monte_carlo: bool = True
    ):
        """
        Initialize PPO with the specified parameters.

        Args:
            epsilon (float): Clipping parameter for PPO.
            c1 (float): Weight for the critic (value) loss.
            kl_coeff (float): Coefficient for the KL divergence penalty.
            policy (torch.nn.Module): The policy network.
            optimizer (torch.optim.Optimizer): Optimizer for policy parameters.
            ref_model (torch.nn.Module): A reference model (unused here, but can be used for e.g. KL control).
            updates_per_iter (int): Number of epochs per iteration.
            gamma (float): Discount factor.
            lam (float): Lambda for Generalized Advantage Estimation.
            entropy_coef (float): Coefficient for the entropy bonus.
            batch_size (int): Mini-batch size for PPO updates.
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
            group_observations (torch.Tensor): Tensor of shape (num_episodes, max_steps, obs_dim).
            group_actions (torch.Tensor): Tensor of shape (num_episodes, max_steps, action_dim).
            group_rewards (torch.Tensor): Tensor of shape (num_episodes, max_steps).
            group_masks (torch.Tensor): Tensor of shape (num_episodes, max_steps); 1.0 for valid steps, 0.0 for padded/terminal.
        """

        c1 = self.c1
        gamma = self.gamma
        lam = self.lam
        kl_coeff = self.kl_coeff

        num_groups, num_episodes, max_steps, obs_dim = group_observations.shape

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

        group_masks_float = group_masks.float()

        values = self.policy.value(observations)
        group_values = values.view(num_groups, num_episodes, max_steps)

        group_advantages = torch.zeros_like(group_rewards)
        group_rtgs = torch.zeros_like(group_rewards)

        


        if self.monte_carlo:

            for i in reversed(range(max_steps)):
                if i < max_steps - 1:
                    group_rtgs[:,:,i] = group_rewards[:,:,i]*group_masks_float[:,:,i] + gamma*group_rtgs[:,:,i+1]*group_masks_float[:,:,i+1]
                else:
                    group_rtgs[:,:,i] = group_rewards[:,:,i]*group_masks_float[:,:,i]

            group_advantages = group_rtgs - group_values
        
        else:


            for i in reversed(range(max_steps)):
                if i < max_steps - 1:
                    next_value = group_values[:,:,i+1]*group_masks_float[:,:,i+1]
                    delta = group_rewards[:,:,i] + gamma*next_value - group_values[:,:,i]
                    group_advantages[:,:,i] = delta + gamma*lam*group_advantages[:,:,i+1]*group_masks_float[:,:,i+1]
                else:
                    next_value = 0
                    delta = group_rewards[:,:,i] + gamma*next_value - group_values[:,:,i]
                    group_advantages[:,:,i] = delta
                
                group_rtgs = group_values + group_advantages


        rtgs = group_rtgs.view(-1).detach() 
        advantages = group_advantages.view(-1).detach()

        
        observations = observations[mask.bool()]
        actions = actions[mask.bool()]
        rtgs = rtgs[mask.bool()]
        advantages = advantages[mask.bool()]

        # Normalization of advantages and rtgs
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-8)
        
        # Compute old log probabilities
        with torch.no_grad():
            old_log_probs, _ = self.policy.log_prob(observations, actions)

        # --- PPO Updates using mini-batches ---
        data_size = observations.size(0)
        for epoch in range(self.updates_per_iter):
            # Shuffle data
            perm = torch.randperm(data_size)
            for i in range(0, observations.size(0), self.batch_size):
                batch_indices = perm[i:i + self.batch_size]
                batch_observations = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_rtgs = rtgs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                batch_log_probs, entropy = self.policy.log_prob(batch_observations, batch_actions)
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)

                # PPO surrogate loss
                surr1 = ratio*batch_advantages
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)*batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                batch_values = self.policy.value(batch_observations)
                critic_loss = torch.nn.MSELoss()(batch_values, batch_rtgs)

                # Entropy bonus
                entropy =  entropy.mean()
                entropy_loss = -self.entropy*entropy

                # KL divergence penalty
                kl_div = (torch.exp(batch_old_log_probs)*(batch_old_log_probs - batch_log_probs)).mean()
                kl_loss = kl_coeff*kl_div

                # if kl_div > 0.01*self.epsilon:
                #     print("Early stopping due to reaching max KL divergence")
                #     break

                print("Actor Loss: ", actor_loss, "Critic Loss: ", critic_loss, "Entropy Loss: ", entropy_loss)

                # Total loss
                total_loss = actor_loss + c1*critic_loss + entropy_loss + kl_loss  

                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()  
                self.optimizer.step()


                # total_norm = 0.0
                # for p in self.policy.parameters():
                #     if p.grad is not None:
                #         param_norm = p.grad.data.norm(2)
                #         total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** 0.5
                # print("Gradient norm:", total_norm)

        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())


                        




 