import numpy as np
import torch

class RolloutWorker:
    def __init__(self, worker_id: int, env, policy, episodes_completed):
        """
        Initializes the rollout worker.

        Args:
            worker_id (int): Unique identifier for the worker.
            env: The environment instance.
            policy: A function or model that outputs actions given a state.
        """
        self.worker_id = worker_id
        self.env = env
        self.policy = policy
        self.episodes_completed = episodes_completed

    def run_episodes(self, num_episodes: int = 5, restart: bool = True):
        """
        Runs multiple episodes in the environment and collects data.

        Args:
            num_episodes (int): Number of episodes to run.
            restart (bool): Whether to restart (return to initial state) after each episode or reset (randomly initialize) the environment.

        Returns:
            tuple: (observations, actions, rewards, episode_lengths)
        """

        observation, info = self.env.reset()  # Reset environment to create an initial state

        max_steps = self.env.max_steps
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        batch_observations = np.zeros((num_episodes, max_steps, obs_dim))
        batch_actions = np.zeros((num_episodes, max_steps, act_dim))
        batch_rewards = np.zeros((num_episodes, max_steps))
        batch_rtgs = np.zeros((num_episodes, max_steps))
        batch_lengths = np.zeros(num_episodes, dtype=int)

        for episode in range(num_episodes):
            observations = np.zeros((max_steps, obs_dim))
            actions = np.zeros((max_steps, act_dim))
            rewards = np.zeros(max_steps)

            done = False
            step_idx = 0  # Explicit step counter

            while not done and step_idx < max_steps:
                action, _, _ = self.policy(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)

                observations[step_idx] = observation
                actions[step_idx] = action
                rewards[step_idx] = reward

                done = terminated or truncated
                step_idx += 1  # Increment step counter

            batch_observations[episode, :step_idx] = observations[:step_idx]
            batch_actions[episode, :step_idx] = actions[:step_idx]
            batch_rewards[episode, :step_idx] = rewards[:step_idx]
            batch_rtgs[episode, :step_idx] = self.calculate_rtg(rewards[:step_idx])
            batch_lengths[episode] = step_idx  # Store episode length

            if restart:
                observation, info = self.env.restart()  # Return to initial state for the next episode
            else:
                observation, info = self.env.reset() # Randomly initialize the environment

            self.episodes_completed[self.worker_id] = episode + 1

        # Compress the data from multiple episodes into a single batch and convert to torch tensors
        batch_observations = self.compress_episodes(batch_observations, batch_lengths)
        batch_actions = self.compress_episodes(batch_actions, batch_lengths)
        batch_rewards = self.compress_episodes(batch_rewards, batch_lengths)
        batch_rtgs = self.compress_episodes(batch_rtgs, batch_lengths)

        # Convert to torch tensors
        batch_observations = torch.from_numpy(batch_observations).float()
        batch_actions = torch.from_numpy(batch_actions).float()
        batch_rewards = torch.from_numpy(batch_rewards).float()
        batch_rtgs = torch.from_numpy(batch_rtgs).float()
        
        return batch_observations, batch_actions, batch_rewards, batch_rtgs, batch_lengths
    
    def calculate_rtg(self, rewards, gamma = 0.99):
        """
        Calculates the reward-to-go (RTG) for each step in the episode.

        Args:
            rewards (np.ndarray): Array of rewards for each step in the episode.
            gamma (float): Discount factor.

        Returns:
            torch.Tensor: Array of RTG values for each step in the episode.
        """
        
        rtg = np.zeros_like(rewards)

        for i in range(len(rewards) - 1, -1, -1):
            if i == len(rewards) - 1:
                rtg[i] = rewards[i]
            else:
                rtg[i] = rewards[i] + gamma * rtg[i + 1]
        return rtg
        
    
    def compress_episodes(self, batch_data, batch_lengths):
        """
        Compresses the data from multiple episodes into a single batch.

        Args:
            batch (np.ndarray): Array of data from multiple episodes.
            batch_lengths (np.ndarray): Array of lengths for each episode.

        Returns:
            tuple: Compressed batch data.
        """

        total_length = np.sum(batch_lengths)

        # Determine the shape of a single episode (ignoring the first dimension amd replacing the second)
        _, _, *episode_shape = batch_data.shape
        if isinstance(batch_data, np.ndarray):
            compressed_data = np.zeros((total_length, *episode_shape), dtype=batch_data.dtype)
        else:
            compressed_data = torch.zeros((total_length, *episode_shape), dtype=batch_data.dtype)
            
        start_idx = 0
        for i, episode in enumerate(batch_data):
            end_ix = start_idx + batch_lengths[i]
            compressed_data[start_idx:end_ix] = episode[:batch_lengths[i]]
            start_idx = end_ix
            
        return compressed_data
        
