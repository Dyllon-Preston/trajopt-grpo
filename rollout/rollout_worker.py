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

    def run_episodes(self, num_episodes: int = 5, restart: bool = False):
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

        episodic_observations = np.zeros((num_episodes, max_steps, obs_dim))
        episodic_actions = np.zeros((num_episodes, max_steps, act_dim))
        episodic_rewards = np.zeros((num_episodes, max_steps))
        episodic_lengths = np.zeros(num_episodes, dtype=int)
        episodic_masks = np.zeros((num_episodes, max_steps))

        for episode in range(num_episodes):
            observations = np.zeros((max_steps, obs_dim))
            actions = np.zeros((max_steps, act_dim))
            rewards = np.zeros(max_steps)

            done = False
            step_idx = 0  # Explicit step counter

            while not done and step_idx < max_steps:

                observations[step_idx] = observation # Ensure that the observation is stored before the action is taken

                action, _, _ = self.policy(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)

                actions[step_idx] = action
                rewards[step_idx] = reward

                done = terminated or truncated
                step_idx += 1  # Increment step counter

            episodic_observations[episode, :step_idx] = observations[:step_idx]
            episodic_actions[episode, :step_idx] = actions[:step_idx]
            episodic_rewards[episode, :step_idx] = rewards[:step_idx]
            episodic_lengths[episode] = step_idx  # Store episode length
            episodic_masks[episode, :step_idx] = 1  # Mark valid steps

            if restart:
                observation, info = self.env.restart()  # Return to initial state for the next episode
            else:
                observation, info = self.env.reset() # Randomly initialize the environment

            self.episodes_completed[self.worker_id] = episode + 1



        # Convert to torch tensors
        episodic_observations = torch.from_numpy(episodic_observations).float()
        episodic_actions = torch.from_numpy(episodic_actions).float()
        episodic_rewards = torch.from_numpy(episodic_rewards).float()
        episodic_lengths = torch.from_numpy(episodic_lengths).int()
        episodic_masks = torch.from_numpy(episodic_masks).float()
        
        return episodic_observations, episodic_actions, episodic_rewards, episodic_lengths, episodic_masks