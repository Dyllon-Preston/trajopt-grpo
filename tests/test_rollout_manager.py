from rollout import RolloutManager
from rollout import RolloutWorker
from environments import CartPole

# Define the environment creation function outside of the test for pickling
def env_fn():
    return CartPole(max_steps=10)

env = env_fn()

# Define a random policy for testing outside of the test for pickling
def random_policy(state):
    return (env.action_space.sample(), 0)

# Define the worker class outside of the test for pickling
def worker_class(worker_id, env, policy, episodes_completed):
    return RolloutWorker(worker_id, env, policy, episodes_completed)

def test_rollout_manager():
    """Tests the RolloutManager class."""

    # Initialize RolloutManager with the random policy
    manager = RolloutManager(
        env_fn=env_fn,  # Pass the class method for creating the env
        worker_class=worker_class,
        policy=random_policy,  # Use the class method for policy
        num_workers=2,  # Using 2 workers for testing
        num_episodes_per_worker=3,  # Each worker will run 3 episodes
    )

    # Run rollout process
    group_observations, group_actions, group_log_probs, group_rewards, group_lengths = manager.rollout()

    # Extract the number of workers, episodes, and max steps
    num_workers = manager.num_workers
    num_episodes = manager.num_episodes_per_worker
    max_steps = env.max_steps

    # Ensure the dimensions of the results are as expected
    assert group_observations.shape == (num_workers, num_episodes, max_steps, env.observation_space.shape[0]), \
        f"Unexpected observations shape: {group_observations.shape}"

    assert group_actions.shape == (num_workers, num_episodes, max_steps, env.action_space.shape[0]), \
        f"Unexpected actions shape: {group_actions.shape}"
    
    assert group_log_probs.shape == (num_workers, num_episodes, max_steps), \
        f"Unexpected log_probs shape: {group_log_probs.shape}"

    assert group_rewards.shape == (num_workers, num_episodes, max_steps), \
        f"Unexpected rewards shape: {group_rewards.shape}"

    assert group_lengths.shape == (num_workers, num_episodes), \
        f"Unexpected episode lengths shape: {group_lengths.shape}"

    # Ensure episodes_completed is updated correctly across workers
    assert list(manager.episodes_completed) == [3, 3], \
        f"Unexpected episodes_completed: {manager.episodes_completed}"

    # Shutdown the manager
    manager.shutdown()

