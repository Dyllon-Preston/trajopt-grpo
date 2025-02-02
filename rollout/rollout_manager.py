import multiprocessing as mp
import numpy as np
import sys
import time

# Define a worker process that runs episodes in the environment
# Must be defined outside of the RolloutManager class for pickling (multiprocessing)
def worker_process(worker, task_queue, result_queue, num_episodes_per_worker):
    while True:
        if not task_queue.empty():
            task = task_queue.get(timeout=1)  # Wait for a task to be assigned
            if task == "ROLLOUT":
                worker.episodes_completed[worker.worker_id] = 0 # Shouldn't be necessary, but just in case
                result = worker.run_episodes(num_episodes = num_episodes_per_worker)
                result_queue.put((worker, result))
            elif task == "SHUTDOWN":
                break  # Exit process cleanly

class RolloutManager:
    def __init__(self, 
                 env_fn: callable,
                 worker_class,
                 policy,
                 num_workers: int = 4,
                 num_episodes_per_worker: int = 5
                 ):
        
        self.env_fn = env_fn  # callable that returns a gym environment
        self.worker_class = worker_class
        self.policy = policy
        self.num_workers = num_workers
        self.num_episodes_per_worker = num_episodes_per_worker

        # Create task and result queues
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        # Use a Manager list to share progress between processes
        manager = mp.Manager()
        self.episodes_completed = manager.list([0 for _ in range(num_workers)])

        self.env = env_fn()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.max_steps = self.env.max_steps

        # Create worker processes
        self.processes = []
        for i in range(num_workers):
            env = env_fn()
            worker = worker_class(i, env, policy, self.episodes_completed)
            p = mp.Process(target=worker_process, args=(worker, self.task_queue, self.result_queue, num_episodes_per_worker))
            p.start()
            self.processes.append(p)
        
    def print_progress(self):
        """Prints a live-updating progress bar for each worker."""
        sys.stdout.write("\rWorker Progress: ")
        for i in range(self.num_workers):
            sys.stdout.write(f"[Worker {i}: {self.episodes_completed[i]}/{self.num_episodes_per_worker}] ")
        sys.stdout.flush()
    
    def rollout(self):

        for _ in range(self.num_workers):
            self.task_queue.put("ROLLOUT")

        group_observations = np.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps, self.obs_dim))
        group_actions = np.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps, self.act_dim))
        group_log_probs = np.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps))
        group_rewards = np.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps))
        group_lengths = np.zeros((self.num_workers, self.num_episodes_per_worker))

        while any(ep < self.num_episodes_per_worker for ep in self.episodes_completed):
            self.print_progress()
            time.sleep(0.1)
        
        self.print_progress()

        for _ in range(self.num_workers):
            worker, worker_result = self.result_queue.get()
            observations, actions, log_probs, rewards, lengths = worker_result

            group_observations[worker.worker_id] = observations
            group_actions[worker.worker_id] = actions
            group_log_probs[worker.worker_id] = log_probs
            group_rewards[worker.worker_id] = rewards
            group_lengths[worker.worker_id] = lengths

        return group_observations, group_actions, group_log_probs, group_rewards, group_lengths
        
    def shutdown(self):
        for _ in range(self.num_workers):
            self.task_queue.put("SHUTDOWN")
        
        for p in self.processes:
            p.join()