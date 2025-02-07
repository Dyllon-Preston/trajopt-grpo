import torch.multiprocessing as mp
import torch
import sys
import time
import shutil

# Define a worker process that runs episodes in the environment
# Must be defined outside of the RolloutManager class for pickling (multiprocessing)
def worker_process(worker, task_queue, result_queue, num_episodes_per_worker):
    while True:
        if not task_queue.empty():
            task = task_queue.get(timeout=1)  # Wait for a task to be assigned
            if task == "ROLLOUT":
                if worker.episodes_completed[worker.worker_id] == 0:
                    worker_results = worker.run_episodes(num_episodes=num_episodes_per_worker)
                    result_queue.put((worker, worker_results))
                else:
                    task_queue.put("ROLLOUT")  # Re-queue the task if worker already completed episodes
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
        
        for i in range(num_workers):
            self.episodes_completed[i] = 0
        
    def print_progress(self):
        """
        Prints a live-updating progress bar for each worker.
        Each progress bar is adaptive to the terminal width and uses colors
        to highlight the completed (green) and pending (gray) portions.
        """
        # Get current terminal width; fallback to 80 if detection fails.
        term_width = shutil.get_terminal_size(fallback=(80, 20)).columns

        # Prepare a header line
        header = "Worker Progress:\n"
        progress_lines = [header]

        # Estimate how many characters we can devote to the bar.
        # We'll allocate fixed space for worker labels and counts.
        # For example, a line looks like: "Worker 0: [bar]  10/100"
        # Reserve 30 characters for static text per line:
        static_space = 30
        # The remaining width is available for the progress bar.
        bar_length = max(term_width - static_space, 10)  # ensure a minimum length

        for i in range(self.num_workers):
            completed = self.episodes_completed[i]
            total = self.num_episodes_per_worker
            # Avoid division by zero
            progress_fraction = completed / total if total > 0 else 0

            # Calculate how many characters in the bar should be "filled"
            filled_length = int(round(bar_length * progress_fraction))

            # Create the filled and unfilled portions.
            # Use green (ANSI code 92) for filled and bright black/gray (ANSI code 90) for unfilled.
            filled_bar = "\033[92m" + "█" * filled_length + "\033[0m"
            empty_bar = "\033[90m" + "─" * (bar_length - filled_length) + "\033[0m"

            # Assemble the line for this worker.
            line = f"Worker {i}: [{filled_bar}{empty_bar}] {completed}/{total}"
            progress_lines.append(line)

        # Move the cursor to the top of the screen and clear from there.
        # (If you do not want to clear the whole screen, you might use alternative cursor management.)
        sys.stdout.write("\033[H\033[J")
        sys.stdout.write("\n".join(progress_lines))
        sys.stdout.flush()
    
    def rollout(self):

        for _ in range(self.num_workers):
            self.task_queue.put("ROLLOUT")

        group_observations = torch.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps, self.obs_dim))
        group_actions = torch.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps, self.act_dim))
        group_rewards = torch.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps))
        group_rtgs = torch.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps))
        group_lengths = torch.zeros((self.num_workers, self.num_episodes_per_worker))
        group_masks = torch.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps))

        while any(ep < self.num_episodes_per_worker for ep in self.episodes_completed):
            self.print_progress()
            time.sleep(0.1)
        
        self.print_progress()

        for _ in range(self.num_workers):
            worker, worker_results = self.result_queue.get()
            self.episodes_completed[worker.worker_id] = 0

            episodic_observations, episodic_actions, episodic_rewards, episodic_rtgs, episodic_lengths, episodic_masks = worker_results

            group_observations[worker.worker_id] = episodic_observations
            group_actions[worker.worker_id] = episodic_actions
            group_rewards[worker.worker_id] = episodic_rewards
            group_rtgs[worker.worker_id] = episodic_rtgs
            group_lengths[worker.worker_id] = episodic_lengths
            group_masks[worker.worker_id] = episodic_masks

        return group_observations, group_actions, group_rewards, group_rtgs, group_lengths, group_masks
        
    def shutdown(self):
        for _ in range(self.num_workers):
            self.task_queue.put("SHUTDOWN")
        
        for p in self.processes:
            p.join()