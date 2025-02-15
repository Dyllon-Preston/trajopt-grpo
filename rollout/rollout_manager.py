import torch.multiprocessing as mp
import torch
import sys
import time
import shutil
from .rollout_worker import RolloutWorker

def worker_process(worker, task_queue, result_queue, num_episodes_per_worker):
    while True:
        if not task_queue.empty():
            task = task_queue.get()
            if task == "ROLLOUT":
                if worker.episodes_completed[worker.worker_id] == 0:
                    worker_results = worker.run_episodes(num_episodes=num_episodes_per_worker)
                    result_queue.put((worker, worker_results))
                else:
                    task_queue.put("ROLLOUT")
            elif task == "SHUTDOWN":
                break

class RolloutManager:
    def __init__(self, 
                 env_fn: callable,
                 policy,
                 worker_class = RolloutWorker,
                 restart = False,
                 num_workers: int = 4,
                 num_episodes_per_worker: int = 5,
                 use_multiprocessing: bool = True):
        
        self.env_fn = env_fn
        self.worker_class = worker_class
        self.policy = policy
        self.restart = restart
        self.num_workers = num_workers
        self.num_episodes_per_worker = num_episodes_per_worker
        self.use_multiprocessing = use_multiprocessing  # Toggle multiprocessing

        self.env = env_fn()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.max_steps = self.env.max_steps

        if self.use_multiprocessing:
            # Multiprocessing setup
            self.task_queue = mp.Queue()
            self.result_queue = mp.Queue()
            manager = mp.Manager()
            self.episodes_completed = manager.list([0 for _ in range(num_workers)])

            self.processes = []
            for i in range(num_workers):
                env = env_fn()
                worker = worker_class(i, env, policy, self.episodes_completed)
                p = mp.Process(target=worker_process, args=(worker, self.task_queue, self.result_queue, num_episodes_per_worker))
                p.start()
                self.processes.append(p)
        else:
            # Single-threaded setup
            self.episodes_completed = [0 for _ in range(num_workers)]
            self.workers = [worker_class(i, env_fn(), policy, self.episodes_completed) for i in range(num_workers)]

    def print_progress(self):
        term_width = shutil.get_terminal_size(fallback=(80, 20)).columns
        header = "Worker Progress:\n"
        progress_lines = [header]
        static_space = 30
        bar_length = max(term_width - static_space, 10)

        for i in range(self.num_workers):
            completed = self.episodes_completed[i]
            total = self.num_episodes_per_worker
            progress_fraction = completed / total if total > 0 else 0
            filled_length = int(round(bar_length * progress_fraction))

            filled_bar = "\033[92m" + "█" * filled_length + "\033[0m"
            empty_bar = "\033[90m" + "─" * (bar_length - filled_length) + "\033[0m"
            line = f"Worker {i}: [{filled_bar}{empty_bar}] {completed}/{total}"
            progress_lines.append(line)

        sys.stdout.write("\033[H\033[J")
        sys.stdout.write("\n".join(progress_lines))
        sys.stdout.flush()
    
    def rollout(self):
        group_observations = torch.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps, self.obs_dim))
        group_actions = torch.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps, self.act_dim))
        group_rewards = torch.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps))
        group_lengths = torch.zeros((self.num_workers, self.num_episodes_per_worker))
        group_masks = torch.zeros((self.num_workers, self.num_episodes_per_worker, self.max_steps))

        if self.use_multiprocessing:
            for _ in range(self.num_workers):
                self.task_queue.put("ROLLOUT")

            while any(ep < self.num_episodes_per_worker for ep in self.episodes_completed):
                self.print_progress()
                time.sleep(0.1)

            self.print_progress()

            for _ in range(self.num_workers):
                worker, worker_results = self.result_queue.get()
                self.episodes_completed[worker.worker_id] = 0

                episodic_observations, episodic_actions, episodic_rewards, episodic_lengths, episodic_masks = worker_results

                group_observations[worker.worker_id] = episodic_observations
                group_actions[worker.worker_id] = episodic_actions
                group_rewards[worker.worker_id] = episodic_rewards
                group_lengths[worker.worker_id] = episodic_lengths
                group_masks[worker.worker_id] = episodic_masks
        else:
            for worker in self.workers:
                worker_results = worker.run_episodes(num_episodes=self.num_episodes_per_worker, restart=self.restart)

                episodic_observations, episodic_actions, episodic_rewards, episodic_lengths, episodic_masks = worker_results

                group_observations[worker.worker_id] = episodic_observations
                group_actions[worker.worker_id] = episodic_actions
                group_rewards[worker.worker_id] = episodic_rewards
                group_lengths[worker.worker_id] = episodic_lengths
                group_masks[worker.worker_id] = episodic_masks

        return group_observations, group_actions, group_rewards, group_lengths, group_masks

    def shutdown(self):
        if self.use_multiprocessing:
            for _ in range(self.num_workers):
                self.task_queue.put("SHUTDOWN")
            
            for p in self.processes:
                p.join()
