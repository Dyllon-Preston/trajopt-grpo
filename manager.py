import multiprocessing as mp


class RolloutManger:
    def __init__(self, 
                 env_fn: callable,
                 worker_class,
                 policy,
                 num_workers: int=4,
                 max_steps: int=500
                 ):
        
        self.env_fn = env_fn # callable that returns a gym environment
        self.worker_class = worker_class
        self.policy = policy
        self.num_workers = num_workers
        self.max_steps = max_steps

    
        # Create a list of worker ids
        self.worker_ids = list(range(num_workers))
        # Create a list of worker objects with unique ids and environments
        self.workers = []
        for worker_id in self.worker_ids:
            self.workers.append(
                worker_class(worker_id, env_fn(), policy)
            )
        
        # Create a persistent pool of worker processes
        self.pool = mp.Pool(num_workers)

    def _worker_process(self, 
                        worker
                        ):
        return worker.run_epsisode(self.max_steps)
    
    def rollout(self):
        rollouts = self.pool.map(self._worker_process, self.workers)
        return rollouts

    def shutdown(self):
        self.pool.close()
        self.pool.join()
        

