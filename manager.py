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

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        self.processes = []
        for i in range(num_workers):
            env = env_fn()
            worker = worker_class(i, env, policy)
            p = mp.Process(target=self._worker_process, args=(worker,))
            p.start()
            self.processes.append(p)

    def _worker_process(self, 
                        worker
                        ):
        
        
        while True:
            task = self.task_queue.get()
            if task == "ROLLOUT":
                result = worker.run_episodes()
                self.result_queue.put((worker, result))
            
            elif task == "SHUTDOWN":
                break
    
    def rollout(self):
        
        for _ in range(self.num_workers):
            self.task_queue.put("ROLLOUT")

        rollouts = []
        for _ in range(self.num_workers):
            worker_id, worker_result = self.result_queue.get()
            rollouts.append(self.result_queue.get())
        

    def shutdown(self):
        self.pool.close()
        self.pool.join()
        

