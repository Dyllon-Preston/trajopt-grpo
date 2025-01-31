class RolloutWorker():

    def __init__(
            self, 
            worker_id: int, 
            env, 
            policy, 
            rollout_length=200
            ):
        
        self.worker_id = worker_id
        self.env = env
        self.policy = policy
        self.rollout_length = rollout_length
    
    def run_episode(self, 
                    max_steps: int = 500
                ):
        
        pass
