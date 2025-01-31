class RolloutWorker():

    def __init__(
            self, 
            worker_id: int, 
            env, 
            policy, 
            ):
        
        self.worker_id = worker_id
        self.env = env
        self.policy = policy
    
    def run_episode(self, 
                    max_steps: int = 500,
                    num_episodes: int = 5,
                ):
        
        self.env.reset()
        done = False
        
        for _ in range(num_episodes):
            
            while not done and self.env.steps < max_steps:
                action = self.policy(self.env.state)
                state, reward, done = self.env.step(action)

            self.env.restart() # Reset the environment to the initial state when reset was last called
            

        pass
