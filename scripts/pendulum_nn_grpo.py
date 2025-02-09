"""
Cartpole with GRPO and Neural Network Policy
"""
import torch
from environments import Pendulum
from buffers import Rollout_Buffer
from policies import GaussianActor_NeuralNetwork
from algorithms import GRPO
from train import Trainer, load_checkpoint

from rollout import RolloutWorker
from rollout import RolloutManager

def env_fn():
    return Pendulum(env_name="Pendulum_Swingup")

policy = GaussianActor_NeuralNetwork(
    input_dim=3,
    output_dim=1,
    hidden_dims=(512, 512, 512),
    cov=0.5
)

worker_class = RolloutWorker

rollout_manager = RolloutManager(
    env_fn=env_fn,
    worker_class=worker_class,
    policy=policy,
    num_workers=10,
    num_episodes_per_worker=15,
    use_multiprocessing=True,
    restart=True
)

buffer = Rollout_Buffer(
    rollout_manager = rollout_manager,
)


ref_model = None

optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

algo = GRPO(
    epsilon=0.2,
    beta=0.01,
    policy=policy,
    optimizer = optimizer,
    ref_model = ref_model,
    updates_per_iter = 5
    
)

trainer = Trainer(
    test_name = "pendulum_nn_grpo",
    checkpoint_name = "001",
    buffer = buffer,
    policy = policy,
    ref_model = ref_model,
    algorithm = algo,
    epochs = 1000,
    render = False,
    render_freq = 20,
    max_episodes_per_render = 1,
    

)

trainer.run()

breakpoint()

Trainer.shutdown()
