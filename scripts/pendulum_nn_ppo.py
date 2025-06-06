"""
Cartpole with GRPO and Neural Network Policy
"""
import torch
from environments import Pendulum
from buffers import Rollout_Buffer
from policies import GaussianActorCritic_NeuralNetwork
from algorithms import PPO, PPO_Simple
from train import Trainer, load_checkpoint

from rollout import RolloutWorker
from rollout import RolloutManager

def env_fn():
    return Pendulum(swingup=True)

policy = GaussianActorCritic_NeuralNetwork(
    input_dim=3,
    output_dim=1,
    hidden_dims=(512, 512, 512),
    cov=0.2
)

worker_class = RolloutWorker

rollout_manager = RolloutManager(
    env_fn=env_fn,
    worker_class=worker_class,
    policy=policy,
    num_workers=10,
    num_episodes_per_worker=10,
    use_multiprocessing=True,
    restart = False,
)

buffer = Rollout_Buffer(
    rollout_manager = rollout_manager,
)



ref_model = None

optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

algo = PPO(
    epsilon=0.2,
    c1 = 0.01,
    policy=policy,
    optimizer = optimizer,
    ref_model = ref_model,
    updates_per_iter = 10,
    gamma = 0.99,
    lam = 0.95,
    entropy = 0.01,
    batch_size = 1024,
)

trainer = Trainer(
    test_name = "pendulum_swingup_nn_ppo",
    checkpoint_name = "001",
    buffer = buffer,
    policy = policy,
    ref_model = ref_model,
    algorithm = algo,
    epochs = 1000,
    render = False,
    render_freq = 20,
    max_episodes_per_render = 5
)


# load_checkpoint(
#     policy = policy,
#     buffer = buffer,
#     optimizer = optimizer,
#     trainer = trainer,
#     path = 'archive/Pendulum/pendulum_swingup_nn_ppo/001'

# )


trainer.run()

breakpoint()

Trainer.shutdown()
