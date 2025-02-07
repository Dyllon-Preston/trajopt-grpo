"""
Cartpole with GRPO and Neural Network Policy
"""
import torch
from environments import CartPole
from buffers import Rollout_Buffer
from policies import GaussianActorCritic_NeuralNetwork
from algorithms import PPO
from train import Trainer

from rollout import RolloutWorker
from rollout import RolloutManager

def env_fn():
    return CartPole()

policy = GaussianActorCritic_NeuralNetwork(
    input_dim=5,
    output_dim=1,
    hidden_dims=(256, 256, 256),
    cov=0.3
)

worker_class = RolloutWorker

rollout_manager = RolloutManager(
    env_fn=env_fn,
    worker_class=worker_class,
    policy=policy,
    num_workers=10,
    num_episodes_per_worker=15,
)

buffer = Rollout_Buffer(
    rollout_manager = rollout_manager,
)



ref_model = None

optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

algo = PPO(
    epsilon=0.2,
    c1 = 0.1,
    policy=policy,
    optimizer = optimizer,
    ref_model = ref_model,
    updates_per_iter = 10
    
)

trainer = Trainer(
    test_name = "cartpole_nn_ppo",
    checkpoint_name = "001",
    buffer = buffer,
    policy = policy,
    ref_model = ref_model,
    algorithm = algo,
    epochs = 1000,
    render = True,
    render_freq = 20,
    max_episodes_per_render = 1
)

trainer.run()

breakpoint()

Trainer.shutdown()
