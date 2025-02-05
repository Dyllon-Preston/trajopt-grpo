"""
Cartpole with GRPO and Neural Network Policy
"""
import torch
from environments import CartPole
from buffers import Rollout_Buffer
from policies import GaussianActor_NeuralNetwork
from algorithms import GRPO
from train import Trainer

from rollout import RolloutWorker
from rollout import RolloutManager

def env_fn():
    return CartPole()

policy = GaussianActor_NeuralNetwork(
    input_dim=5,
    output_dim=1,
    hidden_dims=(64, 64),
)

worker_class = RolloutWorker

rollout_manager = RolloutManager(
    env_fn=env_fn,
    worker_class=worker_class,
    policy=policy,
    num_workers=1,
    num_episodes_per_worker=25,
)

buffer = Rollout_Buffer(
    rollout_manager = rollout_manager,
)



ref_model = None

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

algo = GRPO(
    epsilon=0.2,
    beta=0.01,
    policy=policy,
    optimizer = optimizer,
    ref_model = ref_model,
    
)

trainer = Trainer(
    test_name = "cartpole_nn_grpo",
    checkpoint_name = "001",
    buffer = buffer,
    policy = policy,
    ref_model = ref_model,
    algorithm = algo,
    epochs = 1
)

trainer.run()

breakpoint()

Trainer.shutdown()
