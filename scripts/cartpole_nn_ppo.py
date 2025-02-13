"""
Cartpole with GRPO and Neural Network Policy
"""
import torch
from environments import CartPole
from buffers import Rollout_Buffer
from policies import GaussianActorCritic_NeuralNetwork
from algorithms import PPO
from train import Trainer
from visualize import Dashboard

from rollout import RolloutWorker, RolloutManager

def env_fn():
    return CartPole()

policy = GaussianActorCritic_NeuralNetwork(
    input_dim=5,
    output_dim=1,
    hidden_dims=(128, 128, 128, 128),
    cov=0.3
)

worker_class = RolloutWorker

rollout_manager = RolloutManager(
    env_fn=env_fn,
    worker_class=worker_class,
    policy=policy,
    num_workers=10,
    num_episodes_per_worker=5,
)

buffer = Rollout_Buffer(
    rollout_manager = rollout_manager,
)



ref_model = None

optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

algo = PPO(
    epsilon=0.2,
    c1 = 0.5,
    kl_coeff=0.5,
    policy=policy,
    optimizer = optimizer,
    ref_model = ref_model,
    updates_per_iter = 16,
    gamma = 0.99,
    lam = 0.95,
    entropy = 0.01,
    batch_size = int(1e6),
)


db = Dashboard(
    test_name = "cartpole_nn_ppo",
    checkpoint_name = "002",
    env_name = "CartPole",
    algorithm_name = "PPO",
    policy_metadata = {"num_parameters": policy.metadata()['num_parameters']},
    render = True,
    max_episodes_per_render = 5
)

breakpoint()


trainer = Trainer(
    test_name = "cartpole_nn_ppo",
    checkpoint_name = "002",
    buffer = buffer,
    policy = policy,
    ref_model = ref_model,
    algorithm = algo,
    epochs = 1000,
    render = True,
    render_freq = 40,
    max_episodes_per_render = 5
)

trainer.run()

breakpoint()

Trainer.shutdown()
