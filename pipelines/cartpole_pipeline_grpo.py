import torch
from environments import CartPole
from buffers import Rollout_Buffer
from policies import GaussianActor_NeuralNetwork
from algorithms import GRPO
from visualize import Dashboard
from rollout import RolloutWorker, RolloutManager
from pipelines import Pipeline
from typing import Callable, Optional
from publish import Publisher

def create_cartpole_env() -> CartPole:
    """
    Instantiate the CartPole environment.

    Returns:
        CartPole: A new CartPole environment instance.
    """
    return CartPole()

def create_cartpole_pipeline_grpo(
    test_name: str,
    checkpoint_name: str,
    env_fn: Optional[Callable[[], CartPole]] = None,
    policy: Optional[GaussianActor_NeuralNetwork] = None,
    algorithm: Optional[GRPO] = None,
    rollout_manager: Optional[RolloutManager] = None,
    buffer: Optional[Rollout_Buffer] = None,
    visualizer: Optional[Dashboard] = None,
    publisher: Optional[Publisher] = None,
    logger: Optional[object] = None,
    load_path: Optional[str] = None,
) -> Pipeline:
    """
    Constructs a pipeline for executing a CartPole experiment.

    Args:
        test_name (str): Name of the test/experiment.
        checkpoint_name (str): Identifier for the checkpoint.
        env_fn (Optional[Callable]): Function to create the environment.
        policy (Optional): The policy network instance.
        algorithm (Optional): The training algorithm instance (e.g., GRPO).
        rollout_manager (Optional): Manager for collecting rollouts.
        buffer (Optional): Rollout buffer for experience storage.
        visualizer (Optional): Dashboard for visualizing training progress.
        publisher (Optional): Optional publisher for notifications.
        logger (Optional): Optional logger for detailed logging.
        load_path (Optional[str]): Path to load a pretrained model if any.

    Returns:
        Pipeline: A configured pipeline ready for the experiment.
    """
    env_fn = env_fn or create_cartpole_env
    policy = policy or GaussianActor_NeuralNetwork(
        input_dim=5,
        output_dim=1,
        hidden_dims=(128, 128, 128, 128),
        cov=0.5
    )
    algorithm = algorithm or GRPO(
        epsilon=0.15,
        beta=0.5,
        gamma=0.5,
        policy=policy,
        optimizer=torch.optim.Adam(policy.parameters(), lr=3e-4),
        ref_model=None,
        updates_per_iter=1
    )
    rollout_manager = rollout_manager or RolloutManager(
        env_fn=env_fn,
        worker_class=RolloutWorker,
        policy=policy,
        num_workers=10,
        num_episodes_per_worker=10,
        restart = False
    )
    buffer = buffer or Rollout_Buffer(
        rollout_manager=rollout_manager,
    )
    visualizer = visualizer or Dashboard(
        env = env_fn(),
        buffer=buffer,
        max_episodes_per_render=5,
    )

    publisher = publisher or Publisher(
        buffer=buffer,
        visualizer=visualizer,
        author=None,
        frame_skip=3
    )    

    return Pipeline(
        test_name=test_name,
        checkpoint_name=checkpoint_name,
        env_fn=env_fn,
        policy=policy,
        algorithm=algorithm,
        rollout_manager=rollout_manager,
        buffer=buffer,
        visualizer=visualizer,
        publisher=publisher,
        logger=logger,
        load_path=load_path,
    )
