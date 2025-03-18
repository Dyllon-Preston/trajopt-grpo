import os
import json
import datetime
from typing import Optional, Callable, Any, Dict


class Pipeline:
    """
    Orchestrates the training, testing, and publishing of a model in an end-to-end workflow.
    """

    def __init__(
        self,
        test_name: str,
        checkpoint_name: str,
        env_fn: Callable[[], Any],
        policy: Any,
        algorithm: Any,
        rollout_manager: Any,
        buffer: Any,
        visualizer: Optional[Any],
        publisher: Any,
        logger: Optional[Any] = None,
        load_path: Optional[str] = None,
        save_freq: int = 10,
        render_freq: int = 40,
    ) -> None:
        """
        Initialize the Pipeline.

        Args:
            test_name (str): Name of the test run.
            checkpoint_name (str): Identifier for the checkpoint.
            env_fn (Callable[[], Any]): Function to instantiate the environment.
            policy (Any): Policy component.
            algorithm (Any): Algorithm component.
            rollout_manager (Any): Rollout manager component.
            buffer (Any): Buffer component.
            visualizer (Optional[Any]): Visualizer component for rendering.
            publisher (Any): Publisher component.
            logger (Optional[Any]): Logger component. Defaults to None.
            load_path (Optional[str]): Path to load previous state. Defaults to None.
            save_freq (int): Frequency (in epochs) to save pipeline components. Defaults to 10.
            render_freq (int): Frequency (in epochs) to render visualization. Defaults to 40.
        """
        self.test_name = test_name
        self.checkpoint_name = checkpoint_name

        # Initialize environment
        self.env_fn = env_fn
        self.env = env_fn()
        self.env_name = self.env.env_name

        # Components of the pipeline
        self.policy = policy
        self.algorithm = algorithm
        self.rollout_manager = rollout_manager
        self.buffer = buffer
        self.visualizer = visualizer
        self.publisher = publisher
        self.logger = logger

        self.load_path = load_path
        self.save_freq = save_freq
        self.render_freq = render_freq

        self.today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load pipeline components if available
        if load_path is not None:
            self.load()

        self.initialize()

    def initialize(self) -> None:
        """
        Initialize pipeline components by loading metadata if available and initializing the visualizer.
        """

        # Set up directories for archiving and publishing
        self.archive_path = os.path.join(".", "archive", self.env_name, self.test_name, self.checkpoint_name)
        self.publish_path = os.path.join(".", "reports", self.env_name, self.test_name, self.checkpoint_name)
        os.makedirs(self.archive_path, exist_ok=True)

        if self.load_path is not None:
            metadata_path = os.path.join(self.load_path, "metadata.json")
            self.load_metadata(metadata_path)

        metadata = self.get_metadata()
        if self.visualizer is not None:
            self.visualizer.initialize(metadata)

    def load(self) -> None:
        """
        Load the state of the algorithm, policy, and buffer from the provided load path.
        """
        if self.load_path is not None:
            self.algorithm.load(self.load_path)
            self.policy.load(self.load_path)
            self.buffer.load(self.load_path)
            # if self.logger is not None:
            #     self.logger.load(self.load_path)

    def save(self, path: str) -> None:
        """
        Save the state of the algorithm, policy, and buffer along with the metadata.

        Args:
            path (str): Directory path to save the components.
        """
        self.algorithm.save(path)
        self.policy.save(path)
        self.buffer.save(path)

        metadata = self.get_metadata()
        metadata_file = os.path.join(path, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Gather metadata from the pipeline components.

        Returns:
            dict: A dictionary containing pipeline metadata.
        """
        return {
            "test_name": self.test_name,
            "checkpoint_name": self.checkpoint_name,
            "creation_date": self.today,
            "env_name": self.env_name,
            "policy": self.policy.metadata(),
            "algorithm": self.algorithm.metadata(),
            "buffer": self.buffer.metadata(),
            "visualizer": self.visualizer.metadata() if self.visualizer is not None else {},
            "publisher": self.publisher.metadata() if self.publisher is not None else {},
            "logger": self.logger.metadata() if self.logger is not None else {},
        }

    def load_metadata(self, path: str) -> Dict[str, Any]:
        """
        Load metadata from a JSON file.

        Args:
            path (str): Path to the metadata JSON file.

        Returns:
            dict: The loaded metadata.
        """
        with open(path, "r") as f:
            metadata = json.load(f)
        return metadata

    def train(self, epochs: int) -> None:
        """
        Train the model for a specified number of epochs.

        Args:
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            # Sample data and perform learning
            self.buffer.sample()
            self.algorithm.learn(self.buffer)

            # check if visualizer has plot method
            if hasattr(self.visualizer, "plot"):
                self.visualizer.plot()

            # Render visualization at specified frequency
            if self.visualizer is not None and epoch % self.render_freq == 0:
                self.visualizer.render()

            # Save pipeline components at specified frequency
            if epoch % self.save_freq == 0:
                self.save(self.archive_path)

    def test(self) -> None:
        """
        Test the model by sampling from the buffer.
        """
        self.buffer.sample()

    def publish(self) -> None:
        """
        Publish the model results by sampling from the buffer, publishing the results,
        and saving the current pipeline state.
        """
        os.makedirs(self.publish_path, exist_ok=True)
        self.buffer.sample()
        self.publisher.publish(self.publish_path)
        self.publisher.report(self.publish_path, self.get_metadata())
        self.save(self.publish_path)

    def save_trajectory(self) -> None:
        """
        Save the trajectory of the model by sampling from the buffer and saving the trajectory.
        """
        self.buffer.sample()
        self.buffer.save_trajectory(self.archive_path)


    def shutdown(self) -> None:
        """
        Shutdown the pipeline by closing any open resources.
        """
        self.rollout_manager.shutdown()

        if self.visualizer is not None:
            self.visualizer.close()
        if self.logger is not None:
            self.logger.close()
        print("\n\nPipeline shutdown complete.")
