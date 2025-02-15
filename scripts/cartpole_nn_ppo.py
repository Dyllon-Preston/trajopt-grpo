"""
Main entry point for running the Cartpole simulation using PPO with a Neural Network Policy.
"""

if __name__ == "__main__":
    # Import the pipeline creation function for the Cartpole simulation.
    from pipelines import create_cartpole_pipeline

    # Create the cartpole pipeline with specified test parameters.
    pipeline = create_cartpole_pipeline(
        test_name="cartpole_nn_ppo",      # Identifier for this test instance.
        checkpoint_name="001",           # Checkpoint identifier for saving/loading progress.
        # Optional: Specify a load path to resume from a previous checkpoint.
        # load_path='archive/CartPole/cartpole_nn_ppo/001'
    )

    # Set metadata for publishing results.
    pipeline.publisher.author = "Dyllon Preston"

    # Configure visualization: skip frames for intermediate visualization steps.
    pipeline.visualizer.skip = 10

    # Optionally disable rendering by uncommenting the line below.
    # pipeline.render = False