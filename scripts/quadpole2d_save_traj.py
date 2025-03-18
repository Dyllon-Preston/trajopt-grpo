"""

"""

if __name__ == "__main__":
    # Import the pipeline creation function for the QuadPole2D simulation.
    from pipelines import create_quadpole2d_pipeline_ppo

    # Create the cartpole pipeline with specified test parameters.
    pipeline = create_quadpole2d_pipeline_ppo(
        test_name="quadpole2d_nn_ppo",      # Identifier for this test instance.
        checkpoint_name="001",           # Checkpoint identifier for saving/loading progress.
        # Optional: Specify a load path to resume from a previous checkpoint.
        load_path='archive/QuadPole2D/quadpole2d_nn_ppo/001'
    )

    # Set metadata for publishing results.
    pipeline.publisher.author = "Dyllon Preston"
    pipeline.initialize() # Update pipelien to refresh author name

    # Save trajectory
    pipeline.save_trajectory()

    # Shutdown the pipeline.
    pipeline.shutdown()