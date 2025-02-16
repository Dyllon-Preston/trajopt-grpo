import os
import json
from typing import Any, Dict, Optional
import json

class Publisher:
    """
    Handles the finalization of a reinforcement learning project by saving model checkpoints,
    generating optimized GIF visualizations of simulation plots, saving metadata, and creating a detailed
    Markdown report.
    """

    def __init__(
        self,
        buffer: Any,
        visualizer: Any,
        max_episodes_per_render: int = 5,
        author: Optional[str] = None,
    ) -> None:
        """
        Initialize the Publisher.

        Args:
            buffer (Any): Buffer containing rollout data and simulation information.
            visualizer (Any): Visualizer component that produces simulation frames.
            max_episodes_per_render (int): Maximum number of episodes to render in simulation plots.
            author (Optional[str]): Name of the report author.
        """
        self.buffer = buffer
        self.visualizer = visualizer
        self.max_episodes_per_render = max_episodes_per_render
        self.author = author

        # Initialize environment via the buffer's rollout manager
        self.env = self.buffer.rollout_manager.env_fn()
        self.env_name = getattr(self.env, "env_name", "Unknown Environment")

    def create_gif(self, path: str, skip = 1, fps: Optional[int] = None) -> None:
        """
        Create an optimized GIF of the simulation plots that loops indefinitely.

        Args:
            path (str): The file path where the GIF will be saved.
            skip (FIX ME)
            fps (Optional[int]): Frames per second for the GIF. If None, defaults to the inverse of the environment's timestep.
        """
        frames = self.visualizer.frames()

        if fps is None:
            timestep = getattr(self.env, "timestep", None)
            fps = 1 / timestep if timestep else 1

        # Save the GIF with optimizations: duration is in milliseconds per frame.
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,  # Loop indefinitely.
            optimize=True,
        )

    def publish(self, path: str) -> None:
        """
        Publish the final report by creating a GIF visualization and a Markdown report.

        Args:
            path (str): Directory path where the final report will be published.
        """
        os.makedirs(path, exist_ok=True)
        gif_path = os.path.join(path, "simulation.gif")

        # Create the simulation GIF.
        self.create_gif(path=gif_path)

    def metadata(self) -> Dict[str, Any]:
        """
        Generate metadata for the final report.

        Returns:
            Dict[str, Any]: A dictionary containing metadata details.
        """
        return {
            "max_episodes_per_render": self.max_episodes_per_render,
            "author": self.author,
            "env_name": self.env_name,
        }

    def report(self, report_dir: str, metadata: dict) -> None:
        """
        Generate a detailed Markdown report summarizing the reinforcement learning project.

        The report includes:
            - Project Overview
            - Simulation GIF
            - Environment Details
            - Model (Policy) Configuration
            - Algorithm Parameters
            - Performance Metrics (Buffer)
            - Visualization & Publishing Details
            - Logger Information (if any)
            - The complete metadata dump in JSON format

        Args:
            report_dir (str): Directory where the Markdown report will be saved.
            metadata (dict): The complete metadata dictionary containing project details.
        """

        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "report.md")

        # Prettify the metadata as a JSON string.
        metadata_json = json.dumps(metadata, indent=4)

        # Extract nested metadata for ease of use.
        policy = metadata.get("policy", {})
        algorithm = metadata.get("algorithm", {})
        buffer_meta = metadata.get("buffer", {})
        visualizer = metadata.get("visualizer", {})
        publisher = metadata.get("publisher", {})
        logger_meta = metadata.get("logger", {})

        markdown_content = \
f"""# Project Report: {metadata.get("test_name", "Unnamed Project")}

**Checkpoint:** {metadata.get("checkpoint_name", "N/A")}  
**Creation Date:** {metadata.get("creation_date", "N/A")}  
**Environment:** {metadata.get("env_name", "N/A")}

---

## Overview

This report provides a comprehensive overview of the reinforcement learning project. The details below summarize the model configuration, algorithm parameters, performance metrics, and visualization settings.

---

## Simulation

![Simulation GIF](simulation.gif)

## Model Details

### Policy Configuration
- **Input Dimension:** {policy.get("input_dim", "N/A")}
- **Output Dimension:** {policy.get("output_dim", "N/A")}
- **Hidden Layers:** {", ".join(map(str, policy.get("hidden_dims", [])))}
- **Activation Function:** {policy.get("activation", "N/A")}
- **Covariance:** {policy.get("cov", "N/A")}
- **Number of Parameters:** {policy.get("num_parameters", "N/A")}

---

## Algorithm Configuration

- **Algorithm:** {algorithm.get("algorithm", "N/A")}
- **Epsilon:** {algorithm.get("epsilon", "N/A")}
- **C1 (Value Loss Coefficient):** {algorithm.get("c1", "N/A")}
- **KL Coefficient:** {algorithm.get("kl_coeff", "N/A")}
- **Gamma (Discount Factor):** {algorithm.get("gamma", "N/A")}
- **Lambda (GAE):** {algorithm.get("lam", "N/A")}
- **Entropy Coefficient:** {algorithm.get("entropy", "N/A")}
- **Batch Size:** {algorithm.get("batch_size", "N/A")}
- **Updates per Iteration:** {algorithm.get("updates_per_iter", "N/A")}

---

## Performance Metrics

### Buffer
- **Average Reward:** {buffer_meta.get("avg_reward", "N/A")}

---

## Visualization & Publishing

### Visualizer
- **Max Episodes per Render:** {visualizer.get("max_episodes_per_render", "N/A")}

### Publisher
- **Max Episodes per Render:** {publisher.get("max_episodes_per_render", "N/A")}
- **Author:** {publisher.get("author", "N/A")}
- **Publisher Creation Date:** {publisher.get("creation_date", "N/A")}
- **Environment:** {publisher.get("env_name", "N/A")}

---

## Logger
{ "- No logger details provided." if not logger_meta else json.dumps(logger_meta, indent=4) }

---

## Complete Metadata

```json
{metadata_json}
```

This report was automatically generated by the Publisher class. """
        
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write(markdown_content)
