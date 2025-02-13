import torch
from typing import Union

class NeuralNetwork(torch.nn.Module):
    """
    A fully-connected neural network with customizable hidden layers and activation function.
    """

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: list, 
        activation: Union[str, list] = 'ReLu', # Common valid activation functions ['ReLu', 'Sigmoid', 'Tanh']
        ):
        """
        Initializes the NeuralNetwork module.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_dims (list): List of integers specifying the size of each hidden layer.
            activation (Union[str, list], optional): Either a string specifying the activation function
                                                    (as defined in torch.nn) to be used for all hidden layers,
                                                    or a list of activation function names (one per hidden layer).
                                                    Defaults to 'ReLU'.
        """
        super(NeuralNetwork, self).__init__()

        # Store dimensions and layer configuration.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # When hidden layers exist, define an activation for each layer.
        if hidden_dims:
            # If a single activation is provided as string, replicate it for every hidden layer.
            if isinstance(activation, str):
                activations = [activation] * len(hidden_dims)
            # If a list of activations is provided, ensure its length matches the hidden layers count.
            elif isinstance(activation, list):
                assert len(activation) == len(hidden_dims), \
                    "Number of activation functions must equal the number of hidden layers."
                activations = activation
            else:
                raise TypeError("activation must be either a string or a list of strings.")

            layers = []
            # Add input layer and its activation.
            layers.append(torch.nn.Linear(self.input_dim, self.hidden_dims[0]))
            layers.append(getattr(torch.nn, activations[0])())

            # Add subsequent hidden layers with corresponding activations.
            for i in range(1, len(self.hidden_dims)):
                layers.append(torch.nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
                layers.append(getattr(torch.nn, activations[i])())

            # Add the final output layer (without an activation).
            layers.append(torch.nn.Linear(self.hidden_dims[-1], self.output_dim))
            self.network = torch.nn.Sequential(*layers)
        else:
            # No hidden layers provided; only the output layer is created.
            self.network = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.output_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        return self.network(x)