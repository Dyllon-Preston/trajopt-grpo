import pytest
import torch
from models import NeuralNetwork

@pytest.mark.parametrize("input_dim, output_dim, hidden_dims, activation", [
    (4, 2, [8, 6], 'ReLU'),  # Standard case
    (10, 3, [16, 8, 4], 'Tanh'),  # Deeper network
    (5, 1, [10], 'Sigmoid'),  # Single hidden layer
    (2, 2, [], 'ReLU'),  # No hidden layers
    (3, 1, [6, 4], ['ReLU', 'Tanh']),  # List of activations
])
def test_network_construction(input_dim, output_dim, hidden_dims, activation):
    """Tests if the network is correctly initialized and structured."""
    model = NeuralNetwork(input_dim, output_dim, hidden_dims, activation)

    # Verify the number of layers
    expected_layer_count = 2*len(hidden_dims) + 1  # Each hidden layer has Linear + Activation
    assert len(model.network) == expected_layer_count, \
        f"Expected {expected_layer_count} layers, but got {len(model.network)}."

    # Check input and output sizes
    assert model.network[0].in_features == input_dim, "Input layer size mismatch."
    assert model.network[-1].out_features == output_dim, "Output layer size mismatch."

@pytest.mark.parametrize("activation", ['ReLU', 'Tanh', 'Sigmoid'])
def test_activation_functions(activation):
    """Tests if the correct activation functions are used in the network."""
    model = NeuralNetwork(4, 2, [8, 6], activation)

    for i, layer in enumerate(model.network):
        if isinstance(layer, torch.nn.Linear):
            continue
        assert layer.__class__.__name__ == activation, \
            f"Expected {activation}, but found {layer.__class__.__name__} at index {i}."

def test_forward_pass():
    """Tests if the model produces an output of correct shape."""
    model = NeuralNetwork(5, 2, [10, 6], 'ReLU')
    x = torch.randn(3, 5)  # Batch of 3 samples, input_dim = 5
    y = model(x)
    
    assert y.shape == (3, 2), f"Expected output shape (3,2), but got {y.shape}."

def test_no_hidden_layers():
    """Tests behavior when no hidden layers are provided."""
    model = NeuralNetwork(4, 2, [], 'ReLU')
    assert len(model.network) == 1, "Model should only contain the output layer."
    assert isinstance(model.network[0], torch.nn.Linear), "Single output layer should be Linear."

def test_invalid_activation_length():
    """Tests that an error is raised when incorrect activation list length is given."""
    with pytest.raises(AssertionError, match="Number of activation functions must equal the number of hidden layers."):
        NeuralNetwork(3, 1, [6, 4], ['ReLU'])  # Only one activation for two hidden layers

