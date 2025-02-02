import pytest
import torch
import numpy as np
from policies import GaussianActor_NeuralNetwork

@pytest.mark.parametrize("input_dim, output_dim, hidden_dims, activation", [
    (4, 2, [8, 6], 'ReLU'),   # Standard case
    (10, 3, [16, 8, 4], 'Tanh'),  # Deep network
    (3, 1, [6], 'Sigmoid'),  # Single hidden layer
    (5, 2, [], 'ReLU'),  # No hidden layers
])
def test_network_initialization(input_dim, output_dim, hidden_dims, activation):
    """Tests if the network is correctly initialized."""
    model = GaussianActor_NeuralNetwork(input_dim, output_dim, hidden_dims, activation)

    # Check input and output layer dimensions
    assert model.network[0].in_features == input_dim, "Mismatch in input layer dimensions."
    assert model.network[-1].out_features == output_dim, "Mismatch in output layer dimensions."

@pytest.mark.parametrize("batch_size, input_dim, output_dim", [
    (1, 4, 2),
    (5, 6, 3),
    (10, 8, 4),
])
def test_action_output_shape(batch_size, input_dim, output_dim):
    """Tests if the action() method returns correctly shaped outputs."""
    model = GaussianActor_NeuralNetwork(input_dim, output_dim, [16, 8], 'ReLU')
    
    state = torch.randn(batch_size, input_dim)
    cov = torch.eye(output_dim)  # Identity covariance for simplicity

    mean_action, log_prob = model.action(state, cov)

    assert mean_action.shape == (batch_size, output_dim), \
        f"Expected action shape ({batch_size}, {output_dim}), got {mean_action.shape}."
    assert log_prob.shape == (batch_size,), \
        f"Expected log_prob shape ({batch_size},), got {log_prob.shape}."

def test_log_prob_validity():
    """Tests that log probability values are finite (not NaN or inf)."""
    model = GaussianActor_NeuralNetwork(4, 2, [8, 6], 'ReLU')

    state = torch.randn(3, 4)  # Batch of 3 states
    cov = torch.eye(2)  # Identity covariance

    _, log_prob = model.action(state, cov)

    # is finite checks for NaN and Inf values
    assert np.all(np.isfinite(log_prob)), "Log probabilities contain NaN or Inf values."
    

@pytest.mark.parametrize("covariance", [
    torch.eye(2),  # Valid identity covariance
    torch.diag(torch.tensor([0.5, 0.2])),  # Valid diagonal covariance
])
def test_valid_covariances(covariance):
    """Tests that valid covariance matrices work without error."""
    model = GaussianActor_NeuralNetwork(4, 2, [8, 6], 'ReLU')
    state = torch.randn(3, 4)
    try:
        model.action(state, covariance)
    except Exception as e:
        pytest.fail(f"Valid covariance matrix caused an error: {e}")

@pytest.mark.parametrize("invalid_cov", [
    torch.zeros(2, 2),  # Zero covariance matrix (invalid)
    torch.ones(2, 2),  # Non-positive-definite covariance
    torch.tensor([[1.0, 0.5], [0.5, -1.0]]),  # Negative eigenvalue (invalid)
])
def test_invalid_covariances(invalid_cov):
    """Tests that invalid covariance matrices raise an error."""
    model = GaussianActor_NeuralNetwork(4, 2, [8, 6], 'ReLU')
    state = torch.randn(3, 4)

    with pytest.raises(ValueError, match="covariance_matrix"):
        model.action(state, invalid_cov)

