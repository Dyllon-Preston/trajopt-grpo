import torch
from torch.distributions import MultivariateNormal
from models import NeuralNetwork

class GaussianActor_NeuralNetwork(NeuralNetwork):
    """
    A Gaussian actor that utilizes a fully-connected neural network 
    to compute actions. A Gaussian distribution is constructed over the 
    action space to encourage exploration using the provided covariance matrix.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, activation: str = 'ReLU'):
        """
        Initializes the GaussianActor_NeuralNetwork.

        Parameters:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_dims (list): Sizes of each hidden layer.
            activation (str, optional): Activation function (as defined in torch.nn). Defaults to 'ReLU'.
        """
        super().__init__(input_dim, output_dim, hidden_dims, activation)

    def action(self, state, cov):
        """
        Compute the action and its log probability by combining the network output 
        with a Gaussian distribution. The covariance matrix controls the extent of noise, 
        thereby facilitating exploration.

        Parameters:
            state (np.ndarray or torch.Tensor): The current state.
            cov (torch.Tensor): The covariance matrix dictating the exploration noise.

        Returns:
            tuple: 
                - mean_action (torch.Tensor): The network's output, representing the mean action.
                - log_prob (torch.Tensor): Log probability of the mean action under the Gaussian distribution.
        """
        # Compute the mean action through a forward pass of the neural network.
        mean_action = self.forward(state)
        
        # Create a Gaussian distribution with the computed mean and provided covariance matrix.
        dist = MultivariateNormal(mean_action, cov)
        
        # Evaluate the log probability of taking the mean action.
        log_prob = dist.log_prob(mean_action)

        return mean_action, log_prob.detach().numpy()  # Detach gradients and convert to NumPy array for compatibility.