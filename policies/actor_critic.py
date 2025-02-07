import torch
from torch.distributions import MultivariateNormal
from models import NeuralNetwork
from typing import Union
from abc import ABC, abstractmethod
import numpy as np

class ActorCritic(ABC):
    """
    Abstract base class for Actor-Critic methods.
    """
    @abstractmethod
    def forward(self, state):
        """
        Abstract method to be implemented by subclasses.
        """
        pass
    @abstractmethod
    def parameters(self):
        """
        Abstract method to be implemented by subclasses.
        """
        pass
    def __call__(self, state):
        return self.forward(state)

class RandomUniformActorCritic(ActorCritic):
    """
    A random actor critic that generates actions uniformly between -1 and 1.
    """

    def __init__(self, action_dim: int):
        """
        Initializes the RandomUniformActor.

        Parameters:
            action_dim (int): The dimensionality of the action space.
        """
        self.action_dim = action_dim
    
    def forward(self, state):
        """
        Generate a random action uniformly between -1 and 1.

        Parameters:
            state (np.ndarray or torch.Tensor): The current state.
            cov (torch.Tensor): Ignored in this implementation.

        Returns:
            tuple: 
                - action (torch.Tensor): A tensor representing the randomly generated action.
                - log_prob (torch.Tensor): Log probability of the action, set to 0 as it's uniform.
                - value (torch.Tensor): A tensor representing the value, set to 0.
        """
        # Generate a random action uniformly between -1 and 1
        action = torch.rand(self.action_dim) * 2 - 1
        
        # Log probability of the action is zero for uniform distribution
        log_prob = torch.zeros(1)

        # Value is set to zero
        value = torch.zeros(1)
        
        return action, log_prob, value
    
    def parameters(self):
        """
        Returns an empty list as there are no parameters to optimize.
        """
        return []
    
class GaussianActor_NeuralNetwork(ActorCritic):
    """
    A Gaussian actor that utilizes a fully-connected neural network 
    to compute actions. A Gaussian distribution is constructed over the 
    action space to encourage exploration using the provided covariance matrix.
    """
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            hidden_dims: Union[list, tuple], 
            activation: str = 'ReLU',
            cov: Union[list, float] = 0.1):
        """
        Initializes the GaussianActor_NeuralNetwork.

        Parameters:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_dims (list): Sizes of each hidden layer.
            activation (str, optional): Activation function (as defined in torch.nn). Defaults to 'ReLU'.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        if isinstance(cov, list):
            self.cov = torch.diag(torch.tensor(cov))
        else:
            self.cov = torch.diag(torch.tensor([cov] * output_dim))

        self.actor = NeuralNetwork(input_dim, output_dim, hidden_dims, activation)

    def forward(self, state):
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
                - None: actor does not return a value.
        """

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        # Compute the mean action through a forward pass of the neural network.
        mean_action = self.actor(state)
        
        # Create a Gaussian distribution with the computed mean and provided covariance matrix.
        dist = MultivariateNormal(mean_action, self.cov)
        
        action = dist.sample()

        # Evaluate the log probability of taking the mean action.
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob, None
    
    def log_prob(self, observation, action):
        """

        Compute the log probability of an action given an observation.

        Parameters:
            observation (torch.Tensor): The observation for which to compute the log probability.
            action (torch.Tensor): The action for which to compute the log probability.
        Returns:
            torch.Tensor: Log probability of the action.
       
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        mean_action = self.actor(observation)
        dist = MultivariateNormal(mean_action, self.cov)
        return dist.log_prob(action)
    
    def value(self, state):
        """
        Compute the value of a given state.

        Parameters:
            state (torch.Tensor): The state for which to compute the value.

        Returns:
            torch.Tensor: Value of the state.
        """
        return [None]*state.shape[0]
    
    def parameters(self):
        """
        Returns the parameters of the actor network for optimization.
        """
        return self.actor.parameters()
    
    def state_dict(self):
        """
        Returns the state dictionary of the actor network.
        """
        return self.actor.state_dict()
    
    def load_state_dict(self, state_dict):
        """
        Load the state dictionary into the actor network.
        """
        self.actor.load_state_dict(state_dict)

    def metadata(self):
        """
        Returns metadata about the actor network.

        Returns:
            dict: A dictionary containing metadata about the actor network.
        """
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'cov': self.cov.tolist() if isinstance(self.cov, torch.Tensor) else self.cov,
            'num_parameters': sum(p.numel() for p in self.actor.parameters()),
        }

    def save(self, save_path):
        """
        Save the actor network's state dictionary to a file.

        Parameters:
            save_path (str): The path to the file where the state dictionary will be saved.
        """
        torch.save(self.actor.state_dict(), save_path + 'model.pt')

        metadata = self.metadata()
        with open(save_path + 'metadata.txt', 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")



class GaussianActorCritic_NeuralNetwork(ActorCritic):
    """
    A Gaussian actor-critic that utilizes a fully-connected neural network 
    to compute actions and values. A Gaussian distribution is constructed over the 
    action space to encourage exploration using the provided covariance matrix.
    """
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            hidden_dims: Union[list, tuple], 
            activation: str = 'ReLU',
            cov: Union[list, float] = 0.1):
        """
        Initializes the GaussianActorCritic_NeuralNetwork.

        Parameters:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_dims (list): Sizes of each hidden layer.
            activation (str, optional): Activation function (as defined in torch.nn). Defaults to 'ReLU'.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        if isinstance(cov, list):
            self.cov = torch.diag(torch.tensor(cov))
        else:
            self.cov = torch.diag(torch.tensor([cov] * output_dim))

        self.actor = NeuralNetwork(input_dim, output_dim, hidden_dims, activation)
        self.critic = NeuralNetwork(input_dim, 1, hidden_dims, activation)

    def forward(self, state):
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
                - value (torch.Tensor): The value of the state.
        """

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        # Compute the mean action through a forward pass of the neural network.
        mean_action = self.actor(state)
        
        # Create a Gaussian distribution with the computed mean and provided covariance matrix.
        dist = MultivariateNormal(mean_action, self.cov)
        
        action = dist.sample()

        # Evaluate the log probability of taking the mean
        log_prob = dist.log_prob(action)

        # Compute the value of the state
        value = self.critic(state)

        return action.detach().numpy(), log_prob, value
    
    def log_prob(self, observation, action):
        """

        Compute the log probability of an action given an observation.

        Parameters:
            observation (torch.Tensor): The observation for which to compute the log probability.
            action (torch.Tensor): The action for which to compute the log probability.
        Returns:
            torch.Tensor: Log probability of the action.
       
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        mean_action = self.actor(observation)
        dist = MultivariateNormal(mean_action, self.cov)
        return dist.log_prob(action)
    
    def value(self, state):
        """
        Compute the value of a given state.

        Parameters:
            state (torch.Tensor): The state for which to compute the value.

        Returns:
            torch.Tensor: Value of the state.
        """
        return self.critic(state).squeeze()
    
    def parameters(self):
        """
        Returns the parameters of the actor and critic networks for optimization.
        """
        return list(self.actor.parameters()) + list(self.critic.parameters())
    
    def state_dict(self):
        """
        Returns the state dictionary of the actor and critic networks.
        """
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """
        Load the state dictionary into the actor and critic networks.
        """
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])

    def metadata(self):
        """
        Returns metadata about the actor and critic networks.

        Returns:
            dict: A dictionary containing metadata about the actor and critic networks.
        """
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'cov': self.cov.tolist() if isinstance(self.cov, torch.Tensor) else self.cov,
            'num_parameters': sum(p.numel() for p in self.parameters()),
        }
    
    def save(self, save_path):
        """
        Save the actor and critic networks' state dictionaries to a file.

        Parameters:
            save_path (str): The path to the file where the state dictionaries will be saved.
        """
        torch.save(self.state_dict(), save_path + 'model.pt')

        metadata = self.metadata()
        with open(save_path + 'metadata.txt', 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
                
