from abc import ABC, abstractmethod

class Algorithm(ABC):
    """
    Abstract class for an algorithm.
    """
    def __init__(self):
        pass

    @abstractmethod
    def learn(self):
        """
        Abstract method for learning.
        """
        pass

    @abstractmethod
    def metadata(self):
        """
        Return metadata about the algorithm.
        """
        return {}
    
    @abstractmethod
    def save(self, path: str):
        """
        Save the algorithm to a file.
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        Load the algorithm from a file.
        """
        pass