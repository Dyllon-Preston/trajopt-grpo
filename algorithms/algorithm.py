from abc import ABC, abstractmethod

class Algorithm(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def learn(self):
        """
        Abstract method for learning.
        """
        pass