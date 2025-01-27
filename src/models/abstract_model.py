from abc import ABC, abstractmethod
from data_manager import DataManager


class AbstractModel(ABC):
    """Abstract wrapper for classification models"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def statistics(self):
        pass
