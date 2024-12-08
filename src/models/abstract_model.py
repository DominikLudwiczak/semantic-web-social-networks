from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self, data_manager):
        self.data_manager = data_manager

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def statistics(self):
        pass
