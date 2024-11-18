from src.models.nn_classifier import NNClassifier
from src.models.random_forest import RandomForest
from src.nn_trainer import NNTrainer


class Train:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def train(self):
        print("SIMPLE NN CLASSIFIER")
        trainer = NNTrainer(NNClassifier(), "simple_nn_classifier")
        trainer.train(self.data_manager, epochs=5)

        print("\nRANDOM FOREST")
        rf = RandomForest(self.data_manager)
        rf.train()
        rf.statistics()
        