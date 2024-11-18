from src.models.nn_classifier import NNClassifier
from src.nn_trainer import NNTrainer


class Train:
    def __init__(self, train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        trainer = NNTrainer(NNClassifier(), "simple_nn_classifier", epochs=5)
        trainer.train(self.train_dataloader, self.test_dataloader)
        