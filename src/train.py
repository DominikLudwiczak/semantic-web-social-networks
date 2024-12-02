from models.nn_classifier import NNClassifier
from models.random_forest import RandomForest
from nn_trainer import NNTrainer
from models.svm_classifier import SVMClassifier
from models.gradient_boosting import GradientBoosting
from models.logistic_regression import LogisticRegressionModel


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

        # print("\nSUPPORT VECTOR MACHINE")
        # svm = SVMClassifier(self.data_manager)
        # svm.train()
        # svm.statistics()

        print("\nGRADIENT BOOSTING")
        gb = GradientBoosting(self.data_manager)
        gb.train()
        gb.statistics()

        print("\nLOGISTIC REGRESSION")
        lr = LogisticRegressionModel(self.data_manager)
        lr.train()
        lr.statistics()
