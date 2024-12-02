from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class SVMClassifier:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def train(self):
        self.model = SVC(kernel="linear")
        X_train, y_train = self.data_manager.get_train_dataset()
        self.model.fit(X_train, y_train)

    def statistics(self):
        X_test, y_test = self.data_manager.get_test_dataset()
        y_pred = self.model.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
