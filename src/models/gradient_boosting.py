from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_manager import DataManager


class GradientBoosting:
    """Wrapper for gradient boosting approach matcing project pipeline"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def train(self):
        self.model = GradientBoostingClassifier(n_estimators=10)
        X_train, y_train = self.data_manager.get_train_dataset()
        self.model.fit(X_train, y_train)

    def statistics(self):
        X_test, y_test = self.data_manager.get_test_dataset()
        y_pred = self.model.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
