from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
from data_manager import DataManager


class RandomForest:
    """Wrapper for random forest approach matcing project pipeline"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def train(self):
        self.model = RandomForestClassifier(n_estimators=100)
        X_train, y_train = self.data_manager.get_train_dataset()
        self.model.fit(X_train, y_train)

    def statistics(self):
        X_test, y_test = self.data_manager.get_test_dataset()
        y_pred = self.model.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        importances = self.model.feature_importances_

        feature_importance_df = pd.DataFrame(
            {"Feature": X_test.columns, "Importance": importances}
        )

        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x="Importance", y="Feature")
        plt.title("Feature Importances")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.savefig("feature_importances.png")
