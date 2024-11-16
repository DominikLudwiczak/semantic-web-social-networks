from torch.utils.data import TensorDataset
import torch
from preprocessing.create_features import create_features
from IPython.display import display
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.nn_classifier import NNClassifier
from src.train import Train


def main():
    X, y = create_features()
    X = X.drop(columns=["description", "user_created_at"])
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    nn_model = NNClassifier()
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]

        X_train = X_train.astype(int)
        X_test = X_test.astype(int)

        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        trainer = Train(train_dataset, test_dataset, nn_model, "nn_classifier", epochs=1)
        trainer.train()


    # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # trainer = Train(train_dataset, test_dataset, rf_model, "rf_model", epochs=1)
    # trainer.train()
    #
    # rf_model.fit(X_train, y_train)
    # y_pred = rf_model.predict(X_test)
    #
    # cm = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:")
    # print(cm)
    #
    # report = classification_report(y_test, y_pred)
    # print("Classification Report:")
    # print(report)
    #
    # importances = rf_model.feature_importances_
    #
    # feature_importance_df = pd.DataFrame(
    #     {"Feature": X.columns, "Importance": importances}
    # )
    #
    # feature_importance_df = feature_importance_df.sort_values(
    #     by="Importance", ascending=False
    # )
    #
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=feature_importance_df, x="Importance", y="Feature")
    # plt.title("Feature Importances")
    # plt.xlabel("Importance Score")
    # plt.ylabel("Features")
    # plt.savefig("xd.png")


if __name__ == "__main__":
    main()
