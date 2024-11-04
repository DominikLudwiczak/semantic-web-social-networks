from preprocessing.create_features import create_features
from IPython.display import display
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    X, y = create_features()
    X = X.drop(columns=["description", "user_created_at"])
    print(X.head())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    importances = rf_model.feature_importances_

    feature_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": importances}
    )

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x="Importance", y="Feature")
    plt.title("Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.savefig("xd.png")


if __name__ == "__main__":
    main()
