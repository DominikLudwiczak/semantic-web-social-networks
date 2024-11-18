from torch.utils.data import TensorDataset, DataLoader
import torch
from preprocessing.create_features import create_features
from sklearn.model_selection import train_test_split
from src.models.nn_classifier import NNClassifier
from src.train import Train


def main():
    X, y = create_features()
    X = X.drop(columns=["description", "user_created_at"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nn_model = NNClassifier()
    trainer = Train(nn_model, "nn_classifier", epochs=5)

    X_train = X_train.astype(int)
    X_test = X_test.astype(int)

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    trainer.train(train_dataloader, test_dataloader)

if __name__ == "__main__":
    main()
