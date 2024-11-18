import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.preprocessing.create_features import create_features


class DataManager:
    def __init__(self):
        self.create_dataloaders()

    def create_dataloaders(self):
        X, y = create_features()
        X = X.drop(columns=["description", "user_created_at"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = X_train.astype(int)
        X_test = X_test.astype(int)

        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        self.train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_test_dataloader(self):
        return self.test_dataloader