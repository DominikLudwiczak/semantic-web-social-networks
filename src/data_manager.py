import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from preprocessing.create_features import create_features


class DataManager:
    """Data manager used to create dataloaders"""

    def __init__(self):
        self.create_dataloaders()

    def create_dataloaders(self):
        X, y = create_features()
        X = X.drop(columns=["description", "user_created_at"])
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.X_train = X_train.astype(int)
        self.X_test = X_test.astype(int)

        X_train = torch.tensor(self.X_train.values, dtype=torch.float32)
        X_test = torch.tensor(self.X_test.values, dtype=torch.float32)
        y_train = torch.tensor(self.y_train, dtype=torch.float32)
        y_test = torch.tensor(self.y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        self.train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader

    def get_test_dataloader(self) -> DataLoader:
        return self.test_dataloader

    def get_train_dataset(self):
        return self.X_train, self.y_train

    def get_test_dataset(self):
        return self.X_test, self.y_test
