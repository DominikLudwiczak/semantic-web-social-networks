import torch
from torch.nn import BCELoss
from torch.optim import Adam
from tqdm import tqdm
from data_manager import DataManager
from torch.utils.data import DataLoader


class NNTrainer:
    """
    Wrapper for training and evaluating models
    """

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.criterion = BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.01)

    def train(self, data_maneger: DataManager, epochs: int = 10):
        """Train model initialized in class"""
        self.model.to(self.device)
        train_losses, validation_losses = [], []
        train_dataloader = data_maneger.get_train_dataloader()
        test_dataloader = data_maneger.get_test_dataloader()

        for epoch in range(epochs):
            self.model.train()
            running_losses = self.__train_one_epoch(train_dataloader)
            train_losses.append(sum(running_losses) / len(running_losses))

            validation_loss, accuracy = self.evaluate(test_dataloader)
            validation_losses.append(validation_loss)

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train loss: {train_losses[-1]}")
            print(f"Validation loss: {validation_losses[-1]}")
            print(f"Validation accuracy: {accuracy}")
            print("-" * 10)

        torch.save(
            self.model.state_dict(),
            f"src/results/{self.model_name}.pth",
        )

    def __train_one_epoch(self, train_dataloader: DataLoader):
        running_losses = []
        for batch, (X, labels) in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader), desc="Training"
        ):
            X, labels = X.to(self.device), labels.to(self.device)
            outputs = self.model(X).squeeze(1)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_losses.append(loss.item())
        return running_losses

    def evaluate(self, test_dataloader: DataLoader):
        """Evaluate initialized model"""
        self.model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for batch, (X, labels) in tqdm(
                enumerate(test_dataloader),
                total=len(test_dataloader),
                desc="Validating",
            ):
                X, labels = X.to(self.device), labels.to(self.device)
                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                correct += (outputs.round() == labels).sum().item()

        accuracy = correct / len(test_dataloader.dataset)
        return test_loss / len(test_dataloader.dataset), accuracy
