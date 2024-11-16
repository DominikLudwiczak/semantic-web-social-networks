import torch
from torch.nn import BCELoss
from torch.optim import Adam

class Train:
    def __init__(self, train_data, test_data,  model, model_name, epochs=10):
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.model_name = model_name
        self.epochs = epochs
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.criterion = BCELoss()
        self.optimizer = Adam(model.parameters(), lr=0.2)

    def train(self):
        self.model.to(self.device)
        train_losses, validation_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            running_losses = self.train_one_epoch()
            train_losses.append(sum(running_losses) / len(running_losses))

            validation_loss, accuracy = self.evaluate()
            validation_losses.append(validation_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"Train loss: {train_losses[-1]}")
            print(f"Validation loss: {validation_losses[-1]}")
            print(f"Validation accuracy: {accuracy}")
            print("-" * 10)

        torch.save(
            self.model.state_dict(),
            f"results/{self.model_name}.pth",
        )

    def train_one_epoch(self):
        running_losses = []
        for X, y in self.train_data:
            X, y = X.to(self.device), y.to(self.device)
            output = self.model(X)[0]
            loss = self.criterion(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_losses.append(loss.item())
        return running_losses


    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for X, y in self.test_data:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)[0]
                loss = self.criterion(output, y)
                test_loss += loss.item()
                correct += 1 if output == y else 0

        accuracy = correct / len(self.test_data)
        return test_loss / len(self.test_data), accuracy
        