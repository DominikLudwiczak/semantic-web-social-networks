import torch.nn as nn


class NNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(42, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.sigmoid(x)
        return x
