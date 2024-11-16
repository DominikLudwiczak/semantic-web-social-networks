import torch.nn as nn
import torch

class NNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(42, 20)
        self.fc2 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
