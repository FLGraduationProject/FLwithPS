import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 200),
            nn.Tanh(),
            nn.Linear(200, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        out = self.model(x)
        return out