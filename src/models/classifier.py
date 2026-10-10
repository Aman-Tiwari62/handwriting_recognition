import torch
import torch.nn as nn


class CharacterClassifier(nn.Module):
    def __init__(self, input_size=512, num_classes=78):
        super().__init__()

        self.linear = nn.Linear(
            input_size,
            num_classes
        )

    def forward(self, x):
        return self.linear(x)